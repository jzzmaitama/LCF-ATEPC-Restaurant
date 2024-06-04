# -*- coding: utf-8 -*-
# file: lcf_atepc.py

from transformers.models.bert.modeling_bert import BertForTokenClassification, BertPooler, BertSelfAttention
from torch.nn import CrossEntropyLoss
import torch
import torch.nn as nn
import numpy as np
import copy


class SelfAttention(nn.Module):
    def __init__(self, config, opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_vec = np.zeros((inputs.size(0), 1, 1, self.opt.max_seq_length))
        zero_tensor = torch.tensor(zero_vec).float().to(self.opt.device)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])


class LCF_ATEPC(BertForTokenClassification):
    def __init__(self, bert_base_model, args):
        super(LCF_ATEPC, self).__init__(config=bert_base_model.config)
        config = bert_base_model.config
        self.bert_for_global_context = bert_base_model
        self.args = args
        # Initialize the local context BERT model
        self.bert_for_local_context = copy.deepcopy(
            self.bert_for_global_context) if not self.args.use_unique_bert else self.bert_for_global_context
        self.pooler = BertPooler(config)
        self.num_emotion_labels = 6
        self.dense = torch.nn.Linear(768, 4)  # For aspect categories
        self.emotion_classifier = nn.Linear(config.hidden_size, self.num_emotion_labels)  # 6 for the number of emotions
        self.dropout = nn.Dropout(self.args.dropout)
        self.SA1 = SelfAttention(config, args)
        self.SA2 = SelfAttention(config, args)
        self.linear_double = nn.Linear(768 * 2, 768)
        self.linear_triple = nn.Linear(768 * 3, 768)

    def get_batch_token_labels_bert_base_indices(self, labels):
        if labels is None:
            return
        labels = labels.detach().cpu().numpy()
        for text_i in range(len(labels)):
            sep_index = np.argmax((labels[text_i] == 5))
            labels[text_i][sep_index + 1:] = 0
        return torch.tensor(labels).to(self.args.device)

    def get_batch_emotions(self, b_emotions):
        b_emotions = b_emotions.detach().cpu().numpy()
        shape = b_emotions.shape
        emotions = np.zeros((shape[0], self.args.max_seq_length))
        for i in range(shape[0]):
            emotions_idx = np.flatnonzero(b_emotions[i] + 1)
            try:
                emotions[i, :len(emotions_idx)] = b_emotions[i, emotions_idx[0]]
            except:
                pass
        emotions = torch.from_numpy(emotions).long().to(self.args.device)
        return emotions

    def get_batch_polarities(self, b_polarities):
        b_polarities = b_polarities.detach().cpu().numpy()
        print(b_polarities)
        shape = b_polarities.shape
        print(b_polarities.shape)
        polarities = np.zeros((shape[0]))
        for i, polarity in enumerate(b_polarities):
            polarity_idx = np.flatnonzero(polarity + 1)
            try:
                polarities[i] = polarity[polarity_idx[0]]
            except:
                pass
        return torch.from_numpy(polarities).long().to(self.args.device)



    def feature_dynamic_weighted(self, text_local_indices, polarities):
        text_ids = text_local_indices.detach().cpu().numpy()
        asp_ids = polarities.detach().cpu().numpy()
        weighted_text_raw_indices = np.ones((text_local_indices.size(0), text_local_indices.size(1), 768),
                                            dtype=np.float32)
        SRD = self.args.SRD
        for text_i, asp_i in zip(range(len(text_ids)), range(len(asp_ids))):
            a_ids = np.flatnonzero(asp_ids[asp_i] + 1)
            text_len = np.flatnonzero(text_ids[text_i])[-1] + 1
            asp_len = len(a_ids)
            try:
                asp_begin = a_ids[0]
            except:
                asp_begin = 0
            asp_avg_index = (asp_begin * 2 + asp_len) / 2
            distances = np.zeros((text_len), dtype=np.float32)
            for i in range(len(distances)):
                if abs(i - asp_avg_index) + asp_len / 2 > SRD:
                    distances[i] = 1 - (abs(i - asp_avg_index) + asp_len / 2 - SRD) / len(distances)
                else:
                    distances[i] = 1
            for i in range(len(distances)):
                weighted_text_raw_indices[text_i][i] = weighted_text_raw_indices[text_i][i] * distances[i]
        return torch.from_numpy(weighted_text_raw_indices).to(self.args.device)

    def feature_dynamic_mask(self, text_local_indices, polarities):
        text_ids = text_local_indices.detach().cpu().numpy()
        asp_ids = polarities.detach().cpu().numpy()
        SRD = self.args.SRD
        masked_text_raw_indices = np.ones((text_local_indices.size(0), text_local_indices.size(1), 768),
                                          dtype=np.float32)
        for text_i, asp_i in zip(range(len(text_ids)), range(len(asp_ids))):
            a_ids = np.flatnonzero(asp_ids[asp_i] + 1)
            try:
                asp_begin = a_ids[0]
            except:
                asp_begin = 0
            asp_len = len(a_ids)
            mask_begin = max(0, asp_begin - SRD)
            masked_text_raw_indices[text_i, :mask_begin] = 0
            masked_text_raw_indices[text_i, asp_begin + asp_len + SRD:] = 0
        return torch.from_numpy(masked_text_raw_indices).to(self.args.device)

    def get_ids_for_local_context_extractor(self, text_indices):
        text_ids = text_indices.detach().cpu().numpy()
        for text_i in range(len(text_ids)):
            sep_index = np.argmax((text_ids[text_i] == 102))
            text_ids[text_i][sep_index + 1:] = 0
        return torch.tensor(text_ids).to(self.args.device)

    def forward(self, input_ids_spc, token_type_ids=None, attention_mask=None, labels=None, polarities=None,
                valid_ids=None, attention_mask_label=None, emotions=None):
        if not self.args.use_bert_spc:
            input_ids_spc = self.get_ids_for_local_context_extractor(input_ids_spc)
            labels = self.get_batch_token_labels_bert_base_indices(labels)
        global_context_out = self.bert_for_global_context(input_ids_spc, token_type_ids, attention_mask)[
            'last_hidden_state']
        polarity_labels = self.get_batch_polarities(polarities)
        emotion_labels = self.get_batch_emotions(emotions)

        print("Polarity Labels Unique Values:", torch.unique(polarity_labels))
        print("Polarity Labels Shape:", polarity_labels.shape)

        # Debugging output to check the range and values of emotion labels
        print("Emotion Labels Unique Values:", torch.unique(emotion_labels))
        print("Emotion Labels Shape:", emotion_labels.shape)

        # Ensure emotion labels are within the valid range
        if torch.min(emotion_labels) < 0 or torch.max(emotion_labels) >= self.num_emotion_labels:
            raise ValueError("Emotion labels are out of bounds. They should be in the range [0, {}]".format(
                self.num_emotion_labels - 1))

        batch_size, max_len, feat_dim = global_context_out.shape
        global_valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32).to(self.args.device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    global_valid_output[i][jj] = global_context_out[i][j]
        global_context_out = self.dropout(global_valid_output)
        ate_logits = self.classifier(global_context_out)
        # Add emotion logits
        emotion_logits = self.emotion_classifier(global_context_out)

        if self.args.local_context_focus is not None:
            local_context_ids = self.get_ids_for_local_context_extractor(
                input_ids_spc) if self.args.use_bert_spc else input_ids_spc
            local_context_out = self.bert_for_local_context(input_ids_spc)['last_hidden_state']
            batch_size, max_len, feat_dim = local_context_out.shape
            local_valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32).to(self.args.device)
            for i in range(batch_size):
                jj = -1
                for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        local_valid_output[i][jj] = local_context_out[i][j]
            local_context_out = self.dropout(local_valid_output)

            if 'cdm' in self.args.local_context_focus:
                cdm_vec = self.feature_dynamic_mask(local_context_ids, polarities)
                cdm_context_out = torch.mul(local_context_out, cdm_vec)
                cat_out = torch.cat((global_context_out, cdm_context_out), dim=-1)
                cat_out = self.linear_double(cat_out)
                logits = self.classifier(cat_out)
                emotion_logits = self.emotion_classifier(cat_out)
            elif 'cdw' in self.args.local_context_focus:
                cdw_vec = self.feature_dynamic_weighted(local_context_ids, polarities)
                cdw_context_out = torch.mul(local_context_out, cdw_vec)
                cat_out = torch.cat((global_context_out, cdw_context_out), dim=-1)
                cat_out = self.linear_double(cat_out)
                logits = self.classifier(cat_out)
                emotion_logits = self.emotion_classifier(cat_out)
            elif 'fusion' in self.args.local_context_focus:
                cdm_vec = self.feature_dynamic_mask(local_context_ids, polarities)
                cdm_context_out = torch.mul(local_context_out, cdm_vec)
                cdm_context_out = self.SA1(cdm_context_out)
                cdw_vec = self.feature_dynamic_weighted(local_context_ids, polarities)
                cdw_context_out = torch.mul(local_context_out, cdw_vec)
                cdw_context_out = self.SA2(cdw_context_out)
                cat_out = torch.cat((global_context_out, cdm_context_out, cdw_context_out), dim=-1)
                cat_out = self.linear_triple(cat_out)
                logits = self.classifier(cat_out)
                emotion_logits = self.emotion_classifier(cat_out)
            else:
                logits = ate_logits
        else:
            logits = ate_logits

        outputs = (logits,)
        loss_fct = CrossEntropyLoss(ignore_index=0)
        if labels is not None:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            emotion_loss = loss_fct(emotion_logits.view(-1, self.num_emotion_labels), emotion_labels.view(-1))
            outputs = (loss + emotion_loss,) + outputs

        return outputs