# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os, sys
import random

from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler, SMOTE

from time import strftime, localtime
import torch.nn.functional as F
import numpy as np
import torch
from torch import device
from transformers.optimization import AdamW
from transformers.models.bert.modeling_bert import BertModel
from transformers import BertTokenizer
from seqeval.metrics import classification_report
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from utils.data_utils import ATEPCProcessor, convert_examples_to_features
from model.lcf_atepc import LCF_ATEPC

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

os.makedirs('logs', exist_ok=True)
time = '{}'.format(strftime("%y%m%d-%H%M%S", localtime()))
log_file = 'logs/{}.log'.format(time)
logger.addHandler(logging.FileHandler(log_file))
logger.info('log file: {}'.format(log_file))


def main(config):
    args = config

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processor = ATEPCProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1

    args.bert_model = "bert-base-uncased"
    args.data_dir = "atepc_datasets/restaurant"

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    train_examples = processor.get_train_examples(args.data_dir)
    eval_examples = processor.get_test_examples(args.data_dir)
    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    bert_base_model = BertModel.from_pretrained(args.bert_model)
    bert_base_model.config.num_labels = num_labels

    model = LCF_ATEPC(bert_base_model, args=args)

    for arg in vars(args):
        logger.info('>>> {0}: {1}'.format(arg, getattr(args, arg)))

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.00001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.00001}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=0.00001)
    eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length,
                                                 tokenizer)
    all_spc_input_ids = torch.tensor([f.input_ids_spc for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    all_polarities = torch.tensor([f.polarities for f in eval_features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
    all_emotions = torch.tensor([f.emotions for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_spc_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                              all_polarities, all_valid_ids, all_lmask_ids, all_emotions)  # Modify this line
    # Run prediction for full data
    eval_sampler = RandomSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    def evaluate(eval_ATE=True, eval_APC=True, eval_emotion=True):
        apc_result = {'max_apc_test_acc': 0, 'max_apc_test_f1': 0}
        ate_result = 0
        emotion_result = {'max_emotion_test_acc': 0, 'max_emotion_test_f1': 0}
        y_true = []
        y_pred = []
        n_test_correct, n_test_total = 0, 0
        n_e_test_correct,n_e_test_total = 0,0
        test_apc_logits_all, test_polarities_all = None, None
        test_emotion_logits_all, test_emotions_all = None, None
        model.eval()
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        for input_ids_spc, input_mask, segment_ids, label_ids, polarities, valid_ids, l_mask, emotions in eval_dataloader:
            input_ids_spc = input_ids_spc.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            valid_ids = valid_ids.to(device)
            label_ids = label_ids.to(device)
            polarities = polarities.to(device)
            l_mask = l_mask.to(device)
            emotions = emotions.to(device)

            with torch.no_grad():
                ate_logits, apc_logits, emotion_logits = model(input_ids_spc, segment_ids, input_mask,
                                                               valid_ids=valid_ids, polarities=polarities,
                                                               attention_mask_label=l_mask, emotions=emotions)
            if eval_APC:
                polarities = model.get_batch_polarities(polarities)
                n_test_correct += (torch.argmax(apc_logits, -1) == polarities).sum().item()
                n_test_total += len(polarities)

                if test_polarities_all is None:
                    test_polarities_all = polarities
                    test_apc_logits_all = apc_logits
                else:
                    test_polarities_all = torch.cat((test_polarities_all, polarities), dim=0)
                    test_apc_logits_all = torch.cat((test_apc_logits_all, apc_logits), dim=0)
            if eval_emotion:
                emotions = model.get_batch_emotions(emotions)
                # emotion_logits = emotion_logits.view(emotions.shape)
                n_e_test_correct += (torch.argmax(emotion_logits, -1) == emotions).sum().item()
                n_e_test_total += len(emotions)

                if test_emotions_all is None:
                    test_emotions_all = emotions
                    test_emotion_logits_all = emotion_logits
                else:
                    test_emotions_all = torch.cat((test_emotions_all, emotions), dim=0)
                    test_emotion_logits_all = torch.cat((test_emotion_logits_all, emotion_logits), dim=0)
            if eval_ATE:
                if not args.use_bert_spc:
                    label_ids = model.get_batch_token_labels_bert_base_indices(label_ids)
                ate_logits = torch.argmax(F.log_softmax(ate_logits, dim=2), dim=2)
                ate_logits = ate_logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                input_mask = input_mask.to('cpu').numpy()
                for i, label in enumerate(label_ids):
                    temp_1 = []
                    temp_2 = []
                    for j, m in enumerate(label):
                        if j == 0:
                            continue
                        elif label_ids[i][j] == len(label_list):
                            y_true.append(temp_1)
                            y_pred.append(temp_2)
                            break
                        else:
                            temp_1.append(label_map.get(label_ids[i][j], 'O'))
                            temp_2.append(label_map.get(ate_logits[i][j], 'O'))
        if eval_APC:
            test_acc = n_test_correct / n_test_total
            if args.dataset in {'camera', 'car', 'phone', 'notebook'}:
                test_f1 = f1_score(torch.argmax(test_apc_logits_all, -1).cpu(), test_polarities_all.cpu(),
                                   labels=[0, 1], average='macro')
            else:
                test_f1 = f1_score(torch.argmax(test_apc_logits_all, -1).cpu(), test_polarities_all.cpu(),
                                   labels=[0, 1, 2], average='macro')
            test_acc = round(test_acc * 100, 2)
            test_f1 = round(test_f1 * 100, 2)
            apc_result = {'max_apc_test_acc': test_acc, 'max_apc_test_f1': test_f1}

        if eval_ATE:
            report = classification_report(y_true, y_pred, digits=4)
            tmps = report.split()
            ate_result = round(float(tmps[7]) * 100, 2)
        if eval_emotion:
            print(test_apc_logits_all)
            print(test_emotions_all)
            # Compute the F1 score
            emotion_f1 = f1_score(test_emotions_all,torch.argmax(test_emotion_logits_all,-1).cpu(), labels=[0, 1, 2, 3, 4, 5], average='macro')
            emotion_acc = accuracy_score(test_emotions_all,torch.argmax(test_emotion_logits_all,-1).cpu())
            emotion_acc = round(emotion_acc * 100, 2)
            emotion_f1 = round(emotion_f1 * 100, 2)
            emotion_result = {'max_emotion_test_acc': emotion_acc, 'max_emotion_test_f1': emotion_f1}
        return apc_result, ate_result, emotion_result
    def save_model(path):
        # Save a trained model and the associated configuration,
        # Take care of the storage!
        os.makedirs(path, exist_ok=True)
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        model_to_save.save_pretrained(path)
        tokenizer.save_pretrained(path)
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        model_config = {"bert_model": args.bert_model, "do_lower": True, "max_seq_length": args.max_seq_length,
                        "num_labels": len(label_list) + 1, "label_map": label_map}
        json.dump(model_config, open(os.path.join(path, "config.json"), "w"))
        logger.info('save model to: {}'.format(path))
    def train():
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_spc_input_ids = torch.tensor([f.input_ids_spc for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
        all_polarities = torch.tensor([f.polarities for f in train_features], dtype=torch.long)
        emotions = [f.emotions for f in train_features]
        all_emotions = torch.tensor(emotions, dtype=torch.long)
        train_data = TensorDataset(all_spc_input_ids, all_input_mask, all_segment_ids,
                                   all_label_ids, all_polarities, all_valid_ids, all_lmask_ids,
                                   all_emotions)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        max_apc_test_acc = 0
        max_apc_test_f1 = 0
        max_ate_test_f1 = 0
        max_emotion_test_acc = 0
        max_emotion_test_f1 = 0

        global_step = 0
        for epoch in range(int(args.num_train_epochs)):
            logger.info('#' * 80)
            logger.info('Train {} Epoch{}'.format(args.seed, epoch + 1, args.data_dir))
            logger.info('#' * 80)
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                input_ids_spc, input_mask, segment_ids, label_ids, polarities, valid_ids, l_mask, emotions = batch

                loss_ate, loss_apc,loss_emo = model(input_ids_spc, segment_ids, input_mask, label_ids, polarities, valid_ids,
                                           l_mask,emotions)
                loss = loss_ate + loss_apc + loss_emo
                loss.backward()
                nb_tr_examples += input_ids_spc.size(0)
                nb_tr_steps += 1
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % args.eval_steps == 0:
                    if epoch >= args.num_train_epochs - 2 or args.num_train_epochs <= 2:
                        # evaluate in last 2 epochs
                        apc_result, ate_result, emotion_result = evaluate(eval_ATE=not args.use_bert_spc,
                                                                          eval_emotion=True)
                        if apc_result['max_apc_test_acc'] > max_apc_test_acc:
                            max_apc_test_acc = apc_result['max_apc_test_acc']
                        if apc_result['max_apc_test_f1'] > max_apc_test_f1:
                            max_apc_test_f1 = apc_result['max_apc_test_f1']
                        if ate_result > max_ate_test_f1:
                            max_ate_test_f1 = ate_result
                        if emotion_result['max_emotion_test_acc'] > max_emotion_test_acc:
                            max_emotion_test_acc = emotion_result['max_emotion_test_acc']
                        if emotion_result['max_emotion_test_f1'] > max_emotion_test_f1:
                            max_emotion_test_f1 = emotion_result['max_emotion_test_f1']

                        current_apc_test_acc = apc_result['max_apc_test_acc']
                        current_apc_test_f1 = apc_result['max_apc_test_f1']
                        current_ate_test_f1 = round(ate_result, 2)
                        current_emotion_test_acc = emotion_result['max_emotion_test_acc']
                        current_emotion_test_f1 = emotion_result['max_emotion_test_f1']

                        logger.info('*' * 80)
                        logger.info('Train {} Epoch{}, Evaluate for {}'.format(args.seed, epoch + 1, args.data_dir))
                        logger.info(f'APC_test_acc: {current_apc_test_acc}(max: {max_apc_test_acc})  '
                                    f'APC_test_f1: {current_apc_test_f1}(max: {max_apc_test_f1})')
                        if args.use_bert_spc:
                            logger.info(f'ATE_test_F1: {current_ate_test_f1}(max: {max_ate_test_f1})'
                                        f' (Unreliable since `use_bert_spc` is "True".)')
                        else:
                            logger.info(f'ATE_test_f1: {current_ate_test_f1}(max:{max_ate_test_f1})')
                        logger.info(
                            f'Emotion_test_acc: {current_emotion_test_acc}(max: {max_emotion_test_acc})'
                            f' Emotion_test_f1: {current_emotion_test_f1}(max: {max_emotion_test_f1})')
                        logger.info('*' * 80)
        return [max_apc_test_acc, max_apc_test_f1, max_ate_test_f1, max_emotion_test_acc,
                max_emotion_test_f1]

    # output_dir = "model_output"
    # save_model_path = os.path.join(output_dir, 'saved_model')
    # save_model(save_model_path)

    return train()


def parse_experiments(path):
    configs = []
    opt = argparse.ArgumentParser()
    with open(path, "r", encoding='utf-8') as reader:
        json_config = json.loads(reader.read())
    for id, config in json_config.items():
        # Hyper Parameters
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", default=config['dataset'], type=str)
        parser.add_argument("--output_dir", default=config['output_dir'], type=str)
        parser.add_argument("--SRD", default=int(config['SRD']), type=int)
        parser.add_argument("--learning_rate", default=float(config['learning_rate']), type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--use_unique_bert", default=bool(config['use_unique_bert']), type=bool)
        parser.add_argument("--use_bert_spc", default=bool(config['use_bert_spc_for_apc']), type=bool)
        parser.add_argument("--local_context_focus", default=config['local_context_focus'], type=str)
        parser.add_argument("--num_train_epochs", default=float(config['num_train_epochs']), type=float,
                            help="Total number of training epochs to perform.")
        parser.add_argument("--train_batch_size", default=int(config['train_batch_size']), type=int,
                            help="Total batch size for training.")
        parser.add_argument("--dropout", default=float(config['dropout']), type=int)
        parser.add_argument("--max_seq_length", default=int(config['max_seq_length']), type=int)
        parser.add_argument("--eval_batch_size", default=32, type=int, help="Total batch size for eval.")
        parser.add_argument("--eval_steps", default=20, help="evaluate per steps")
        parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument("--config_path", default='experiments.json', type=str,
                            help='Path of experiments config file')
        configs.append(parser.parse_args())
    return configs


if __name__ == "__main__":
    experiments = argparse.ArgumentParser()
    experiments.add_argument('--config_path', default='experiments.json', type=str,
                             help='Path of experiments config file')
    experiments = experiments.parse_args()
    # from utils.Pytorch_GPUManager import GPUManager
    # index = GPUManager().auto_choice()
    # device = torch.device("cuda:" + str(index) if torch.cuda.is_available() else "cpu")
    #
    device = torch.device("cpu")
    exp_configs = parse_experiments(experiments.config_path)
    n = 5
    for config in exp_configs:
        logger.info('-' * 80)
        logger.info('Config {} (totally {} configs)'.format(exp_configs.index(config) + 1, len(exp_configs)))
        results = []
        max_apc_test_acc, max_apc_test_f1, max_ate_test_f1 = 0, 0, 0
        for i in range(n):
            config.device = device
            config.seed = i + 1
            logger.info('No.{} training process of {}'.format(i + 1, n))
            apc_test_acc, apc_test_f1, ate_test_f1 = main(config)

            if apc_test_acc > max_apc_test_acc:
                max_apc_test_acc = apc_test_acc
            if apc_test_f1 > max_apc_test_f1:
                max_apc_test_f1 = apc_test_f1
            if ate_test_f1 > max_ate_test_f1:
                max_ate_test_f1 = ate_test_f1
            logger.info('max_ate_test_f1:{} max_apc_test_acc: {}\tmax_apc_test_f1: {} \t'
                        .format(max_ate_test_f1, max_apc_test_acc, max_apc_test_f1))
