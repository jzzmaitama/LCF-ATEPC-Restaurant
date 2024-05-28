# -*- coding: utf-8 -*-
# file: data_utils.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.

import os
import random

from sklearn.preprocessing import LabelEncoder


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, sentence_label=None, aspect_label=None, polarity=None, emotion=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.sentence_label = sentence_label
        self.aspect_label = aspect_label
        self.polarity = polarity
        self.emotion = emotion


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids_spc, input_mask, segment_ids, label_id, polarities=None, valid_ids=None,
                 label_mask=None, emotions=None):
        self.input_ids_spc = input_ids_spc
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.polarities = polarities
        self.emotions = emotions


def readfile(filename):
    '''
    read file
    '''
    f = open(filename, encoding='utf8')
    data = []
    sentence = []
    tag = []
    polarity = []
    emotion = []  # Add this line
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, tag, polarity, emotion))  # Modify this line
                sentence = []
                tag = []
                polarity = []
                emotion = []  # Add this line
            continue
        splits = line.split(' ')
        if len(splits) != 4:  # Modify this line
            print('warning! detected error line(s) in input file:{}'.format(line))
        sentence.append(splits[0])
        tag.append(splits[-3])  # Modify this line
        polarity.append(int(splits[-2]))  # Modify this line
        emotion.append(splits[-1][:-1])  # Add this line

    if len(sentence) > 0:
        data.append((sentence, tag, polarity, emotion))  # Modify this line
    return data


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


# Initialize the LabelEncoder
le = LabelEncoder()

# Fit the LabelEncoder to the emotions
le.fit(["Joy", "Anger", "Fear", "Sadness", "Surprise", "Disgust", "None"])  # Add all possible emotions


class ATEPCProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "Restaurants.atepc.train.dat")), "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "Restaurants.atepc.test.dat")), "test")

    def get_labels(self):
        # return ["O", "B-ASP", "I-ASP", "[CLS]", "[SEP]", "Anger", "Disgust", "Fear", "Joy", "Sadness", "Surprise"]
        return ["O", "B-ASP", "I-ASP", "[CLS]", "[SEP]"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, tag, polarity, emotion) in enumerate(lines):
            aspect = []
            aspect_tag = []
            aspect_polarity = [-1]
            aspect_emotion = []  # Add this line
            for w, t, p, e in zip(sentence, tag, polarity, emotion):  # Modify this line
                if p != -1:
                    aspect.append(w)
                    aspect_tag.append(t)
                    aspect_polarity.append(-1)
                aspect_emotion.append(e)  # Add this line
            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            text_b = aspect
            polarity.extend(aspect_polarity)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, sentence_label=tag,
                                         aspect_label=aspect_tag, polarity=polarity,
                                         emotion=aspect_emotion))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 1)}

    features = []
    for (ex_index, example) in enumerate(examples):
        text_spc_tokens = example.text_a
        aspect_tokens = example.text_b
        sentence_label = example.sentence_label
        aspect_label = example.aspect_label
        polaritiylist = example.polarity
        emotionlist = example.emotion  # Add this line
        tokens = []
        labels = []
        polarities = []
        emotions = []  # Add this line
        valid = []
        label_mask = []
        text_spc_tokens.extend(['[SEP]'])
        text_spc_tokens.extend(aspect_tokens)
        enum_tokens = text_spc_tokens
        diff = len(enum_tokens) - len(emotionlist)
        if diff > 0:
            emotionlist += emotionlist * (diff // len(emotionlist)) + emotionlist[:diff % len(emotionlist)]
        sentence_label.extend(['[SEP]'])
        emotionlist = list(map(int, emotionlist))
        sentence_label.extend(aspect_label)
        label_lists = sentence_label
        for i, word in enumerate(enum_tokens):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = label_lists[i]
            polarity_1 = polaritiylist[i]
            emotion_1 = emotionlist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    polarities.append(polarity_1)
                    emotions.append(emotion_1)  # Add this line
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            polarities = polarities[0:(max_seq_length - 2)]
            emotions = emotions[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 1)
        label_mask.insert(0, 1)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])
        input_ids_spc = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids_spc)
        label_mask = [1] * len(label_ids)
        while len(input_ids_spc) < max_seq_length:
            input_ids_spc.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        while len(polarities) < max_seq_length:
            polarities.append(-1)
        while len(emotions) < max_seq_length:
            emotions.append(-1)
        assert len(input_ids_spc) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length
        assert len(emotions) == max_seq_length
        features.append(
            InputFeatures(input_ids_spc=input_ids_spc,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          polarities=polarities,
                          emotions=emotions,
                          valid_ids=valid,
                          label_mask=label_mask))

    print(len(features[5].emotions))
    return features
