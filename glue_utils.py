from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import random
import sys
from io import open
import xml.etree.ElementTree as ET
from collections import Counter
import pickle
logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd
# import pickle


class InputExamples():
    def __init__(self,tokens,label,domain):
        self.text_a = tokens
        self.text_b = None
        self.label = label
        self.domain = domain

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label, domain,m,ppt):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label
        self.domain = domain

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir,domain_schema):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir,domain_schema):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir,domain_schema):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

class SentProcessor(DataProcessor):
    def get_train_examples(self, data_dir, domain_schema,half=[0]):
        return self._create_examples(data_dir=data_dir, set_type='train', domain_schema=domain_schema,half=half)

    def get_dev_examples(self, data_dir, domain_schema):
        return self._create_examples(data_dir=data_dir, set_type='dev', domain_schema=domain_schema)

    def get_test_examples(self, data_dir, domain_schema):
        return self._create_examples(data_dir=data_dir, set_type='test', domain_schema=domain_schema)

    def _create_examples(self, data_dir, set_type, domain_schema, half = [0]):
        input_examples = []
        print("half = "+str(half) )
        for key in domain_schema.keys():
            dom_features = []
            with open(data_dir + key + '.task.'+set_type, 'r', encoding='ISO-8859-2') as inf:
                line = inf.readline().strip()
                while (line):
                    label, word = line.split('\t')
                    dom_features.append(InputExamples(word, int(label), domain_schema[key]))
                    line = inf.readline().strip()
            if len(half) == 1:
                input_examples.extend(dom_features)
            else:
                input_examples.extend(dom_features[int(len(dom_features)*half[0]):int(len(dom_features)*half[1])])
        return input_examples

class Sent2Processor(DataProcessor):
    def get_examples(self, data_dir,domain_schema):
        # domain_schema = {}
        train,test = [],[]
        for key in domain_schema.keys():
            d_examples = []
            reviews,negReviews,posReviews = XML2arrayRAW(data_dir +'/' +key + '/positive.parsed', data_dir +'/'  + key + '/negative.parsed')
            for a in negReviews:
                a = a.replace('\n','.')
                d_examples.append(InputExamples(a, 0, domain_schema[key]))
            for a in posReviews:
                a = a.replace('\n','.')
                d_examples.append(InputExamples(a, 1, domain_schema[key]))
        return d_examples

    def get_test_examples(self, data_dir,domain_schema):
        # domain_schema = {}
        test = []
        for key in domain_schema.keys():
            d_examples = []
            reviews,negReviews,posReviews = XML2arrayRAW(data_dir +'/' +key + '/positive.parsed', data_dir +'/'  + key + '/negative.parsed')
            for a in negReviews:
                a = a.replace('\n','.')
                d_examples.append(InputExamples(a, 0, domain_schema[key]))
            for a in posReviews:
                a = a.replace('\n','.')
                d_examples.append(InputExamples(a, 1, domain_schema[key]))
            test.extend(d_examples)
        return test

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        tokens_a.pop()



def convert_examples_to_features(examples, max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        ms = [sequence_a_segment_id] * len(tokens)
        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
            m = [1 if tokens_b[i]=='[MASK]' else 0 for i in range(len(tokens_b))]
            ms += (m+[0])
        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
            ms = ms + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids
            ms = [0]+ ms 
        # print(len(ms))
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
            ms = ms + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label=example.label,
                              domain=example.domain))
    return features

def XML2arrayRAW(pos_path,neg_path):
    reviews = []
    negReviews = []
    posReviews = []

    neg_tree = ET.parse(neg_path)
    neg_root = neg_tree.getroot()
    for rev in neg_root.iter('review'):
        reviews.append(rev.text)
        negReviews.append(rev.text)

    pos_tree = ET.parse(pos_path)
    pos_root = pos_tree.getroot()

    for rev in pos_root.iter('review'):
        reviews.append(rev.text)
        posReviews.append(rev.text)

    return reviews,negReviews,posReviews

def get_domain_schema(domains):
    domain_schema = {}
    dom_id = 0
    for dom in domains:
        domain_schema[dom] = dom_id
        dom_id += 1
    return domain_schema
    
