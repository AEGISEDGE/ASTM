# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import math
import sys
import pickle
import torch

# ====================================================================================================================================================
# Corpus object

class BOW_TopicModel_Corpus(Dataset):
    def __init__(self, vocabulary_size, data_path, device, dev=False, prop=None, loader=None):
        # data_path: path route for train.feat and test.feat
        self.device = device
        self.vocabulary_size = vocabulary_size
        self.doc_vec_set = []
        self.lable_set = []
        self.word_count_set = []
        self.doc_count = 0.0
        doc_index = 0
        with open(data_path, 'r') as f:
            docs = f.readlines()
            for doc in docs:
                item_list = doc.strip().split(' ')
                class_label = item_list[0]
                doc_tokens = item_list[1:]
                vec, word_count = self.tokens2vec(doc_tokens)
                self.doc_vec_set.append(vec)
                self.lable_set.append(class_label)
                self.word_count_set.append(word_count)
                doc_index += 1
        if dev:
            end_num = int(prop * self.__len__())
            self.doc_vec_set = self.doc_vec_set[:end_num]
            self.lable_set = self.lable_set[:end_num]
            self.word_count_set = self.word_count_set[:end_num]
        self.loader = loader
        self.doc_count = doc_index

    def __len__(self):
        return len(self.doc_vec_set)

    def tokens2vec(self, token_list):
        vocabulary_size = self.vocabulary_size
        vec = torch.zeros(vocabulary_size)
        word_count = 0
        for token in token_list:
            # <token index>:tf
            token_index = int(token.split(':')[0])
            token_tf = int(token.split(':')[1])
            word_count += token_tf
            vec[token_index-1] = float(token_tf)
        return vec, word_count

    def __getitem__(self, index):
        return self.doc_vec_set[index], self.lable_set[index], self.word_count_set[index]
