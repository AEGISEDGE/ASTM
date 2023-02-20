# -*- coding: utf-8 -*-
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# ====================================================================================================================================================
# Symbolic constant
MODEL_SAV_PATH = os.getcwd()+'//sav//'
MAX_TO_KEEP = 5
RAPID_RELOAD_DATALOADER = True
# ====================================================================================================================================================

def get_file_list(dir):
    file_with_path = []
    file_list = os.listdir(dir)
    for f in file_list:
        tmp_filewithpath = dir+'/'+f
        if os.path.isfile(tmp_filewithpath):
            file_with_path.append(tmp_filewithpath)
        elif os.path.isdir(tmp_filewithpath):
            file_with_path += get_file_list(tmp_filewithpath)
    return file_with_path

def draw_curve(x, y, x_label, y_label, title,  save_flg=False):
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_title(title)
    assert len(x) == len(y), "Unequal length of input data"
    ax.set_xlabel(x_label, fontsize=12, fontstyle='italic')
    ax.set_ylabel(y_label, fontsize=12, fontstyle='italic')
    ax.plot(x, y, color='b')
    if save_flg:
        plt.savefig(title+'.png')


def multiline_curve(data_list, x_label, y_label, title, save_flg=False):
    colormap = ['r', 'g', 'b', 'y', 'm', 'c', 'k']
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlabel(x_label, fontsize=12, fontstyle='italic')
    ax.set_ylabel(y_label, fontsize=12, fontstyle='italic')
    for index, row in enumerate(data_list[1:]):
        ax.plot(data_list[0], row, color=colormap[index])
    if save_flg:
        plt.savefig(title + '.png')

def build_wordembedding_from_dict(wordembedding_dict, id2word, n_dim, vocabulary_size):
    vec_list = []
    idlist = list(id2word.keys())
    idlist.sort()
    for idx in idlist:
        word = id2word[idx]
        if word not in wordembedding_dict.keys():
            embedding = wordembedding_dict['unk']
        else:
            embedding = wordembedding_dict[word]
        vec_list.append(embedding)
    return vec_list

def ReadDoc(name):
    fp = open(name, 'r')
    doc = fp.readlines()
    fp.close()
    return doc

def ReadDictionary(vocabpath):
    word2id = {}
    id2word = {}
    vocabulary = {}
    txt = ReadDoc(vocabpath)
    for i in range(0, len(txt)):
        txt_item = txt[i].strip().split(' ')
        if len(txt_item) > 1:
            tmp_list = txt[i].strip().split(' ')
            word2id[tmp_list[0]] = i
            id2word[i] = tmp_list[0]
            vocabulary[tmp_list[0]] = tmp_list[1]
    return word2id, id2word

def Abnormal_value(value, pos_str):
    assert not torch.isinf(value).any(), "We got inf value at "+pos_str
    assert not torch.isnan(value).any(), "We got nan value at "+pos_str
