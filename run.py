# -*- coding: utf-8 -*-
import torch
import argparse
import sys
import os
import pickle
import math

from model import ASTM
from dataset import BOW_TopicModel_Corpus

# Requirement:
# matplotlib, seaborn, panda, keops, geomloss

import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from utils import draw_curve, ReadDictionary, build_wordembedding_from_dict
# ====================================================================================================================================================

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

# ====================================================================================================================================================
parser = argparse.ArgumentParser(
    description='Gaussian Softmax Model parameters description.')
parser.add_argument('--n-hidden', type=int, default=512, metavar='N',
                    help="The size of hidden units in doc inference network (default 512)")
parser.add_argument('--dropout', type=float, default=0.5, metavar='N',
                    help="The drop-out probability of MLP (default 0.5)")
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help="The learning rate of model (default 1e-5)")
parser.add_argument('--topics', type=int, default=50, metavar='N',
                    help="The amount of topics to be discover")
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help="Training batch size.")
parser.add_argument('--topicembedsize', type=int, default=100, metavar='N',
                    help="Topic embedding size of topic modelling")
parser.add_argument('--alternative-epoch', type=int, default=5, metavar='N',
                    help="Alternative epoch size for wake sleep algorithm")
parser.add_argument('--rapid', action='store_true', default=True,
                    help="Flag for enable rapid reloading dataset.")
parser.add_argument('--inf-nonlinearity', default='tanh', metavar='N',
                    help="Options for non-linear function.(default tanh)")
parser.add_argument('--data-path', default='data/20news/', metavar='N',
                    help="Directory for corpus. Default is preprocessed Twitter corpus.")
parser.add_argument('--coel', type=float, default=0.1, metavar='N',
                    help="Coefficient for Sinkhorn divergence between prior and posterior.")
parser.add_argument('--coea', type=float, default=5.0, metavar='N',
                    help="Coefficient for Topic Diversity Sinkhorn Regularization.")
parser.add_argument('--topk', type=int, default=10, metavar='N',
                    help="Top-k word for Topic Diversity Sinkhorn Regularization.")

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
# ==================================================================================================================================================


def main(args):
    # Load dataset
    print("===========================================================================================================")
    print('Encoder hidden units: %d , drop-out rate: %f , topic number: %d, coel: %f, coea: %f, topk: %d' %
          (args.n_hidden, args.dropout, args.topics, args.coel, args.coea, args.topk))
    # Rapid reload dataset object:
    word2id, id2word = ReadDictionary(args.data_path + 'vocab.new')
    vocabulary_size = len(word2id.keys())
    # Devices context manager
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Rapid reload dataset from previously dumped binary files
    if os.path.exists("corpus_obj.bin") and args.rapid:
        print("*** Rapid reload dataset object from previously dumped file. ***")
        full_dataset = pickle.load(open("corpus_obj.bin", 'rb'))
        train_dataset = full_dataset[0]
        test_dataset = full_dataset[1]
        dev_dataset = full_dataset[2]
    else:
        print("Processing dataset file... ...")
        train_dataset = BOW_TopicModel_Corpus(
            device=device, data_path=args.data_path+"train.feat", vocabulary_size=vocabulary_size)
        test_dataset = BOW_TopicModel_Corpus(
            device=device, data_path=args.data_path+"test.feat", vocabulary_size=vocabulary_size)
        dev_dataset = BOW_TopicModel_Corpus(
            device=device, data_path=args.data_path+"test.feat", dev=True, prop=0.1, vocabulary_size=vocabulary_size)
        full_dataset = [train_dataset, test_dataset, dev_dataset]
        pickle.dump(full_dataset, open("corpus_obj.bin", 'wb'))
        # Build dataloader
    # ==========================================================================
    # ==========================================================================
    # Load word embedding
    word2embedding_dict = pickle.load(
        open('embedding_dir/word2vec_glove.6B.100d.txt.bin', 'rb'))
    wordembedding_mat = build_wordembedding_from_dict(
        word2embedding_dict, id2word, args.topicembedsize, vocabulary_size)
    print("Building dataLoader... ...")
    train_loader = DataLoaderX(train_dataset,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=8,
                               pin_memory=True,
                               drop_last=False)
    dev_dataloader = DataLoaderX(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 num_workers=8,
                                 pin_memory=True,
                                 drop_last=False)
    test_loader = DataLoaderX(test_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=8,
                              pin_memory=True,
                              drop_last=False)
    # =========================================================================
    # Build model
    # =========================================================================
    print("Building model... ...")
    model = ASTM(vocabulary_size=vocabulary_size,
                 n_hidden=args.n_hidden,
                 dropout_prob=args.dropout,
                 n_topics=args.topics,
                 embeddings_size=args.topicembedsize,
                 inf_nonlinearity=args.inf_nonlinearity,
                 alternative_epoch=args.alternative_epoch,
                 wordembedding_mat=wordembedding_mat,
                 coel=args.coel,
                 coea=args.coea,
                 topk=args.topk,
                 device=device)
    # model=model.to(device)
    # Train & test process
    print("Initializing training procedure... ...")
    print("===========================================================================================================")
    epoch_trend, SD_trend, STDR_trend, tu_trend = model.train_model(train_dataloader=train_loader,
                                                                    dev_dataloader=dev_dataloader,
                                                                    test_dataloader=test_loader,
                                                                    id2word=id2word,
                                                                    batch_size=args.batch_size,
                                                                    learning_rate=args.lr,
                                                                    alternative_epoch=args.alternative_epoch)
    #draw_curve(epoch_trend, ppx_trend, 'Epoch', 'Perplexity', 'Perplexity trend fig', True)
    draw_curve(epoch_trend, SD_trend, 'Epoch',
               'Sinkhorn Divergence', 'SD trend', True)
    draw_curve(epoch_trend, STDR_trend, 'Epoch', 'STDR', 'STDR trend', True)
    draw_curve(epoch_trend, tu_trend, 'Epoch', 'TU', 'TU trend', True)


if __name__ == '__main__':
    main(parser.parse_args())
