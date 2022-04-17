import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertForSequenceClassification, AutoModel, AutoTokenizer, BertTokenizer, BertForMaskedLM, AdamW, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import logging
import argparse
import torch
import torch.nn as nn
import towards_biden
import towards_trump

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add models\' arguments')
    parser.add_argument('--type', default='bt',
                    help='use different pretrained model to train and test')
    parser.add_argument('--qid', default='1')
    parser.add_argument('--ep', default='10', help='number of epochs')
    args = parser.parse_args()

    if args.type == 'bt':
        model_name = 'bert-base-uncased'
    elif args.type == 'rb':
        model_name = 'roberta-base'
    elif args.type == 'simbt':
        model_name = "princeton-nlp/sup-simcse-bert-base-uncased"
    elif args.type == 'simrb':
        model_name = "princeton-nlp/sup-simcse-roberta-base"
    else :
        model_name = args.type