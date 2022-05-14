import pandas as pd
import numpy as np
import os
import re
import argparse
import time
import math
from tqdm import tqdm
from emoji import demojize
from multiprocessing import Process
import networkx as nx


def cal_linkset(tweet_f):
    df_tweets = pd.read_csv(tweet_f, on_bad_lines='skip', index_col=0)
    selected_cols = ['user', 'mentions']



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tweets_dir', type=str, default='../data/tweets/',
                        help='Path to load tweet dataset')
    parser.add_argument('--save_path', type=str, default='../data_cleaned/tweets',
                        help='Path to save the cleaned dataset')
    args = parser.parse_args()
    save_path = args.save_path
    tweets_dir = args.tweets_dir

    os.makedirs(save_path, exist_ok=True)
    tweets_fs = []
    for root, _, files in os.walk(tweets_dir):
        for file in sorted(files):
            if file[0] == '.':
                continue
            tweets_fs.append(os.path.join(root, file))






