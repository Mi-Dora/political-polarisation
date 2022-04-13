import pandas as pd
import numpy as np
import os
import argparse
import math
import tqdm
from multiprocessing import Process
from langdetect import detect, detect_langs, DetectorFactory


drop_col = ['lat', 'long', 'likes', 'retweets', 'replies', 'quote_count']
fill_col = {'original_tweet_id': 0}

def clean(_tweets_fs):
    for tweets_f in _tweets_fs:
        df = pd.read_csv(tweets_f, on_bad_lines='skip')
        df.drop(drop_col, axis=1, inplace=True)
        df.fillna(value=fill_col)
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tweets_dir', type=str, default='../data/tweets/',
                        help='Path to load tweet dataset')
    parser.add_argument('--save_path', type=str, default='../data_cleaned/',
                        help='Path to save the cleaned dataset')
    parser.add_argument('--num_threads', type=int, default=8,
                        help='Numb er of thread using for downloading.')
    args = parser.parse_args()
    save_path = args.save_path
    tweets_dir = args.tweets_dir
    num_threads = args.num_threads
    tweets_fs = []
    for root, _, files in os.walk(tweets_dir):
        for file in files:
            tweets_fs.append(os.path.join(root, file))
    thread_handle = []
    num_csv_per_thread = math.ceil(len(tweets_fs)/num_threads)
    for i in range(num_threads):
        if i == num_threads-1:
            thread_handle.append(Process(target=clean, args=(tweets_fs[i*num_csv_per_thread:],)))
        else:
            thread_handle.append(Process(target=clean, args=(tweets_fs[i*num_csv_per_thread:(i+1)*num_csv_per_thread],)))
        thread_handle[i].start()
    for i in range(num_threads):
        thread_handle[i].join()


















