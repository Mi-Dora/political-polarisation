import pandas as pd
import numpy as np
import os
import re
import argparse
import math
from tqdm import tqdm
from emoji import demojize
from multiprocessing import Process
from langdetect import detect, detect_langs, DetectorFactory, lang_detect_exception


drop_col = ['lat', 'long', 'likes', 'retweets', 'replies', 'quote_count']
fill_col = {'original_tweet_id': 0}
url_re = re.compile(r'https?://[-\./\w]+')

mention_re = re.compile(r'(?<=@)[-_A-Za-z0-9]+')  # @
rm_mention_re = re.compile(r'(RT )?@[-_A-Za-z0-9]+:?')

hashtag_re = re.compile(r'#[-_A-Za-z0-9]+')


def de_emoji(string):
    if string is np.nan:
        return string
    else:
        try:
            return demojize(string)
        except TypeError:
            return string


def find_url(string):
    try:
        return url_re.findall(string)
    except TypeError:
        return []


def rm_url(string):
    try:
        return url_re.sub('', string)
    except TypeError:
        return string


def find_mention(string):
    try:
        return mention_re.findall(string)
    except TypeError:
        return []


def rm_mention(string):
    try:
        return rm_mention_re.sub('', string)
    except TypeError:
        return string


def mentions2list(string):
    if string is np.nan:
        return []
    else:
        return string.split(';')


def find_hashtag(string):
    try:
        return hashtag_re.findall(string)
    except TypeError:
        return []


def rm_hashtag(string):
    try:
        return hashtag_re.sub('', string)
    except TypeError:
        return string


def detect_language(string):
    try:
        return detect(string)
    except:
        return np.nan


def text_handle(df):
    """
    de-emoji, find and remove url, remove non-English text, find and remove mentioned users
    for both 'text' and 'quoted_text'
    """

    tqdm.pandas(desc='pandas bar')

    # handle emoji
    df['text'] = df['text'].progress_map(de_emoji)
    df['quoted_text'] = df['quoted_text'].map(de_emoji)

    # handle url
    df['url'] = df['text'].progress_map(find_url)
    df['text'] = df['text'].progress_map(rm_url)
    df['quoted_url'] = df['quoted_text'].progress_map(find_url)
    df['quoted_text'] = df['quoted_text'].progress_map(rm_url)

    # handle mention
    df['mentions'] = df['mentions'].progress_map(mentions2list)
    df['mentions'] += df['text'].progress_map(find_mention)
    df['text'] = df['text'].progress_map(rm_mention)
    df['mentions'] += df['quoted_text'].progress_map(find_mention)
    df['quoted_text'] = df['quoted_text'].progress_map(rm_mention)
    df['mentions'] = df['mentions'].progress_map(lambda x: set(x))

    # hashtag
    df['hashtag'] = df['text'].progress_map(find_hashtag)
    df['text'] = df['text'].progress_map(rm_hashtag)
    df['hashtag'] += df['quoted_text'].progress_map(find_hashtag)
    df['quoted_text'] = df['quoted_text'].progress_map(rm_hashtag)
    df['hashtag'] = df['hashtag'].progress_map(lambda x: set(x))

    # remove non-English

    df['lang'] = df['text'].progress_map(detect_language)
    df.drop(df[df['lang'] != 'en'].index, inplace=True)
    print(len(df.index))


def clean(_tweets_fs, save_path, tid):
    print(tid, _tweets_fs)
    for tweets_f in _tweets_fs:
        df = pd.read_csv(tweets_f, on_bad_lines='skip')
        df.drop(drop_col, axis=1, inplace=True)
        df.fillna(value=fill_col, inplace=True)
        df['original_tweet_id'] = pd.to_numeric(df['original_tweet_id'], downcast='integer')  # float64 to int64
        text_handle(df)
        df.to_pickle(os.path.join(save_path, os.path.basename(tweets_f).split('.')[0]+'.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tweets_dir', type=str, default='../data/tweets/',
                        help='Path to load tweet dataset')
    parser.add_argument('--save_path', type=str, default='../data_cleaned/tweets',
                        help='Path to save the cleaned dataset')
    parser.add_argument('--num_threads', type=int, default=4,
                        help='Numb er of thread using for downloading.')
    args = parser.parse_args()
    save_path = args.save_path
    tweets_dir = args.tweets_dir
    num_threads = args.num_threads
    os.makedirs(save_path, exist_ok=True)
    tweets_fs = []
    for root, _, files in os.walk(tweets_dir):
        for file in sorted(files):
            tweets_fs.append(os.path.join(root, file))
    thread_handle = []
    base = len(tweets_fs)//num_threads
    remain = len(tweets_fs) % num_threads
    start = 0
    for i in range(num_threads):
        if i < remain:
            num_csv_per_thread = base + 1
        else:
            num_csv_per_thread = base
        thread_handle.append(Process(target=clean, args=(tweets_fs[start:start+num_csv_per_thread], save_path, i, )))
        start += num_csv_per_thread
        print('Thread %d is starting' % i)
        thread_handle[i].start()
    for i in range(num_threads):
        print('Thread %d is running' % i)
        thread_handle[i].join()


















