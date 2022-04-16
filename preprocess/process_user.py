import pandas as pd
import numpy as np
import os
import re
import csv
import argparse
import math
from tqdm import tqdm
from emoji import demojize
from multiprocessing import Process
from langdetect import detect, detect_langs, DetectorFactory, lang_detect_exception


drop_col = ['translator_type', 'protected', 'verified', 'time_zone', 'lang', 'utc_offset', 'contributors_enabled',
            'is_translator', 'profile_background_color', 'utc_offset', 'contributors_enabled']


def de_emoji(string):
    if string is np.nan:
        return string
    else:
        try:
            return demojize(string)
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
    for both 'description' and 'name''
    """

    tqdm.pandas(desc='pandas bar')

    # handle emoji
    df['description'] = df['description'].progress_map(de_emoji)
    df['name'] = df['name'].map(de_emoji)

    # remove non-English
    df['lang'] = df['description'].progress_map(detect_language)
    df.drop(df[df['lang'] != 'en'].index, inplace=True)
    print(len(df.index))


def df_preproc(fpath):
    csv_f = csv.reader(open(fpath))
    header = csv_f.__next__()
    header = list(map(lambda x: x.strip(), header))
    rows = []
    for line in csv_f:
        if len(line) < len(header):
            continue
        if len(line) > len(header):
            loc_num = len(line) - len(header) + 1
            line[2] = ', '.join(line[2:2+loc_num])
            del line[3:2+loc_num]
        rows.append(line)
    return header, rows


def clean(_users_fs, save_path):
    for users_f in _users_fs:
        header, rows = df_preproc(users_f)
        df = pd.DataFrame(rows, columns=header, dtype=int)
        df.drop(drop_col, axis=1, inplace=True)
        text_handle(df)
        df.to_csv(os.path.join(save_path, os.path.basename(users_f)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--users_dir', type=str, default='../data/users/',
                        help='Path to load tweet dataset')
    parser.add_argument('--save_path', type=str, default='../data_cleaned/users',
                        help='Path to save the cleaned dataset')
    parser.add_argument('--num_threads', type=int, default=1,
                        help='Number of thread using for downloading.')
    args = parser.parse_args()
    save_path = args.save_path
    users_dir = args.users_dir
    num_threads = args.num_threads
    os.makedirs(save_path, exist_ok=True)
    users_fs = []
    for root, _, files in os.walk(users_dir):
        for file in sorted(files):
            users_fs.append(os.path.join(root, file))
    # for users_f in users_fs:
    #     clean([users_f], save_path, 0)
    thread_handle = []
    base = len(users_fs) // num_threads
    remain = len(users_fs) % num_threads
    start = 0
    for i in range(num_threads):
        if i < remain:
            num_csv_per_thread = base + 1
        else:
            num_csv_per_thread = base
        a = users_fs[start:start + num_csv_per_thread]
        thread_handle.append(Process(target=clean, args=(users_fs[start:start + num_csv_per_thread],save_path, )))
        start += num_csv_per_thread
        print('Thread %d is starting' % i)
        thread_handle[i].start()

    for i in range(num_threads):
        print('Thread %d is running' % i)
        thread_handle[i].join()

