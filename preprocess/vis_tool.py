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


df_tweets = pd.read_csv('../data_cleaned/tweets/hash/out_3143_hash.csv', on_bad_lines='skip', index_col=0)
df_users = pd.read_csv('../data_cleaned/users/out_3143.csv', on_bad_lines='skip', index_col=0)
df_retweets = pd.read_csv('../data/retweets//out_3143.csv', on_bad_lines='skip', index_col=0)
df_ori_tweets = pd.read_csv('../data/tweets//out_3143.csv', on_bad_lines='skip', index_col=0)
df_user_feature = pd.read_csv('../data_cleaned/user_feature/out_3143_hash_user_feature.csv', on_bad_lines='skip', index_col=0)

pass
# df = pd.read_pickle('../data_cleaned/tweets/out_3143_hash.pkl')
# df.to_csv('../data_cleaned/tweets/out_3143_hash1.csv')


