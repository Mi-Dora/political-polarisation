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


df = pd.read_csv('../data_cleaned/tweets/out_3143_hash.csv')
pass
# df = pd.read_pickle('../data_cleaned/tweets/out_3143_hash.pkl')
# df.to_csv('../data_cleaned/tweets/out_3143_hash1.csv')


