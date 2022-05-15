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
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from colorspacious import cspace_converter
import networkx as nx


attitude_list = ['Against Biden', 'Favor Biden', 'None Biden', 'Against Trump', 'Favor Trump', 'None Trump']

# def cal_linkset(tweet_f):
#     df_tweets = pd.read_csv(tweet_f, on_bad_lines='skip', index_col=0)
#     selected_cols = ['user', 'mentions']


def build_graph(tweet_f):
    df_tweets = pd.read_csv(tweet_f, on_bad_lines='skip', index_col=0)
    selected_cols = ['user', 'mentions']
    g = nx.Graph()
    g.add_node('node1')
    g.add_node('node1')
    pass


def get_user_feature(tweet_df, reduce=True):
    user_aggr_df = pd.DataFrame(columns=['user'] + attitude_list + ['num'])
    selected_cols = ['user'] + attitude_list
    user_aggr_df[selected_cols] = tweet_df[selected_cols]
    user_aggr_df['num'] = 1
    user_aggr_df = user_aggr_df.groupby('user').sum().dropna()
    if reduce:
        reduced_user_df = pd.DataFrame(columns=attitude_list)
        for col in attitude_list:
            reduced_user_df[col] = user_aggr_df[col]/user_aggr_df['num']
        return reduced_user_df
    else:
        return user_aggr_df


def cal_2value_attitude(user_feature_df):
    tmp_df = pd.DataFrame(columns=['Biden sum', 'Trump sum', 'Biden_x', 'Trump_y'])
    # tmp_df['user'] = user_feature_df['user']
    tmp_df['Biden sum'] = user_feature_df['Against Biden'] + user_feature_df['Favor Biden'] + user_feature_df['None Biden']
    tmp_df['Trump sum'] = user_feature_df['Against Trump'] + user_feature_df['Favor Trump'] + user_feature_df['None Trump']
    tmp_df['Biden_x'] = user_feature_df['Favor Biden'] / tmp_df['Biden sum'] - user_feature_df['Against Biden'] / tmp_df['Biden sum']
    tmp_df['Trump_y'] = user_feature_df['Favor Trump'] / tmp_df['Trump sum'] - user_feature_df['Against Trump'] / tmp_df['Trump sum']
    return tmp_df[['Biden_x', 'Trump_y']]


def plot_attitude(user_feature_df):
    value_df = cal_2value_attitude(user_feature_df)

    value_df.plot(x='Biden_x', y='Trump_y', kind='scatter', s=1, colormap='jet', figsize=(10, 10), xlim=[-1, 1], ylim=[-1, 1])
    plt.xlabel(r'$\leftarrow$' + "Against Biden    Favor Biden" + r'$\rightarrow$')
    plt.ylabel(r'$\leftarrow$' + "Against Trump    Favor Trump" + r'$\rightarrow$')
    plt.hlines(y=0, xmin=-1, xmax=1, colors=(0.5, 0.5, 0.5), linestyles='dashed')
    plt.vlines(x=0, ymin=-1, ymax=1, colors=(0.5, 0.5, 0.5), linestyles='dashed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tweets_dir', type=str, default='../data/political_incline/ziffer (admin)/',
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
    # build_graph(tweets_fs[0])
    # build_graph('test.csv')
    df_tweets = pd.read_csv(tweets_fs[0], on_bad_lines='skip', index_col=0)
    user_feature_df = get_user_feature(df_tweets, reduce=True)
    plot_attitude(user_feature_df)
    print('Done')





