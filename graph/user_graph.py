import pandas as pd
import numpy as np
import os
import argparse
import json
import time
import math
from tqdm import tqdm
from multiprocessing import Process
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from colorspacious import cspace_converter
import networkx as nx


attitude_list = ['Against Biden', 'Favor Biden', 'None Biden', 'Against Trump', 'Favor Trump', 'None Trump']


def cal_linkset(mention_df, max_num=1000):
    link_df = pd.DataFrame(columns=['n1', 'n2'])
    mention_df['mentions'] = mention_df['mentions'].astype("string")

    def get_row_link(row):
        a = row['mentions']
        if str(a) == 'set()':
            return pd.DataFrame(columns=['n1', 'n2'])
        try:
            mention_list = row['mentions'].strip('{} \'').split(',')
            link_list = []
            for mention in mention_list:
                link_list.append([row['user'], mention.strip("{} \'")])
            return pd.DataFrame(link_list, columns=['n1', 'n2'])
        except AttributeError:
            return pd.DataFrame(columns=['n1', 'n2'])
    ct = 0
    for _, row in tqdm(mention_df.iterrows()):
        link_df = pd.concat([link_df, get_row_link(row)], ignore_index=True)
        ct += 1
        if ct > max_num:
            break
    return link_df


def cos_sim(v1, v2):
    """
    :param v1: (ndarray), vector 1
    :param v2: (ndarray), vector 2
    :return: cosine value of the two vectors
    """
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    inner_dot = np.dot(v1, v2).sum()
    cosine = inner_dot / norm_v1 / norm_v2
    return cosine


def build_graph(tweet_df, user_feature_df, search_dir='./tmp', save_idx=0, cache=True):
    selected_cols = ['user'] + attitude_list
    mention_df = pd.DataFrame(columns=['user', 'mentions'])
    mention_df[['user', 'mentions']] = tweet_df[['user', 'mentions']]
    feature_df = pd.DataFrame(columns=['user', 'features'] + attitude_list)
    feature_df[['user'] + attitude_list] = user_feature_df[['user'] + attitude_list]

    def cat_feature(row):
        features = []
        for col in attitude_list:
            features.append(row[col])
        return np.array(features)
    feature_df['features'] = feature_df.apply(cat_feature, axis=1)
    file = os.path.join(search_dir, 'link.csv')
    if os.path.exists(file) and cache:
        link_df = pd.read_csv(file, on_bad_lines='skip', index_col=0)
    else:
        link_df = cal_linkset(mention_df, max_num=np.Inf)
        link_df.to_csv(file)
    G = nx.from_pandas_edgelist(link_df, 'n1', 'n2')
    return G


def get_user_feature(tweet_df, reduce=True, save_file=None):
    """
    :param tweet_df: Dataframe with tweets features
    :param reduce: whether calculate the average feature value (Ture)
                        or keep the number (of tweets) for each user (False)
    :return: Dataframe with user features (no duplicate user in the df)
    """
    user_aggr_df = pd.DataFrame(columns=['user'] + attitude_list + ['num'])
    selected_cols = ['user'] + attitude_list
    user_aggr_df[selected_cols] = tweet_df[selected_cols]
    user_aggr_df['num'] = 1
    user_aggr_df = user_aggr_df.groupby(['user']).sum()
    user_aggr_df['user'] = user_aggr_df.index
    if reduce:
        reduced_user_df = pd.DataFrame(columns=['user'] + attitude_list)
        reduced_user_df['user'] = user_aggr_df['user']
        for col in attitude_list:
            reduced_user_df[col] = user_aggr_df[col]/user_aggr_df['num']
        if save_file is not None:
            reduced_user_df.to_csv(save_file)
        return reduced_user_df
    else:
        return user_aggr_df


def cal_2value_attitude(user_feature_df):
    tmp_df = pd.DataFrame(columns=['Biden sum', 'Trump sum', 'Biden_x', 'Trump_y'])
    tmp_df['Biden sum'] = user_feature_df['Against Biden'] + user_feature_df['Favor Biden'] + user_feature_df['None Biden']
    tmp_df['Trump sum'] = user_feature_df['Against Trump'] + user_feature_df['Favor Trump'] + user_feature_df['None Trump']
    tmp_df['Biden_x'] = (user_feature_df['Favor Biden'] - user_feature_df['Against Biden']) / tmp_df['Biden sum']
    tmp_df['Trump_y'] = (user_feature_df['Favor Trump'] - user_feature_df['Against Trump']) / tmp_df['Trump sum']
    return tmp_df[['Biden_x', 'Trump_y']]


def plot_attitude(user_feature_df, save_idx=0):
    value_df = cal_2value_attitude(user_feature_df)

    value_df.plot(x='Biden_x', y='Trump_y', kind='scatter', s=1, colormap='jet', figsize=(10, 10), xlim=[-1, 1], ylim=[-1, 1])
    plt.xlabel(r'$\leftarrow$' + "Against Biden    Favor Biden" + r'$\rightarrow$')
    plt.ylabel(r'$\leftarrow$' + "Against Trump    Favor Trump" + r'$\rightarrow$')
    plt.hlines(y=0, xmin=-1, xmax=1, colors=(0.5, 0.5, 0.5), linestyles='dashed')
    plt.vlines(x=0, ymin=-1, ymax=1, colors=(0.5, 0.5, 0.5), linestyles='dashed')
    plt.savefig("./attitude{}.png".format(save_idx))
    plt.close()

# def multi_thread(num_threads, func):
#     thread_handle = []
#     base = len(tweets_fs)//num_threads
#     remain = len(tweets_fs) % num_threads
#     start = 0
#     for i in range(num_threads):
#         if i < remain:
#             num_csv_per_thread = base + 1
#         else:
#             num_csv_per_thread = base
#         thread_handle.append(Process(target=func, args=(tweets_fs[start:start+num_csv_per_thread], save_path, i, )))
#         start += num_csv_per_thread
#         print('Thread %d is starting' % i)
#         thread_handle[i].start()
#     start = time.time()
#     for i in range(num_threads):
#         print('Thread %d is running' % i)
#         thread_handle[i].join()
#     print('Time cost: {}'.format(time.time() - start))


def analyze_graph(G):
    degree_list = list(nx.degree(G))
    degree_list.sort(key=lambda x: x[1], reverse=True)
    print('Degree Top 20:')
    # print(degree_list[:100])
    for i in range(20):
        print(degree_list[i])
    largest_component = max(nx.connected_components(G), key=len)
    largest_component_graph = G.subgraph(largest_component)
    print("Largest Component Graph for the original graph: {}".format(nx.number_of_nodes(largest_component_graph)))
    G.remove_node(degree_list[0][0])
    G.remove_node(degree_list[1][0])
    largest_component = max(nx.connected_components(G), key=len)
    largest_component_graph = G.subgraph(largest_component)
    print("Largest Component Graph after remove Top 2 nodes: {}".format(nx.number_of_nodes(largest_component_graph)))
    degree_list = list(nx.degree(G))
    degree_list.sort(key=lambda x: x[1], reverse=True)
    print('After Degree Top 20:')
    # print(degree_list[:100])
    for i in range(20):
        print(degree_list[i])
    options = {
        # 'node_color': 'blue',
        'node_size': 0.5,
        'width': 0.2,
    }
    # plt.figure(figsize=(10, 10))
    # nx.draw(largest_component_graph, with_labels=False, **options)
    # plt.savefig("graph_{}.png".format(save_idx))
    # plt.close()
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tweets_dir', type=str, default='../data/political_incline/ziffer (admin)/result',
                        help='Path to load tweet dataset')
    parser.add_argument('--save_path', type=str, default='../data_cleaned/user_feature/',
                        help='Path to save the cleaned dataset')
    args = parser.parse_args()
    save_path = args.save_path
    tweets_dir = args.tweets_dir
    # G = nx.Graph()
    # G.add_edge(0, 1)
    # G.add_edge(2, 1)
    # G.add_edge(3, 1)
    # nx.draw(G, with_labels=False)
    # plt.show()
    os.makedirs(save_path, exist_ok=True)
    tweets_fs = []
    for root, _, files in os.walk(tweets_dir):
        for file in sorted(files):
            if file[0] == '.':
                continue
            tweets_fs.append(os.path.join(root, file))
    start = time.time()
    for i, tweet_f in enumerate(tweets_fs):
        df_tweets = pd.read_csv(tweet_f, on_bad_lines='skip', index_col=0, low_memory=False)
        user_feature_df = get_user_feature(df_tweets, reduce=True)
        # user_feature_df.to_csv(os.path.join(save_path, os.path.basename(tweet_f).split('.')[0]+'_user_feature.csv'))
        G = build_graph(df_tweets, user_feature_df, save_idx=i)
        analyze_graph(G)
        # plot_attitude(user_feature_df, save_idx=i)
        break

    print('Time cost: {}'.format(time.time()-start))





