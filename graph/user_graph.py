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
import seaborn as sns

from graph.build_user import get_user_feature, attitude_list


def label(file, sets: list):
    df = pd.read_csv(file)
    df.set_index('user')
    df_res = pd.DataFrame()
    for i, s in enumerate(sets):
        for u in s:
            res = df[(df.user == u)]
            res['label'] = i
            df_res = pd.concat([df_res, res], ignore_index=True)
    return df_res

# l = [set(['001maxlogic','emschneiferr']), set(['0000zerooclock','00010001b'])]
# a = label('C:/Users/ziffer/Desktop/political-polarisation/BERT/user_feature/aggr_user.csv', l)


def cal_linkset(mention_df, max_num=None):
    link_df = pd.DataFrame(columns=['n1', 'n2'])
    mention_df['mentions'] = mention_df['mentions'].astype("string")
    if max_num is None:
        max_num = np.Inf
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


def build_graph(tweet_df, search_name='./tmp/link.csv', cache=True, max_num=None):
    user_feature_df = get_user_feature(df_tweets, reduce=True)
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
    if os.path.exists(search_name) and cache:
        link_df = pd.read_csv(search_name, on_bad_lines='skip', index_col=0)
        if max_num is not None:
            link_df = link_df[:max_num]
    else:
        link_df = cal_linkset(mention_df, max_num=max_num)
        link_df.to_csv(search_name)
    G = nx.from_pandas_edgelist(link_df, 'n1', 'n2')
    return G


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


def analyze_graph(G, save_idx=0):
    degree_list = list(nx.degree(G))
    num_node = len(degree_list)
    options = {
        # 'node_color': 'blue',
        'node_size': 0.05,
        'width': 0.1,
    }
    # plt.figure(figsize=(10, 10))
    # nx.draw(G, with_labels=False, **options)
    # print('Saving...')
    # plt.savefig("graph_ori_{}_{}.png".format(num_node, save_idx), dpi=1200)
    # plt.close()
    # plt.show()

    def draw_deg_distribution(G):
        degree_df = pd.DataFrame(nx.degree(G), columns=['user', 'degree'])
        degree_df['num'] = 1
        degree_df = degree_df.groupby(['degree']).sum()
        degree_df['degree'] = degree_df.index
        degree_df['log_degree'] = degree_df['degree'].apply(np.log10)
        degree_df['log_num'] = degree_df['num'].apply(np.log10)
        ax = sns.scatterplot(x=degree_df['log_degree'], y=degree_df['log_num'])
        ax.set_title('Log Degree Distribution', fontsize=15)
        ax.set_ylabel(r'$\lg$(Number of node)', fontsize=15)
        ax.set_xlabel(r'$\lg (k)$', fontsize=15)
        plt.tight_layout()
        plt.savefig("degree_distribution_{}.png".format(save_idx), dpi=600)

    def draw_deg_rank(G):
        degree_list = list(nx.degree(G))
        degree_list.sort(key=lambda x: x[1], reverse=True)
        deg_name = []
        deg_num = []
        for i in range(20):
            deg_name.append(degree_list[i][0])
            deg_num.append(degree_list[i][1])

        ax = sns.barplot(x=deg_num, y=deg_name, orient='h')
        ax.set_title('Top 20 Degree Node', fontsize=15)
        ax.set_ylabel('user', fontsize=15)
        ax.set_xlabel('Degree', fontsize=15)
        plt.tight_layout()
        plt.savefig("degree_rank_{}.png".format(save_idx), dpi=600)

    # draw_deg_distribution(G)
    # draw_deg_rank(G)

    # print('Degree Top 20:')
    # print(degree_list[:100])

    def get_subgraph(G):
        sub_G_size = list(map(len, list(nx.connected_components(G))))
        sub_G_size.sort(reverse=True)
        subgraph_df = pd.DataFrame(sub_G_size, columns=['nodes'])
        subgraph_df['num'] = 1
        subgraph_df = subgraph_df.groupby(['nodes']).sum()
        subgraph_df['nodes'] = subgraph_df.index
        subgraph_df['log_nodes'] = subgraph_df['nodes'].apply(np.log10)
        subgraph_df['log_num'] = subgraph_df['num'].apply(np.log10)
        ax = sns.scatterplot(x=subgraph_df['log_nodes'], y=subgraph_df['log_num'])
        ax.set_title('SubGraph Size Distribution', fontsize=15)
        ax.set_ylabel(r'$\lg$(# of SubGraph with the same # of nodes)', fontsize=15)
        ax.set_xlabel(r'$\lg$(# of Node in SubGraph)', fontsize=15)
        plt.tight_layout()
        plt.savefig("sub_distribute_{}.png".format(save_idx), dpi=600)

    get_subgraph(G)
    # largest_component = max(nx.connected_components(G), key=len)
    # largest_component_graph = G.subgraph(largest_component)
    # print("Largest Component Graph for the original graph: {}".format(nx.number_of_nodes(largest_component_graph)))

    # largest_component = max(nx.connected_components(G), key=len)
    # largest_component_graph = G.subgraph(largest_component)
    # print("Largest Component Graph after remove Top 2 nodes: {}".format(nx.number_of_nodes(largest_component_graph)))
    degree_list = list(nx.degree(G))
    degree_list.sort(key=lambda x: x[1], reverse=True)
    G.remove_node(degree_list[0][0])
    G.remove_node(degree_list[1][0])
    # print('After Degree Top 20:')
    # print(degree_list[:100])
    # for i in range(20):
    #     print(degree_list[i])


    # plt.figure(figsize=(10, 10))
    # nx.draw(G, with_labels=False, **options)
    # print('Saving...')
    # plt.savefig("graph_remove2_{}_{}.png".format(num_node, save_idx), dpi=1200)
    # plt.close()
    # plt.show()


def subgraph_sim(G):
    sub_G = list(nx.connected_components(G))
    sub_G.sort(reverse=True, key=len)
    del sub_G[0]
    df_res = label('../data_cleaned/user_feature/out_3143_hash_user_feature.csv', sub_G[:10])
    noise_df = pd.DataFrame(np.random.random((df_res.shape[0], len(attitude_list))) * 0.1 - 0.05, columns=attitude_list)
    noise_df['user'] = df_res['user']
    noise_df['label'] = 0
    df_res = df_res+noise_df

    def cal_2value_attitude(user_feature_df):
        tmp_df = pd.DataFrame(columns=['Biden sum', 'Trump sum', 'Biden_x', 'Trump_y', 'label'])
        tmp_df['label'] = user_feature_df['label']

        tmp_df['Biden sum'] = user_feature_df['Against Biden'] + user_feature_df['Favor Biden'] + user_feature_df['None Biden']

        tmp_df['Trump sum'] = user_feature_df['Against Trump'] + user_feature_df['Favor Trump'] + user_feature_df['None Trump']
        tmp_df['Biden_x'] = (user_feature_df['Favor Biden'] - user_feature_df['Against Biden']) / tmp_df['Biden sum']
        tmp_df['Trump_y'] = (user_feature_df['Favor Trump'] - user_feature_df['Against Trump']) / tmp_df['Trump sum']
        # tmp_df['Biden_x'] = user_feature_df['Favor Biden']
        # tmp_df['Trump_y'] = user_feature_df['Favor Trump']
        return tmp_df[['Biden_x', 'Trump_y', 'label']]

    plot_df = cal_2value_attitude(df_res)
    plot_df['label'] = plot_df['label'].astype(str)

    def plot_attitude(user_feature_df, save_idx=0):
        # sns.color_palette("hls", 10)
        ax = sns.scatterplot(x=user_feature_df['Biden_x'], y=user_feature_df['Trump_y'],
                        hue=user_feature_df['label'], legend=False)
        ax.set_title('Top (2-11) SubGraph User Attitude', fontsize=15)
        ax.set_xlabel(r'$\leftarrow$' + "Against Biden    Favor Biden" + r'$\rightarrow$', fontsize=15)
        ax.set_ylabel(r'$\leftarrow$' + "Against Trump    Favor Trump" + r'$\rightarrow$', fontsize=15)
        plt.hlines(y=0, xmin=-1, xmax=1, colors=(0.5, 0.5, 0.5), linestyles='dashed')
        plt.vlines(x=0, ymin=-1, ymax=1, colors=(0.5, 0.5, 0.5), linestyles='dashed')
        plt.savefig("./sub_graph_attitude{}.png".format(save_idx), dpi=600)
        plt.close()

    plot_attitude(plot_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tweets_dir', type=str, default='../data/political_incline/ziffer (admin)/result',
                        help='Path to load tweet dataset')
    parser.add_argument('--save_path', type=str, default='../data_cleaned/user_feature/',
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
    start = time.time()
    for i, tweet_f in enumerate(tweets_fs):
        df_tweets = pd.read_csv(tweet_f, on_bad_lines='skip', index_col=0, low_memory=False)

        G = build_graph(df_tweets, max_num=None)
        subgraph_sim(G)
        # analyze_graph(G, save_idx=4)
        # plot_attitude(user_feature_df, save_idx=i)
        break

    print('Time cost: {}'.format(time.time()-start))





