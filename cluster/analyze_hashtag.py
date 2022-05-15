import os
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from IPython.core.display import display
from wordcloud import WordCloud, STOPWORDS

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('expand_frame_repr', False)

demo_word_set = {'blacklivesmatter', 'BidenHarris2020', 'GullibleWhiteMaleTrumpVoters', 'VoteHimOut', 'Biden',
                 'VoteBlueToEndTheNightmare', 'WalkHimOut', 'Democrats', 'BLM', 'TrumpSurrenderedBidenWont',
                 'BidenHarrisToSaveAmerica', 'VoteBidenHarris', 'WomenForBidenHarris', 'VoteBlue',
                 'BidenHarris2020ToSaveAmerica'}
gop_word_set = {'MAGA', 'Trump2020', 'Trump', 'Trump2020LandslideVictory', 'MAGA2020', 'BidenCrimeFamiIy',
                'KAG', 'FuckYouKeepCounting', 'DonaldTrump', 'TRUMP2020', 'MAGA2020LandslideVictory',
                'TRUMP2020ToSaveAmerica', 'VoteTrump2020', '4MoreYears', 'FourMoreYears', 'TrumpPence2020',
                'VoteRedToSaveAmerica', 'BidenCrimeSyndicate', 'VoteRedLikeYourLifeDependsOnIt', 'AmericaFirst'}
vote_word_set = {'VOTE', 'vote', 'Vote', 'ElectionDay', 'Election2020', 'election2020', '2020Election',
                 'USElections2020', 'Vote2020', 'VoteEarlyDay'}


def trim_hashtag(df):
    df['hashtag'] = df['hashtag'].astype("string")
    df_name = df['hashtag']
    df_name = df_name.str.split(',', expand=True)
    df_name = df_name.stack()
    # 重置索引，并删除多于的索引
    df_name = df_name.reset_index(level=1, drop=True).to_numpy()
    word_count = {}
    for item in df_name:
        item = item.replace('\'', '').replace('{', '').replace('}', '').replace('set()', '').replace(' ', '').replace(
            '#', '')
        if item in word_count:
            word_count[item] += 1
        else:
            word_count[item] = 1
    print({k: v for k, v in sorted(word_count.items(), key=lambda item: item[1], reverse=True)})
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=set(STOPWORDS),
                          min_font_size=10).generate(' '.join(df_name))
    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


def clean_list(x):
    return x.replace('\'', '').replace('{', '').replace('}', '').replace('set()', '').replace(' ', '').replace('#', '').strip()


def build_mention_graph(df):
    count = 0
    dg = nx.Graph()
    for _, row in df.iterrows():
        username = row['user']
        mention_list = row['mentions'].strip().split(',')
        for mention in mention_list:
            dg.add_edge(username, mention.strip())
            count += 1
        if count > 1000000:
            break
        elif count % 1000 == 0:
            print(count)
    options = {
        'node_color': 'black',
        'node_size': 0.2,
        'width': 0.01,
    }
    largest_component = max(nx.connected_components(dg), key=len)
    largest_component_graph = dg.subgraph(largest_component)
    print(nx.number_of_nodes(largest_component_graph))
    biparties = nx.community.kernighan_lin_bisection(largest_component_graph, max_iter=100,)
    print(len(biparties))
    label_communities = nx.community.label_propagation_communities(largest_component_graph)
    label_communities = sorted(label_communities, key=lambda i: len(i), reverse=True)
    print(len(label_communities))
    # for community in communities[10:]:
    #     for item in community:
    #         dg.remove_node(item)
    # nx.draw(dg, with_labels=False, **options)
    # plt.show()
    greedy_communities = nx.algorithms.community.greedy_modularity_communities(largest_component_graph, n_communities=3)
    print(len(greedy_communities))


if __name__ == '__main__':
    # all_df = pd.read_csv('../data_cleaned/tweets/out_3143.csv', delimiter=',', header=0, index_col=0).drop(columns=['url', 'quoted_url'])
    # all_df = all_df[all_df['lang'] == 'en']
    # all_df = all_df.dropna(how='any', subset=['hashtag']).drop(columns=['lang'])
    # all_df['is_retweet'].astype("bool")
    # all_df['is_quote'].astype("bool")
    # all_df['hashtag'].astype("string")
    # all_df['hashtag'] = all_df['hashtag'].map(lambda x: clean_list(x))
    # all_df['mentions'].astype("string")
    # all_df['mentions'] = all_df['mentions'].map(lambda x: clean_list(x))
    all_df = pd.read_pickle('all_df.pkl')
    all_df = all_df[(all_df['is_quote'] != True) & (all_df['is_retweet'] != True) & (all_df['mentions'] != '') & (all_df['user'] != '')]
    build_mention_graph(all_df)
    display(all_df)
