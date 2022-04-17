import os
import matplotlib.pyplot as plt
import pandas as pd
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
    df_name = df_name.dropna(how='all')
    df_name = df_name.map(
        lambda x: x.replace('\'', '').replace('{', '').replace('}', '').replace('set()', '').replace(' ', '').replace(
            '#', ''))
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


if __name__ == '__main__':
    all_df = pd.read_csv('../datasets/tweets/out_3143.csv', delimiter=',', header=0, index_col=0).drop(columns=['url', 'quoted_url'])
    all_df = all_df[all_df['lang'] == 'en']
    all_df = all_df.dropna(how='any', subset=['hashtag']).drop(columns=['lang'])
    all_df['is_retweet'].astype("bool")
    all_df['is_quote'].astype("bool")
    all_df = all_df[all_df['is_retweet'] & (all_df['is_quote'] != True)]
    display(all_df)
