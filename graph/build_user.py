import pandas as pd
import numpy as np
import os
import argparse
import time
from tqdm import tqdm


attitude_list = ['Against Biden', 'Favor Biden', 'None Biden', 'Against Trump', 'Favor Trump', 'None Trump']


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


def aggregate_user(user_aggr_df, save_file=None):
    pd.DataFrame(columns=['user'] + attitude_list + ['num'])
    user_aggr_df = user_aggr_df.groupby(['user']).sum()
    user_aggr_df['user'] = user_aggr_df.index
    reduced_user_df = pd.DataFrame(columns=['user'] + attitude_list)
    reduced_user_df['user'] = user_aggr_df['user']
    for col in tqdm(attitude_list):
        reduced_user_df[col] = user_aggr_df[col] / user_aggr_df['num']
    print("Number of users: {}".format(len(reduced_user_df)))
    if save_file is not None:
        reduced_user_df.to_csv(save_file)
    return reduced_user_df


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
    all_df = pd.DataFrame(columns=['user'] + attitude_list + ['num'])
    print('Processing single file...')
    for tweet_f in tqdm(tweets_fs):

        df_tweets = pd.read_csv(tweet_f, on_bad_lines='skip', index_col=0, low_memory=False)
        user_feature_df = get_user_feature(df_tweets, reduce=False)
        all_df = pd.concat([all_df, user_feature_df], ignore_index=True)
    print('Aggregating...')
    aggregate_user(all_df, '../data_cleaned/user_feature/aggr_user.csv')

    print('Time cost: {}'.format(time.time()-start))






