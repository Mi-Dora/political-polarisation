import os
import re

import pandas as pd


def read_dir_file(rootdir):
    for parent, dirnames, filenames in os.walk(rootdir):
        for dirname in dirnames:
            read_dir_file(dirname)
        for filename in filenames:
            if filename.endswith(".csv"):
                print("{} is handled".format(os.path.join(parent, filename)))
                clean_csv(os.path.join(parent, filename))


def clean_list(x):
    if x:
        x = str(x)
        x = x.lower()  # lowercase everything
        x = bytes(x, 'utf-8').decode('utf-8', 'ignore')  # remove unicode characters
        x = re.sub(r'www*\S+', ' ', x)  # remove links
        x = re.sub(r'http*\S+', ' ', x)
        x = re.sub(r'&\S+', ' ', x)
        # cleaning up text
        x = re.sub(r'\'\w+', '', x)
        x = re.sub(r'\w*\d+\w*', '', x)
        x = re.sub(r'\s{2,}', ' ', x)
        x = re.sub(r'\s[^\w\s]\s', '', x)
        # Heavy
        x = ' '.join([word for word in x.split(' ')])
        x = re.sub(r'@\S', '', x)
        x = re.sub(r'#\S+', ' ', x)
        # remove single letters and numbers surrounded by space
        x = re.sub(r'\s[a-z]\s|\s[0-9]\s', ' ', x)
        x = x.replace('\'', '').replace('\"', '').replace('::', ' ').replace('_', ' ').strip()
    return x


def clean_csv(filename):
    candidate_df = pd.read_csv(filename, delimiter=',', header=0, index_col=0)\
        .drop(columns=['date', 'user', 'is_retweet', 'is_quote', 'hts', 'mentions', 'original_tweet_id', 'url', 'quoted_url', 'mentions'])
    candidate_df = candidate_df[candidate_df['lang'] == 'en']
    candidate_df['text'].astype("string")
    candidate_df['quoted_text'].astype("string")
    candidate_df['text'] = candidate_df['text'].map(lambda x: clean_list(x))
    candidate_df['quoted_text'] = candidate_df['quoted_text'].map(lambda x: clean_list(x))
    text = ''.join(candidate_df['text'].tolist())
    with open('tweet.txt', 'a', encoding="utf-8") as fp:
        fp.write(text)
    text = ''.join(candidate_df['quoted_text'].tolist())
    with open('tweet.txt', 'a', encoding="utf-8") as fp:
        fp.write(text)


if __name__ == '__main__':
    read_dir_file('../data_cleaned/hash')
