import pandas as pd

sentences = pd.read_csv('./data_cleaned/tweets/out_3143.csv')

print(sentences['text'].tolist()[:2])
