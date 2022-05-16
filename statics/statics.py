import seaborn as sns
import os
import pandas as pd

def draw_text(folder = './BERT/result/'):
    files = []
    for root,dirs,fs in os.walk(folder):
        files = [folder+f for f in fs]

    df_res = None
    for f in files:
        df = pd.read_csv(f, usecols=["Against Biden","Favor Biden","None Biden","Against Trump","Favor Trump","None Trump"])
        df_res = pd.concat([df_res,df], axis=0, sort=False)
        break

    sns.displot(data=df_res, x="Favor Biden")
    print("!!!")

draw_text()