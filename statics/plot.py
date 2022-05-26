import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt

def get_text(folder = './BERT/result/'):
    files = []
    for root, dirs, fs in os.walk(folder):
        files = [folder+f for f in fs]

    df_res = None
    for f in files:
        df = pd.read_csv(f, usecols=["Against Biden", "Favor Biden", "None Biden", "Against Trump", "Favor Trump", "None Trump"])
        df_res = pd.concat([df_res, df], axis=0, sort=False)
        break
    return df_res


# df_res = get_text('../BERT/result/')
df_user = get_text('../data_cleaned/user_feature/')

df_a = df_user.copy(deep=True)

df_a['Biden'] = df_a.apply(lambda x: x['Favor Biden'] - x['Against Biden'], axis=1)
df_a['Trump'] = df_a.apply(lambda x: x['Favor Trump'] - x['Against Trump'], axis=1)
sns.displot(x="Biden", data=df_a)
sns.displot(x="Trump", data=df_a)
# plt.figure(dpi=600, figsize=(100, 100))
p = sns.jointplot(data=df_a, x="Biden", y="Trump", height=100)
p.ax_joint.set_xlabel(r'$\leftarrow$' + "Against Biden    Favor Biden" + r'$\rightarrow$', fontsize=100)
p.ax_joint.set_ylabel(r'$\leftarrow$' + "Against Trump    Favor Trump" + r'$\rightarrow$', fontsize=100)
plt.savefig('res.png')
exit()
