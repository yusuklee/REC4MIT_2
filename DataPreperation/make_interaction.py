import ast
import json
from collections import defaultdict
import os

os.makedirs("datas/user_interaction", exist_ok=True)
import pandas as pd
gossip = pd.read_csv("datas/news/gossip.csv")
pol = pd.read_csv("datas/news/pol.csv")

# 'news_id', 'label', 'user_ids', 'user_times'

go_user_news = defaultdict(list)
for _,row in gossip.iterrows():
    users = ast.literal_eval(row["user_ids"])
    times = ast.literal_eval(row["user_times"])
    for u,t in zip(users,times):
        go_user_news[u].append((row['news_id'],t))


pol_user_news = defaultdict(list)

for _,row in pol.iterrows():
    users = ast.literal_eval(row["user_ids"])
    times = ast.literal_eval(row["user_times"])
    for u,t in zip(users,times):
        pol_user_news[u].append((row['news_id'],t))


go_temp = defaultdict(list)
for k,v in go_user_news.items():
    for x in sorted(v,key=lambda x:x[1]):
        go_temp[k].append(x[0])


pol_temp = defaultdict(list)
for k,v in pol_user_news.items():
    for x in sorted(v,key=lambda x:x[1]):
        pol_temp[k].append(x[0])


json.dump(go_temp, open("datas/user_interaction/gossip_user_interaction.json","w"))
json.dump(pol_temp, open("datas/user_interaction/pol_user_interaction.json","w"))
