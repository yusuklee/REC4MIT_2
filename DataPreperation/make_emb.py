import argparse
import os
from os.path import exists

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

TITLE_LEN = 32
DESC_LEN = 128

model = SentenceTransformer("bert-base-uncased")

gossip = pd.read_csv("datas/news/gossip.csv")
pol= pd.read_csv("datas/news/pol.csv")



gossip["title"] = gossip["title"].fillna("").astype(str)
gossip["description"] = gossip["description"].fillna("").astype(str)

gossip_ids =  gossip["news_id"].astype(str).tolist()
titles = []
for t in gossip["title"]:
    if t.strip():
        titles.append(t.strip())
    else:
        titles.append("unknown news")

has_text = gossip["description"].str.len() >5
embed = lambda t: model.encode(t, batch_size=128, normalize_embeddings=True,
                                 convert_to_numpy=True).astype(np.float32)

model.max_seq_length = TITLE_LEN
T = embed(titles)

model.max_seq_length= DESC_LEN
D = np.zeros((len(gossip), 768),dtype=np.float32)
text_idx = np.where(has_text.values)[0].tolist()
if text_idx:
    D[text_idx] = embed([gossip["description"].iloc[i] for i in text_idx])

os.makedirs("datas/emb", exist_ok=True)
np.savez(f"datas/emb/gossip.npz", news_id=np.array(gossip_ids), title = T, description=D)






pol["title"] = pol["title"].fillna("").astype(str)
pol["description"] = pol["description"].fillna("").astype(str)

pol_ids =  pol["news_id"].astype(str).tolist()
titles = []
for t in pol["title"]:
    if t.strip():
        titles.append(t.strip())
    else:
        titles.append("unknown news")

has_text = pol["description"].str.len() >5
embed = lambda t: model.encode(t, batch_size=128, normalize_embeddings=True,
                                 convert_to_numpy=True).astype(np.float32)

model.max_seq_length = TITLE_LEN
T = embed(titles)

model.max_seq_length= DESC_LEN
D = np.zeros((len(pol), 768),dtype=np.float32)
text_idx = np.where(has_text.values)[0].tolist()
if text_idx:
    D[text_idx] = embed([pol["description"].iloc[i] for i in text_idx])

os.makedirs("datas/emb", exist_ok=True)
np.savez(f"datas/emb/pol.npz", news_id=np.array(pol_ids), title = T, description=D)


















