import requests
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib



def create_embedding(text):
    r = requests.post("http://localhost:11434/api/embeddings", json={
        "model": "bge-m3",
        "prompt": text   # ✅ correct key
    })

    res = r.json()
    return res.get("embedding", [])

# def create_embedding(text_list):
#     # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
#     r = requests.post("http://localhost:11434/api/embed", json={
#         "model": "bge-m3",
#         "input": text_list
#     })

#     #embedding = r.json()["embeddings"] 
#     res = r.json() 
#     #embedding = res["embeddings"] if "embeddings" in res else res["embedding"]
#     embedding = res.get("embeddings") or res.get("embedding") or []
#     return embedding


jsons = os.listdir("jsons")  # List all the jsons 
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")
    #embeddings = create_embedding([c['text'] for c in content['chunks']])
    #embeddings = create_embedding([c['text'] for c in content['chunks']])
    texts = [c['text'] for c in content['chunks']]
    #embeddings = [create_embedding(t)[0] for t in texts]
    embeddings = [create_embedding(t) for t in texts]

    if not embeddings:
      print(f"Skipping {json_file} due to embedding error")
      continue
       
    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk) 

df = pd.DataFrame.from_records(my_dicts)
# Save this dataframe
joblib.dump(df, 'embeddings.joblib')

