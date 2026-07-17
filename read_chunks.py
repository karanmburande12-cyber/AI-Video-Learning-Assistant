import requests
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib


def create_embedding(text_list, batch_size=32):
    """
    Creates embeddings for a list of texts using Ollama's /api/embed endpoint.
    Batches requests to avoid overly large payloads.
    https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    """
    all_embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        r = requests.post("http://localhost:11434/api/embed", json={
            "model": "bge-m3",
            "input": batch
        })
        r.raise_for_status()
        data = r.json()
        if "embeddings" not in data:
            raise RuntimeError(f"Unexpected response from Ollama: {data}")
        all_embeddings.extend(data["embeddings"])
    return all_embeddings


jsons = os.listdir("jsons")  # List all the jsons
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}", encoding="utf-8") as f:
        content = json.load(f)

    print(f"Creating Embeddings for {json_file}")
    texts = [c['text'] for c in content['chunks']]
    embeddings = create_embedding(texts)

    assert len(embeddings) == len(content['chunks']), (
        f"{json_file}: expected {len(content['chunks'])} embeddings, "
        f"got {len(embeddings)}"
    )

    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)

df = pd.DataFrame.from_records(my_dicts)
# print(df)

# Save this dataframe so the embeddings aren't lost after the script exits.
joblib.dump(df, 'embeddings.joblib')

# a = create_embedding(["Cat sat on the mat", "Harry dances on a mat"])
# print(a)

