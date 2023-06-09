import openai
import pandas as pd
import os

# read your text files into a list
text_chunks = []
for i in range(17):  # if you have files from chunk_0.txt to chunk_16.txt
    with open(f"chunk_{i}.txt", 'r') as f:
        text_chunks.append(f.read())

# calculate embeddings
EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's best embeddings as of Apr 2023
BATCH_SIZE = 1000  # you can submit up to 2048 embedding inputs per request

embeddings = []
for batch_start in range(0, len(text_chunks), BATCH_SIZE):
    batch_end = batch_start + BATCH_SIZE
    batch = text_chunks[batch_start:batch_end]
    print(f"Batch {batch_start} to {batch_end-1}")
    response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
    for i, be in enumerate(response["data"]):
        assert i == be["index"]  # double check embeddings are in same order as input
    batch_embeddings = [e["embedding"] for e in response["data"]]
    embeddings.extend(batch_embeddings)

df = pd.DataFrame({"text": text_chunks, "embedding": embeddings})

df.to_csv('embeddings.csv')

df.to_pickle('embeddings.pkl')
