import pandas as pd
import openai
from scipy import spatial
import numpy as np


GPT_MODEL = "gpt-4"  # The actual GPT-4 model identifier once it's available
EMBEDDING_MODEL = "text-embedding-ada-002"
df = pd.read_csv('embeddings.csv')

def strings_ranked_by_relatedness(query: str, df: pd.DataFrame, relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x.flatten(), y.flatten()), top_n: int = 100):
    query_embedding_response = openai.Embedding.create(model=EMBEDDING_MODEL, input=query)
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [(row["text"], relatedness_fn(query_embedding, np.array(row["embedding"].split(',')).astype(np.float32).flatten())) for _, row in df.iterrows()]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def query_message(query: str, df: pd.DataFrame, model: str) -> str:
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Here are some articles relevant to your question.'
    question = f"\n\nQuestion: {query}"
    message = introduction
    return message + question

def ask(query: str, df: pd.DataFrame = df, model: str = GPT_MODEL, print_message: bool = False) -> str:
    message = query_message(query, df, model=model)
    if print_message:
        print(message)
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": message}]
    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0.2)
    response_message = response["choices"][0]["message"]["content"]
    return response_message

while True:
    query = input("Prompt: ")
    response_message = ask(query)
    print("AI's response: ", response_message)