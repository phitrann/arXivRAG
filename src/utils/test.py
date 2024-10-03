import os
from embedder import InstructorEmbeddings

os.environ["NO_PROXY"] = "172.16.87.76"
embedder = InstructorEmbeddings(uri="http://172.16.87.76:8081",model_name="BAAI/llm-embedder")
query = "Encode this query and context for searching relevant passages: Are tomatoes classified as vegetables?"
context1 = "Represent this document for retrieval: They are often mistaken for vegetables due to their culinary use, but botanically, they are classified as fruits."
context2 = "Represent this document for retrieval: Tomatoes come in various colors, including red, yellow, orange, and even purple. They are rich in vitamins A and C and are widely used in cooking sauces, salads, and soups."
context3 = "Represent this document for retrieval: The Leaning Tower of Pisa is famous for its tilt, but contrary to popular belief, it has never fallen over despite its lean. Recent efforts have stabilized the tower to prevent further tilting."
context_dummy = "Represent this document for retrieval: This is a dummy document. t o m m m t to toes"
# print(embedder._get_query_embedding(query))
# print(embedder._get_text_embedding(key))
# print(embedder._get_text_embeddings([key, key]))

query_embedding = embedder._get_query_embedding(query)
context1_embedding = embedder._get_text_embedding(context1)
context2_embedding = embedder._get_text_embedding(context2)
context3_embedding = embedder._get_text_embedding(context3)
context4_embedding = embedder._get_text_embedding(context_dummy)

# Cosine similarity
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(cosine_similarity(query_embedding, context1_embedding))
print(cosine_similarity(query_embedding, context2_embedding))
print(cosine_similarity(query_embedding, context3_embedding))
print(cosine_similarity(query_embedding, context4_embedding))

# Inner product

def inner_product(a, b):
    return np.dot(a, b)

print(inner_product(query_embedding, context1_embedding))
print(inner_product(query_embedding, context2_embedding))
print(inner_product(query_embedding, context3_embedding))
print(inner_product(query_embedding, context4_embedding))