from flask import Flask, request

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModel

from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, CollectionStatus, PointStruct


def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy() # (312, )


df = pd.read_hdf('./data/search_relevance_dataset_v1.hdf', 'table').sample(n=1_000)
df.drop(columns=['query_id', 'query_category_id', 'query_microcat_id', 'query_location_id', 'item_id'], inplace=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny2')
model = AutoModel.from_pretrained('cointegrated/rubert-tiny2').to(device)

print()
print('1')
print()
try:
    client = QdrantClient(host='qdrant', port=6333)
except:
    print('Qdrant server is not launched or port is wrong, use local instead')

print()
print('2')
print()

client.recreate_collection(
    collection_name='test_collection', 
    vectors_config=VectorParams(size=312, distance=Distance.DOT),
)

collection_info = client.get_collection(collection_name='test_collection')

assert collection_info.status == CollectionStatus.GREEN
assert collection_info.vectors_count == 0

print()
print('3')
print()

points = []
for row in tqdm(df.itertuples(), total=len(df)):
    points.append(
        PointStruct(
            id=row.Index,
            vector=list(map(float, embed_bert_cls(row.title, model, tokenizer))), # list(map(float.. to make proper type
            payload={
                'title': row.title,
                'description': row.description,
                'keywords': row.keywords,
            },
        )
    )

print()
print('4')
print()

operation_info = client.upsert(
    collection_name='test_collection',
    wait=True,
    points=points
)
print('Database is ready')


app = Flask(__name__)

@app.route('/searchItemsForQuery/<string:text>')
def search_items_for_query(text):
    
    search_result = client.search(
        collection_name='test_collection',
        query_vector=list(map(float, embed_bert_cls(text, model, tokenizer))), 
        limit=10
    )

    return search_result


if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False # для русского языка
    app.run(host='0.0.0.0', port=8080, debug=True)
