from qdrant_client import QdrantClient

import torch
import numpy as np
import pandas as pd


def create_connection():
    """
    Init connection to qdrant db & searh engine

    """
    client = QdrantClient(host='qdrant', port=6333)
    return client


def retrieve_embeddings(df, model, batch_size=256):
    """
    Function for embeddings retrieving for all items in df
    Args:
    - df: df with "title_model" col for each item
    - model: model to use for embeddings
    - batch_size: how many samples to process at once

    Return:
    - matrix of df embeddings

    """
    model.eval()
    df = df.loc[:, 'title_model'].tolist()

    num_batches = (len(df) // batch_size) + (len(df) % batch_size > 0)

    with torch.no_grad():
        embeds = []
        for i in range(num_batches):
            start, end = i * batch_size, (i + 1) * batch_size
            batch = df[start:end]

            embeds.append(model(batch).cpu().numpy())

    return np.concatenate(embeds, axis=0)


def main():
    from qdrant_client.http.models import Distance, VectorParams, OptimizersConfigDiff
    from models import create_model


    df = pd.read_feather('./data/search_relevance_dataset_v1.feather').iloc[:10000]
    df['title_model'] = df.title.apply(lambda text: ' '.join(text.lower().split()))
    df['title_model'] = 'объявление: ' + df['title_model']

    model = create_model()

    embeddings = retrieve_embeddings(df, model)
    payload = [
        {
            'title': row.title,
            'description': row.description,
            'keywords': row.keywords,
        }
        for row in df.itertuples()
    ]

    client = create_connection()

    client.recreate_collection(
        collection_name='test_collection',
        vectors_config=VectorParams(
            size=embeddings.shape[1],
            distance=Distance.COSINE,
        ),
        shard_number=2, # parallelize upload of a large dataset
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=0,
        ),
    )

    client.upload_collection(
        collection_name='test_collection',
        vectors=embeddings,
        payload=payload,
        ids=None, # Vector ids will be assigned automatically
        batch_size=2048,
    )

    client.update_collection(
        collection_name='test_collection',
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20000,
        ),
    )


if __name__ == '__main__':
    main()
