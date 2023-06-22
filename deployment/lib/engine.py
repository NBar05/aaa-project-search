from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, OptimizersConfigDiff

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any


def create_connection() -> QdrantClient:
    """
    Initialize connection to qdrant db & search engine.

    Returns:
        QdrantClient: The initialized client object.
    """
    client = QdrantClient(host='qdrant', port=6333)
    return client


def retrieve_embeddings(df: pd.DataFrame, model: Any, tokenizer: Any, batch_size: int = 512) -> np.ndarray:
    """
    Function for retrieving embeddings for all items in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with "title_model" column for each item.
        model (Any): The model to use for embeddings.
        tokenizer (Any): The tokenizer to use for tokenization.
        batch_size (int, optional): Number of samples to process at once.

    Returns:
        np.ndarray: Matrix of embeddings for the DataFrame.
    """
    df = df['title'].tolist()

    num_batches = (len(df) // batch_size) + (len(df) % batch_size > 0)

    embeds = []
    with torch.no_grad():
        for i in range(num_batches):
            start, end = i * batch_size, (i + 1) * batch_size
            batch = df[start:end]

            batch = tokenizer(batch, type_of_text='item')

            embeds.append(model(batch))

    return np.concatenate(embeds, axis=0)


def main():
    from models import create_model

    df = pd.read_feather('./data/search_relevance_dataset_v1.feather')

    tokenizer, model = create_model()

    embeddings = retrieve_embeddings(df, model, tokenizer)
    payload: List[Dict[str, str]] = [
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
        shard_number=2,
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=0,
        ),
    )

    client.upload_collection(
        collection_name='test_collection',
        vectors=embeddings,
        payload=payload,
        ids=None,
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
