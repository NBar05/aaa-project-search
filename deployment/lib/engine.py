from qdrant_client import QdrantClient


def create_connection():
    client = QdrantClient(host='qdrant', port=6333)
    return client


if __name__ == '__main__':
    import os
    
    from tqdm import tqdm
    import pandas as pd

    from models import create_model
    from qdrant_client.http.models import Distance, VectorParams, CollectionStatus, PointStruct


    df = pd.read_hdf('./data/search_relevance_dataset_v1.hdf', 'table')
    df.drop(columns=['query_id', 'query_category_id', 'query_microcat_id', 'query_location_id'], inplace=True)
    df = df.drop_duplicates(subset=['item_id']).reset_index(drop=True)

    df['title_model'] = df.title.apply(lambda text: ' '.join(text.lower().split()))
    df['title_model'] = 'объявление: ' + df['title_model']

    print()
    print('Shape of dataset to index:', df.shape)
    print()

    client = QdrantClient(host='qdrant', port=6333)

    client.recreate_collection(
        collection_name='test_collection', 
        vectors_config=VectorParams(size=64, distance=Distance.COSINE),
    )

    collection_info = client.get_collection(collection_name='test_collection')

    assert collection_info.status == CollectionStatus.GREEN
    assert collection_info.vectors_count == 0

    model = create_model()

    points = []
    for row in tqdm(df.itertuples(), total=len(df)):
        points.append(
            PointStruct(
                id=row.Index,
                vector=list(map(float, model(row.title_model))), # list(map(float.. to make proper type
                payload={
                    'title': row.title,
                    'description': row.description,
                    'keywords': row.keywords,
                },
            )
        )
        if len(points) % 2000 == 0:
            operation_info = client.upsert(
                collection_name='test_collection',
                wait=True,
                points=points
            )
            points = []

    operation_info = client.upsert(
        collection_name='test_collection',
        wait=True,
        points=points
    )

    print()
    print('Dataset is indexed')
    print()
