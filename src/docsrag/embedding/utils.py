import numpy as np
from llama_index.schema import MetadataMode
from llama_index.vector_stores.types import NodeWithEmbedding


def cosine_similarity(embedding1, embedding2):
    product = np.dot(embedding1, embedding2)
    norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    return product / norm


def _mean_agg(embeddings: list[list[float]]) -> list[float]:
    """Mean aggregation for embeddings."""
    return list(np.array(embeddings).mean(axis=0))


def get_embedding(nodes, *, embed_model):
    """Get embedding."""
    return [
        NodeWithEmbedding(
            node=node,
            embedding=_mean_agg(
                [
                    embed_model._langchain_embedding.embed_query(
                        node.text
                    )
                ]
            ),
        )
        for node in nodes
    ]


def load_embed_model(embedding_model_name):
    import os
    from langchain.embeddings import HuggingFaceBgeEmbeddings
    from llama_index.embeddings.langchain import LangchainEmbedding
    from llama_index.utils import get_cache_dir

    cache_folder = os.path.join(get_cache_dir(), "models")
    return LangchainEmbedding(
        HuggingFaceBgeEmbeddings(
            model_name=embedding_model_name,
            cache_folder=cache_folder,
        )
    )
