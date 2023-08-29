"""Vector Store Index with Ray parallelization."""
from functools import partial
from typing import Sequence

import numpy as np
import ray
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.schema import BaseNode, MetadataMode
from llama_index.vector_stores.types import NodeWithEmbedding


class VectorStoreIndexRay(VectorStoreIndex):
    """Vector Store Index with Ray parallelization."""

    def _get_node_embedding_results(
        self, nodes: Sequence[BaseNode], show_progress: bool = True
    ):
        """Get node embeddings."""
        ray.init(ignore_reinit_error=True)

        return [
            node_with_embedding
            for node_with_embedding in ray.data.from_items(nodes)
            .map_batches(
                partial(get_embedding, embed_model=self.service_context.embed_model),
                batch_size=100,
            )
            .iter_rows()
        ]


def mean_agg(embeddings: list[list[float]]) -> list[float]:
    """Mean aggregation for embeddings."""
    return list(np.array(embeddings).mean(axis=0))


def get_embedding(nodes, *, embed_model):
    """Get embedding."""
    return [
        NodeWithEmbedding(
            node=node,
            embedding=mean_agg(
                [
                    embed_model._langchain_embedding.embed_query(
                        node.get_content(metadata_mode=MetadataMode.EMBED)
                    )
                ]
            ),
        )
        for node in nodes
    ]
