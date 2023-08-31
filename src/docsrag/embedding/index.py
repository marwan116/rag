"""Vector Store Index with Ray parallelization."""
import json
from functools import cached_property, partial
from typing import Any, Sequence

import joblib
import ray
import torch
from llama_index.data_structs.data_structs import IndexDict
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.indices.base import BaseIndex
from llama_index.indices.query.schema import QueryBundle
from llama_index.schema import BaseNode, NodeWithScore
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores.simple import SimpleVectorStore
from pydantic import BaseModel
from sentence_transformers.util import cos_sim

from docsrag.embedding.utils import cosine_similarity, get_embedding, load_embed_model


class VectorStoreSpec(BaseModel):
    """Vector Store."""

    embedding_model_name: str

    def __hash__(self) -> int:
        hash_hex = joblib.hash(self.dict())
        return int(hash_hex, 16)


class VectorStoreIndexRay(BaseIndex[IndexDict]):
    """Vector Store Index with Ray parallelization."""

    def __init__(
        self,
        docstore: SimpleDocumentStore,
        vector_store: SimpleVectorStore,
        index_store: SimpleIndexStore,
        embed_model: LangchainEmbedding,
        embedding_model_name: str,
    ) -> None:
        self._docstore = docstore
        self._vector_store = vector_store
        self._index_store = index_store
        self._embed_model = embed_model
        self._embedding_model_name = embedding_model_name

    @cached_property
    def nodes(self) -> list["BaseNode"]:
        return self._docstore.get_nodes(
            node_ids=list(self._index_store.get_index_struct().nodes_dict.values())
        )

    @cached_property
    def text_hash_to_node(self):
        return {node.metadata["text_hash"]: node for node in self.nodes}

    @classmethod
    def _get_node_embeddings(cls, nodes: Sequence[BaseNode], embed_model, **ray_kwargs):
        """Get node embeddings."""
        ray.init(ignore_reinit_error=True)

        return [
            node_with_embedding
            for node_with_embedding in ray.data.from_items(nodes)
            .map_batches(partial(get_embedding, embed_model=embed_model), **ray_kwargs)
            .iter_rows()
        ]

    @classmethod
    def update_from_nodes(
        cls, nodes, docstore, vector_store, embed_model, index_struct, **ray_kwargs
    ):
        embedding_results = cls._get_node_embeddings(
            nodes=nodes, embed_model=embed_model, **ray_kwargs
        )
        new_ids = vector_store.add(embedding_results)

        for result, new_id in zip(embedding_results, new_ids):
            index_struct.add_node(result.node, text_id=new_id)
            docstore.add_documents([result.node], allow_update=True)

        return docstore, vector_store, index_struct

    @classmethod
    def build_from_spec(cls, nodes, spec: VectorStoreSpec, **ray_kwargs):
        docstore = SimpleDocumentStore()
        vector_store = SimpleVectorStore()
        index_struct = IndexDict()
        embed_model = load_embed_model(spec.embedding_model_name)
        docstore, vector_store, index_struct = cls.update_from_nodes(
            nodes=nodes,
            docstore=docstore,
            vector_store=vector_store,
            embed_model=embed_model,
            index_struct=index_struct,
            **ray_kwargs
        )
        index_store = SimpleIndexStore()
        index_store.add_index_struct(index_struct)
        return cls(
            docstore=docstore,
            vector_store=vector_store,
            index_store=index_store,
            embed_model=embed_model,
            embedding_model_name=spec.embedding_model_name,
        )

    def save(self, dir):
        self._docstore.persist(dir / "docstore.json")
        self._vector_store.persist(dir / "vector_store.json")
        self._index_store.persist(dir / "index_store.json")
        with open(dir / "embedding_model.json", "w") as f:
            json.dump({"embedding_model_name": self._embedding_model_name}, f)

    @classmethod
    def load(cls, persist_dir):
        docstore = SimpleDocumentStore.from_persist_dir(persist_dir)
        index_store = SimpleIndexStore.from_persist_dir(persist_dir)
        vector_store = SimpleVectorStore.from_persist_dir(persist_dir)

        with open(persist_dir / "embedding_model.json", "r") as f:
            embedding_model_name = json.load(f)["embedding_model_name"]
        embed_model = load_embed_model(embedding_model_name)

        return cls(
            docstore=docstore,
            vector_store=vector_store,
            index_store=index_store,
            embed_model=embed_model,
            embedding_model_name=embedding_model_name,
        )

    def _compute_query_embedding_bundle(self, query):
        return QueryBundle(
            query_str=query,
            embedding=self._embed_model.get_agg_embedding_from_queries([query]),
        )

    def compute_similarity(self, query_bundle: QueryBundle, node: "BaseNode"):
        """Compute the similarity between a query and a node."""
        node_embedding = self._vector_store.get(node.node_id)
        return cosine_similarity(query_bundle.embedding, node_embedding)

    def retrieve_most_similiar_nodes(
        self, query: str, similarity_top_k: str
    ) -> Sequence[BaseNode]:
        model = self._embed_model._langchain_embedding
        query_embeddings = torch.tensor([model.embed_query(query)])
        corpus_embeddings = torch.tensor(
            [
                self._vector_store._data.embedding_dict[node.node_id]
                for node in self.nodes
            ]
        )
        pair_scores = cos_sim(query_embeddings, corpus_embeddings)
        pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(
            pair_scores,
            min(similarity_top_k, len(corpus_embeddings)),
            dim=1,
            largest=True,
            sorted=True,
        )

        scores = pair_scores_top_k_values.squeeze().cpu().tolist()
        nodes_idx = pair_scores_top_k_idx.squeeze().cpu().tolist()

        return [
            NodeWithScore(
                node=self.nodes[node_idx],
                score=score,
            )
            for score, node_idx in zip(scores, nodes_idx)
        ]

    # HACK: This is a hack to get around the fact that the base class requires
    # these abstract methods to be implemented, but we don't need them for this
    def _build_index_from_nodes(self, nodes: Sequence[BaseNode]):
        raise NotImplementedError()

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        raise NotImplementedError()

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        raise NotImplementedError()

    def ref_doc_info(self):
        raise NotImplementedError()

    def as_retriever(self, **kwargs: Any):
        raise NotImplementedError()
