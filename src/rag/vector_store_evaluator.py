"""
Given the gold standard dataset of questions and answers and the embedding model,
evaluate the model's performance.

Factors that eventually influence performance:
- the top k similarity nodes to retrieve which is influence by:
    - user experience (don't want to overwhelm the user with too many options)
    - maximum context length to pass to LLM (capacity of model to process)
- the chosen similarity threshold
- the selected embedding model

metric - average_similarity_score_for_exact_match
- Compute the similarity score between the question and its source node
- metric: average similarity score for exact match - the closer to 1, the better
- pros: quick to compute, only requires the source node and the question
- cons: does not take into account if source node will be in top k nodes

plot = percent_found_exact_match_given_similarity_threshold
- Compute the similarity score between the question and its source node
- Choose an array of similarity thresholds
- For each similarity threshold, check if the similarity score is above the threshold:
    - if so, count as a match
    - if not, count as a mismatch
- Compute percent of matches for each similarity threshold
- plot: percent found exact match given similarity threshold
the slower the decrease the better

plot - percent_perplexity_given_similarity_threshold
- Compute the similarity score between out of context question and all nodes
- Choose an array of similarity thresholds
- For each similarity threshold, check if the similarity score is above the threshold:
    - if so, count as a match
    - if not, count as a mismatch
- Compute percent of matches for each similarity threshold
- plot: percent perplexity given similarity threshold - the slower increase the better

metric - percent_exact_match_in_top_k
- Compute the distance between the question and all nodes
- Keep the top k nodes with the smallest distance
- metric: percent exact match in top k - the closer to 1, the better
- pros: takes into account if source node will be in top k nodes
- cons: slow to compute, requires the question and all nodes
"""
from typing import Any, TYPE_CHECKING
from pydantic import BaseModel, Field
from pathlib import Path
import pandas as pd
import numpy as np
import os
from functools import cached_property

from rag.utils import get_data_path

if TYPE_CHECKING:
    from llama_index.schema import BaseNode, NodeWithScore


def cosine_similarity(embedding1, embedding2):
    product = np.dot(embedding1, embedding2)
    norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    return product / norm


PandasRow = Any


class VectorStore(BaseModel):
    """Vector store for a given index."""

    class Config:
        arbitrary_types_allowed = True
        keep_untouched = (cached_property,)

    index_id: str
    storage_path: Path
    embedding_model: str
    similarity_top_k: int
    cosine_similarity_threshold: float

    @cached_property
    def vector_store(self):
        from llama_index import StorageContext, load_index_from_storage

        return load_index_from_storage(
            StorageContext.from_defaults(persist_dir=self.storage_path),
            index_id=self.index_id,
        )

    @cached_property
    def embed_model(self):
        from llama_index.embeddings.langchain import LangchainEmbedding
        from langchain.embeddings import HuggingFaceBgeEmbeddings
        from llama_index.utils import get_cache_dir

        cache_folder = os.path.join(get_cache_dir(), "models")
        return LangchainEmbedding(
            HuggingFaceBgeEmbeddings(
                model_name=self.embedding_model,
                cache_folder=cache_folder,
            )
        )

    @property
    def nodes(self) -> list["BaseNode"]:
        return [
            self.vector_store.docstore.get_node(node_id)
            for node_id in self.vector_store.index_struct.nodes_dict
        ]

    @property
    def text_hash_to_node(self):
        return {node.metadata["text_hash"]: node for node in self.nodes}

    @cached_property
    def retriever(self):
        return self.vector_store.as_retriever(similarity_top_k=10)

    def _compute_query_embedding_bundle(self, query):
        from llama_index.indices.query.schema import QueryBundle

        return QueryBundle(
            query_str=query,
            embedding=self.embed_model.get_agg_embedding_from_queries([query]),
        )

    def query(self, query: str) -> list["NodeWithScore"]:
        """Returns a list of nodes."""
        query_bundle = self._compute_query_embedding_bundle(query)
        nodes_with_scores = self.retriever.retrieve(query_bundle)

        return list(
            sorted(
                [
                    node_with_score.node.metadata["file_path"]
                    for node_with_score in nodes_with_scores
                    if node_with_score.score >= self.cosine_similarity_threshold
                ],
                key=lambda node_with_score: -1 * node_with_score.score,
            )
        )[: self.similarity_top_k]

    def _lookup_node_embedding(self, node: "BaseNode") -> list[float]:
        return self.vector_store._vector_store.get(node.node_id)

    def compute_similarity(self, query: str, node: "BaseNode"):
        """Compute the similarity between a query and a node."""
        query_bundle = self._compute_query_embedding_bundle(query)
        node_embedding = self._lookup_node_embedding(node)
        return cosine_similarity(query_bundle.embedding, node_embedding)


class VectorStoreEvaluator(BaseModel):
    """Evaluate the performance of a vector store."""

    class Config:
        arbitrary_types_allowed = True
        keep_untouched = (cached_property,)

    vector_store: VectorStore
    evaluation_dataset_name: str
    evaluation_dataset_dir: str = Field(default=get_data_path() / "eval_data")

    def _load_evaluation_dataset(self):
        path = Path(self.evaluation_dataset_dir) / self.evaluation_dataset_name
        return pd.read_parquet(path, columns=["question", "answer", "text_hash"])

    def _compute_similarity_score_between_question_and_source_nodes(
        self, rows: list[PandasRow]
    ):
        similarities_with_source_node = []
        for row in rows:
            source_node = self.vector_store.text_hash_to_node[row.text_hash]
            similarity = self.vector_store.compute_similarity(
                query=row.question, node=source_node
            )
            similarities_with_source_node.append(similarity)

        return similarities_with_source_node

    def _compute_similarity_score_between_question_and_all_nodes(self, question):
        for node in self.vector_store.nodes:
            self.vector_store.similarity(question, node)

    def run(self):
        import ray
        import matplotlib.pyplot as plt

        df = self._load_evaluation_dataset()

        non_empty_answer = (df["answer"] != "") & (df["answer"].notnull())
        non_empty_hash = (df["text_hash"] != "") & (df["text_hash"].notnull())
        df_without_noisy_questions = df[non_empty_answer & non_empty_hash]

        similarity_score_exact_match = np.array(
            [
                similarities
                for similarities in ray.data.from_items(
                    list(df_without_noisy_questions.itertuples())
                )
                .map_batches(
                    self._compute_similarity_score_between_question_and_source_nodes,
                    batch_size=500,
                )
                .iter_rows()
            ]
        )
        
        pd.DataFrame(
            {"similarity_score_exact_match": similarity_score_exact_match}
        ).to_parquet("similarity_score_exact_match.parquet")

        avg_score = np.mean(similarity_score_exact_match)
        print(f"Average similarity score for exact match: {avg_score}")

        similarity_thresholds = np.linspace(0, 1, 100)
        pct_found_exact_match_given_similarity_threshold = []
        for threshold in similarity_thresholds:
            pct_found_exact_match_given_similarity_threshold.append(
                (similarity_score_exact_match >= threshold).mean() * 100
            )

        _, ax = plt.subplots(figsize=(10, 10))
        ax.set_title("Percent found exact match given similarity threshold")
        ax.plot(similarity_thresholds, pct_found_exact_match_given_similarity_threshold)
        ax.set_xlabel("Similarity threshold")
        ax.set_ylabel("Percent found exact match")
        plt.show()

        # df_noisy_questions = df[~(non_empty_answer & non_empty_hash)]
        # similarity_score_perplexity = np.array(
        #     [
        #         similarities
        #         for similarities in ray.data.from_items(
        #             list(df_noisy_questions.itertuples())
        #         )
        #         .map_batches(
        #             self._compute_similarity_score_between_question_and_all_nodes,
        #             batch_size=500,
        #         )
        #         .iter_rows()
        #     ]
        # )
