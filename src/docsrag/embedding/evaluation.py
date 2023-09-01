"""Evaluate the performance of a vector store."""
import logging
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, Field
from sentence_transformers.util import cos_sim
from torch import Tensor

from docsrag.embedding.index import VectorStoreIndexRay
from docsrag.utils import get_data_path

PandasRow = Any

logger = logging.getLogger(__name__)


class InformationRetrievalEvaluator:
    """
    This class evaluates an Information Retrieval (IR) setting.

    Given a set of queries and a large corpus set.

    It measures the following metrics given a top-k retrieval setting:
        - Recall @ top k
        - Mean Reciprocal Rank (MRR) @ top k
        - Normalized Discounted Cumulative Gain (NDCG) @ top k

    It also computes the following metric irrespective of any setting:
    - Average Similarity Score
    """

    def __init__(
        self,
        queries: dict[str, str],  # qid => query
        corpus: dict[str, str],  # cid => doc
        relevant_docs: dict[str, set[str]],  # qid => set[cid]
        mrr_at_k: list[int] = [10],
        ndcg_at_k: list[int] = [10],
        recall_at_k: list[int] = [1, 3, 5, 10],
        show_progress_bar: bool = False,
        batch_size: int = 32,
        name: str = "",
        write_csv: bool = True,
        score_functions: list[Callable[[Tensor, Tensor], Tensor]] = {
            "cos_sim": cos_sim,
        },
        main_score_function: str = "cos_sim",
    ):
        self.queries_ids = []
        for qid in queries:
            if qid in relevant_docs and len(relevant_docs[qid]) > 0:
                self.queries_ids.append(qid)

        self.queries = [queries[qid] for qid in self.queries_ids]

        self.corpus_ids = list(corpus.keys())
        self.corpus = [corpus[cid] for cid in self.corpus_ids]

        self.relevant_docs = relevant_docs
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.recall_at_k = recall_at_k
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name
        self.write_csv = write_csv
        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys()))
        self.main_score_function = main_score_function

        if name:
            name = "_" + name

        self.csv_file: str = "Information-Retrieval_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps"]

        for score_name in self.score_function_names:
            for k in recall_at_k:
                self.csv_headers.append("{}-Recall@{}".format(score_name, k))

            for k in mrr_at_k:
                self.csv_headers.append("{}-MRR@{}".format(score_name, k))

            for k in ndcg_at_k:
                self.csv_headers.append("{}-NDCG@{}".format(score_name, k))

    def compute_metrices(
        self, model, corpus_embeddings: Tensor = None
    ) -> dict[str, float]:
        max_k = max(
            max(self.mrr_at_k),
            max(self.ndcg_at_k),
            max(self.recall_at_k),
        )

        # Compute embedding for the queries
        query_embeddings = torch.tensor(
            [model.embed_query(text) for text in self.queries],
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).float()

        queries_result_list = {}
        for name in self.score_functions:
            queries_result_list[name] = [[] for _ in enumerate(query_embeddings)]

        # Compute cosine similarites
        avg_similarity_score = {}
        for name, score_function in self.score_functions.items():
            pair_scores = score_function(query_embeddings, corpus_embeddings)

            query_to_relevant_node = {
                k: self.corpus_ids.index(list(v)[0])
                for k, v in self.relevant_docs.items()
            }

            query_to_relevant_node_scores = pair_scores[
                [query_idx for query_idx in self.queries_ids],
                [query_to_relevant_node[query_idx] for query_idx in self.queries_ids],
            ]

            avg_similarity_score[name] = query_to_relevant_node_scores.mean()

            # Get top-k values
            pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(
                pair_scores,
                min(max_k, len(self.corpus)),
                dim=1,
                largest=True,
                sorted=False,
            )
            pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
            pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(query_embeddings)):
                for sub_corpus_id, score in zip(
                    pair_scores_top_k_idx[query_itr],
                    pair_scores_top_k_values[query_itr],
                ):
                    corpus_id = self.corpus_ids[sub_corpus_id]
                    queries_result_list[name][query_itr].append(
                        {"corpus_id": corpus_id, "score": score}
                    )

            # TODO - Get scores above a threshold

        # Compute scores @ k
        scores = {
            name: {
                **self.compute_metrics_at_k(queries_result_list[name]),
                **{"avg_similarity_score": avg_similarity_score[name]},
            }
            for name in self.score_functions
        }

        # Log scores
        for name in self.score_function_names:
            self.log_scores(scores[name])

        return scores

    def compute_metrics_at_k(self, queries_result_list: list[object]):
        # Init score computation values
        recall_at_k = {k: [] for k in self.recall_at_k}
        MRR = {k: 0 for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}

        # Compute scores on results
        for query_itr in range(len(queries_result_list)):
            query_id = self.queries_ids[query_itr]

            # Sort scores
            top_hits = sorted(
                queries_result_list[query_itr], key=lambda x: x["score"], reverse=True
            )
            query_relevant_docs = self.relevant_docs[query_id]

            # Recall@k
            for k_val in self.recall_at_k:
                num_correct = 0
                for hit in top_hits[0:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1

                recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

            # MRR@k
            for k_val in self.mrr_at_k:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        MRR[k_val] += 1.0 / (rank + 1)
                        break

            # NDCG@k
            for k_val in self.ndcg_at_k:
                predicted_relevance = [
                    1 if top_hit["corpus_id"] in query_relevant_docs else 0
                    for top_hit in top_hits[0:k_val]
                ]
                true_relevances = [1] * len(query_relevant_docs)

                ndcg_value = self.compute_dcg_at_k(
                    predicted_relevance, k_val
                ) / self.compute_dcg_at_k(true_relevances, k_val)
                ndcg[k_val].append(ndcg_value)

        # Compute averages
        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k])

        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k])

        for k in MRR:
            MRR[k] /= len(self.queries)

        return {
            "recall@k": recall_at_k,
            "ndcg@k": ndcg,
            "mrr@k": MRR,
        }

    def log_scores(self, scores):
        for k in scores["recall@k"]:
            logger.info("Recall@{}: {:.2f}%".format(k, scores["recall@k"][k] * 100))

        for k in scores["mrr@k"]:
            logger.info("MRR@{}: {:.4f}".format(k, scores["mrr@k"][k]))

        for k in scores["ndcg@k"]:
            logger.info("NDCG@{}: {:.4f}".format(k, scores["ndcg@k"][k]))

    @staticmethod
    def compute_dcg_at_k(relevances, k):
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  # +2 as we start our idx at 0
        return dcg


def load_evaluation_dataset(
    evaluation_dataset_dir, evaluation_dataset_name, limit: Optional[int]
):
    """Load evaluation dataset."""
    path = Path(evaluation_dataset_dir) / evaluation_dataset_name
    df = pd.read_parquet(path, columns=["question", "answer", "text_hash"])
    non_empty_answer = (df["answer"] != "") & (df["answer"].notnull())
    non_empty_hash = (df["text_hash"] != "") & (df["text_hash"].notnull())
    df_without_noisy_questions = df[non_empty_answer & non_empty_hash].reset_index(
        drop=True
    )
    if limit:
        return df_without_noisy_questions.iloc[:limit]
    return df_without_noisy_questions


class VectorStoreEvaluator(BaseModel):
    """Evaluate the performance of a vector store."""

    class Config:
        arbitrary_types_allowed = True
        keep_untouched = (cached_property,)

    vector_store_index: VectorStoreIndexRay
    top_ks: list[int]

    def run(self, eval_df: pd.DataFrame):
        queries = {row.Index: row.question for row in eval_df.itertuples()}

        corpus = {node.node_id: node.text for node in self.vector_store_index.nodes}

        relevant_docs = {
            row.Index: {
                self.vector_store_index.text_hash_to_node[row.text_hash].node_id
            }
            for row in eval_df.itertuples()
        }

        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            mrr_at_k=self.top_ks,
            ndcg_at_k=self.top_ks,
            recall_at_k=self.top_ks,
            show_progress_bar=True,
            name=self.vector_store_index._embedding_model_name,
            score_functions={"cos_sim": cos_sim},
            main_score_function=cos_sim,
        )

        corpus_embeddings = torch.tensor(
            [
                self.vector_store_index._vector_store._data.embedding_dict[node.node_id]
                for node in self.vector_store_index.nodes
            ],
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).float()

        model = self.vector_store_index._embed_model._langchain_embedding

        metrics = evaluator.compute_metrices(
            model=model, corpus_embeddings=corpus_embeddings
        )

        return (
            pd.DataFrame(metrics["cos_sim"])
            .reset_index()
            .rename(columns={"index": "top_k"})
        )
