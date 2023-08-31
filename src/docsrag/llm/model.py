from pathlib import Path

from llama_index.llm_predictor import LLMPredictor
from llama_index.llms.base import ChatMessage
from llama_index.llms.openai import OpenAI

from docsrag.embedding.index import VectorStoreIndexRay


class LLM:
    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        max_retries: int,
    ) -> None:
        self.predictor = LLMPredictor(
            llm=OpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )
        )

    def query(self, query):
        response = self.predictor.llm.chat([ChatMessage(content=query)])
        return response.message.content


class LLMPlusRag:
    def __init__(
        self,
        vector_store_path: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        max_tokens: int = 1000,
        max_retries: int = 10,
    ) -> None:
        self.underlying_llm = LLM(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
        self.vector_store_index = VectorStoreIndexRay.load(Path(vector_store_path))

    def query(self, query, similarity_top_k):
        qa_prompt_tmpl_str = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        )

        nodes_with_scores = self.vector_store_index.retrieve_most_similiar_nodes(
            query=query,
            similarity_top_k=similarity_top_k,
        )

        text_chunks = [
            node_with_score.node.text for node_with_score in nodes_with_scores
        ]

        context_str = "\n".join(text_chunks)

        return self.underlying_llm.query(
            qa_prompt_tmpl_str.format(
                context_str=context_str,
                query_str=query,
            )
        )
