"""Vector Store."""
import os
from functools import cached_property

import joblib
from pydantic import BaseModel  # , ConfigDict


class VectorStore(BaseModel):
    """Vector Store."""

    # model_config = ConfigDict(arbitrary_types_allowed=True)

    class Config:
        arbitrary_types_allowed = True
        keep_untouched = (cached_property,)

    embedding_model_name: str

    @cached_property
    def embedding_model(self):
        from langchain.embeddings import HuggingFaceBgeEmbeddings
        from llama_index.embeddings.langchain import LangchainEmbedding
        from llama_index.utils import get_cache_dir

        cache_folder = os.path.join(get_cache_dir(), "models")
        return LangchainEmbedding(
            HuggingFaceBgeEmbeddings(
                model_name=self.embedding_model_name,
                cache_folder=cache_folder,
            )
        )

    @cached_property
    def service_context(self):
        from unittest.mock import MagicMock

        from llama_index.callbacks import CallbackManager
        from llama_index.indices.service_context import ServiceContext
        from llama_index.llm_predictor.mock import MockLLMPredictor

        return ServiceContext(
            llm_predictor=MockLLMPredictor(),  # not needed
            embed_model=self.embedding_model,
            prompt_helper=MagicMock(),  # not needed
            node_parser=MagicMock(),  # not needed
            llama_logger=None,
            callback_manager=CallbackManager(),
        )

    def update(self, nodes, persist_dir, num_gpus, batch_size):
        """Update the vector store."""
        from llama_index.storage.storage_context import StorageContext

        from docsrag.vector_store_index import VectorStoreIndexRay

        storage_context = StorageContext.from_defaults()
        vector_store_index = VectorStoreIndexRay(
            nodes=nodes,
            embed_model=self.embedding_model,
            service_context=self.service_context,
            storage_context=storage_context,
            show_progress=True,
            num_gpus=num_gpus,
            batch_size=batch_size,
        )
        vector_store_index.storage_context.persist(persist_dir=persist_dir)

    def __hash__(self) -> int:
        hash_hex = joblib.hash(self.dict())
        return int(hash_hex, 16)
