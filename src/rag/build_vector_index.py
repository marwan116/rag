from llama_index import VectorStoreIndex, ServiceContext, StorageContext
from llama_index.utils import get_cache_dir
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings import (
    HuggingFaceBgeEmbeddings,
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
)
from llama_index.vector_stores.types import NodeWithEmbedding
from llama_index.callbacks import CallbackManager
from typing import Dict
import os

if __name__ == "__main__":
    from pathlib import Path
    import yaml
    import pickle
    from rag.generate_nodes import NodeParser
    from llama_index.llm_predictor.mock import MockLLMPredictor
    from llama_index.indices.prompt_helper import PromptHelper
    from unittest.mock import MagicMock

    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    parser = NodeParser.model_validate(config["generate_nodes"])
    nodes_dir = Path(__file__).parent / "nodes"
    nodes_dir.mkdir(exist_ok=True)
    nodes_path = nodes_dir / f"{hash(parser)}.pkl"

    if not nodes_path.exists():
        raise ValueError(
            f"Nodes not found at {nodes_path}. Run generate_nodes.py to generate them."
        )

    with open(nodes_path, "rb") as f:
        nodes = pickle.load(f)

    nodes = [
        node
        for node in nodes[:200]
        if node.metadata["file_path"].endswith((".rst", ".md"))
    ]

    cache_folder = os.path.join(get_cache_dir(), "models")
    embed_model = LangchainEmbedding(
        HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en",
            cache_folder=cache_folder,
        )
    )

    service_context = ServiceContext(
        llm_predictor=MockLLMPredictor(),
        embed_model=embed_model,
        prompt_helper=MagicMock(),
        node_parser=MagicMock(),
        llama_logger=None,
        callback_manager=CallbackManager(),
    )

    persist_dir = Path(__file__).parent / "vector_store_index_beta"
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)

    vector_store_index = VectorStoreIndex(
        nodes=nodes,
        embed_model=embed_model,
        service_context=service_context,
        storage_context=storage_context,
        show_progress=True,
    )
    vector_store_index.storage_context.persist(persist_dir=persist_dir)
