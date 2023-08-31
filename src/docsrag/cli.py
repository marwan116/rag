import asyncio
import pickle
from pathlib import Path
from typing import Optional

from typer import Typer

app = Typer()


@app.command()
def fetch_documents(
    config_path: Optional[str] = None,
    data_path: Optional[str] = None,
    force_run: bool = False,
):
    """Fetches data from the API."""
    from docsrag.docs_loader import GithubDocumentLoader
    from docsrag.utils import get_data_path, load_config

    config = load_config(config_path)
    doc_fetcher = GithubDocumentLoader.parse_obj(config["fetch_docs"])

    data_path_to_use = Path(data_path) if data_path is not None else get_data_path()
    docs_dir = data_path_to_use / "docs"
    docs_dir.mkdir(exist_ok=True, parents=True)
    docs_path = docs_dir / f"{hash(doc_fetcher)}.pkl"

    if not docs_path.exists() or force_run:
        documents = asyncio.run(doc_fetcher.run())
        with open(docs_path, "wb") as f:
            pickle.dump(documents, f)


@app.command()
def parse_nodes(
    config_path: Optional[str] = None,
    data_path: Optional[str] = None,
    force_run: bool = False,
):
    """Parses nodes from documents."""
    from docsrag.docs_loader import GithubDocumentLoader
    from docsrag.node_parser import NodeParser
    from docsrag.utils import get_data_path, load_config

    config = load_config(config_path)

    loader = GithubDocumentLoader.parse_obj(config["fetch_docs"])

    data_path_to_use = Path(data_path) if data_path is not None else get_data_path()
    docs_dir = data_path_to_use / "docs"
    docs_dir.mkdir(exist_ok=True, parents=True)
    docs_path = docs_dir / f"{hash(loader)}.pkl"

    if not docs_path.exists():
        raise ValueError(
            f"Docs not found at {docs_path}. "
            "Run `docsrag fetch-documents` to generate them."
        )

    parser = NodeParser.parse_obj(config["generate_nodes"])
    nodes_dir = data_path_to_use / "nodes"
    nodes_dir.mkdir(exist_ok=True, parents=True)
    nodes_path = nodes_dir / f"{hash(parser)}.pkl"

    if not nodes_path.exists() or force_run:
        with open(docs_path, "rb") as f:
            documents = pickle.load(f)

        nodes = parser.run(documents)

        with open(nodes_path, "wb") as f:
            pickle.dump(nodes, f)


@app.command()
def build_embedding_vector_store_index(
    config_path: Optional[str] = None,
    data_path: Optional[str] = None,
    force_run: bool = False,
):
    """Takes generated nodes, produces their embedding and stores it in an index."""
    from docsrag.embedding.index import VectorStoreIndexRay, VectorStoreSpec
    from docsrag.node_parser import NodeParser
    from docsrag.utils import get_data_path, load_config

    config = load_config(config_path)
    parser = NodeParser.parse_obj(config["generate_nodes"])

    data_path_to_use = Path(data_path) if data_path is not None else get_data_path()
    nodes_dir = data_path_to_use / "nodes"
    nodes_dir.mkdir(exist_ok=True, parents=True)

    nodes_path = nodes_dir / f"{hash(parser)}.pkl"

    if not nodes_path.exists():
        raise ValueError(
            f"Nodes not found at {nodes_path}. "
            "Run `docsrag parse-nodes` to generate them."
        )

    with open(nodes_path, "rb") as f:
        nodes = pickle.load(f)

    vector_store_spec = VectorStoreSpec.parse_obj(config["build_vector_store"])
    vector_store_dir = data_path_to_use / "vector_store"
    vector_store_path = vector_store_dir / f"{hash(vector_store_spec)}"

    if not vector_store_path.exists() or force_run:
        vector_store_path.mkdir(exist_ok=True, parents=True)
        index = VectorStoreIndexRay.build_from_spec(nodes=nodes, spec=vector_store_spec)
        index.save(vector_store_path)


@app.command()
def query(
    self, query: str, config_path: Optional[str] = None, data_path: Optional[str] = None
):
    """Queries configured LLM model with vector store augmentation."""
    from docsrag.utils import load_config
    from docsrag.vector_store_builder import VectorStore

    config = load_config(config_path)
    vector_store = VectorStore.parse_obj(config["build_vector_store"])

    # step 1 given query fetch nodes
    vector_store.query(query)

    # step 2 postprocess nodes (optional)

    # step 3 update prompt with nodes + query

    # step 4 send compacted text to LLM model

    # step 5 return response


@app.command()
def generate_evaluation_dataset(
    config_path: Optional[str] = None, data_path: Optional[str] = None
):
    """Generates the evaluation dataset."""
    from docsrag.evaluation_dataset_generator import EvaluationDatasetBuilder
    from docsrag.node_parser import NodeParser
    from docsrag.utils import get_data_path, load_config

    config = load_config(config_path)

    parser = NodeParser.parse_obj(config["generate_nodes"])
    data_path_to_use = Path(data_path) if data_path is not None else get_data_path()
    nodes_dir = data_path_to_use / "nodes"
    nodes_dir.mkdir(exist_ok=True, parents=True)
    nodes_path = nodes_dir / f"{hash(parser)}.pkl"

    if not nodes_path.exists():
        raise ValueError(
            f"Nodes not found at {nodes_path}. "
            "Run `docsrag parse-nodes` to generate them."
        )

    with open(nodes_path, "rb") as f:
        nodes = pickle.load(f)

    eval_data_builder = EvaluationDatasetBuilder.parse_obj(
        config["generate_evaluation_dataset"]
    )
    eval_data_dir = data_path_to_use / "eval_data"
    eval_data_path = eval_data_dir / f"{hash(eval_data_builder)}"
    if eval_data_path.exists():
        return

    eval_data_path.mkdir(exist_ok=True, parents=True)
    df = eval_data_builder.build(nodes)
    df.to_parquet(eval_data_path / "data.parquet")


@app.command()
def evaluate_embedding_vector_store(
    config_path: Optional[str] = None, data_path: Optional[str] = None
):
    """Evaluates the vector store."""
    from docsrag.embedding.evaluation import (
        VectorStoreEvaluator,
        load_evaluation_dataset,
    )
    from docsrag.embedding.index import VectorStoreIndexRay, VectorStoreSpec
    from docsrag.evaluation_dataset_generator import EvaluationDatasetBuilder
    from docsrag.utils import get_data_path, load_config

    config = load_config(config_path)
    data_path_to_use = Path(data_path) if data_path is not None else get_data_path()

    eval_data_builder = EvaluationDatasetBuilder.parse_obj(
        config["generate_evaluation_dataset"]
    )
    eval_data_dir = data_path_to_use / "eval_data"
    eval_data_path = eval_data_dir / f"{hash(eval_data_builder)}"
    if not eval_data_path.exists():
        raise ValueError(
            f"Evaluation dataset not found at {eval_data_path}. "
            "Run `docsrag generate-evaluation-dataset` to generate them."
        )

    vector_store_spec = VectorStoreSpec.parse_obj(config["build_vector_store"])
    vector_store_dir = data_path_to_use / "vector_store"
    vector_store_path = vector_store_dir / f"{hash(vector_store_spec)}"

    if not vector_store_path.exists():
        raise ValueError(
            f"Embedding vector store index not found at {vector_store_path}"
            "Run `docsrag build-embedding-vector-store-index` to generate them"
        )

    evaluator = VectorStoreEvaluator(
        vector_store_index=VectorStoreIndexRay.load(vector_store_path),
        top_ks=config["evaluate_embedding_vector_store"]["top_ks"],
    )
    eval_df = load_evaluation_dataset(
        evaluation_dataset_dir=(
            Path(data_path) if data_path is not None else get_data_path()
        ),
        evaluation_dataset_name=hash(eval_data_builder),
    )
    scores = evaluator.run(eval_df)
    print(scores)


if __name__ == "__main__":
    app()
