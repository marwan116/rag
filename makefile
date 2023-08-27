# steps are fetch_docs, parse_nodes, build_evaluation_dataset
# build_embeddings, evaluate_embeddings, train_model, evaluate_model

fetch_docs:
	rag fetch_docs

parse_nodes:
	rag parse_nodes

build_evaluation_dataset:
	rag build_evaluation_dataset

build_embeddings:
	rag build_embeddings

evaluate_embeddings:
	rag evaluate_embeddings

train_model:
	rag train_model

evaluate_model:
	rag evaluate_model

