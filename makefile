# steps are fetch_docs, parse_nodes, build_evaluation_dataset
# build_embeddings, evaluate_embeddings, train_model, evaluate_model

fetch_docs:
	docsrag fetch_docs

parse_nodes:
	docsrag parse_nodes

build_evaluation_dataset:
	docsrag build_evaluation_dataset

build_embeddings:
	docsrag build_embeddings

evaluate_embeddings:
	docsrag evaluate_embeddings

train_model:
	docsrag train_model

evaluate_model:
	docsrag evaluate_model

