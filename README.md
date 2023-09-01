# docsrag

## Motivation
Open source projects tend to have a rich documentation, but it is not always easy to find the information you need. This is especially true for large projects with many modules and classes.

`docsrag` enables you to deploy a service you can use to query questions about the documentation of a project. `docsrag` further provides an example integration with slack to create a slack bot that can answer questions about the documentation of a project.

More specifically, `docsrag` implements a retrieval-augmented generation (RAG) pipeline that can answer questions about the documentation of a project. 

## How to install

Using pip:

```bash
pip install docsrag
```

## User Guide

We recommend you start with our `demo/guide.ipynb` which explains the different components of a `docsrag` pipeline.


## Running docsrag

**Step 1**: Fetch the documents for the project you want to build a slack-bot for:

- Create a config yaml file for the fetcher to use, here is a sample config for the ray-project/ray repo:

```yaml
fetch_docs:
  owner: "ray-project"
  repo: "ray"
  version_tag: "releases/2.6.3"
  paths_to_include: ["doc/source/"]
  file_extensions_to_include: [".md", ".rst"]
  paths_to_exclude: [
    "doc/source/_ext/",
    "doc/source/_includes/",
    "doc/source/_static/",
    "doc/source/_templates/"
  ]
  filenames_to_exclude: []
```

- Set your github token to provide permissions to read from the repository that contains the documentation. For how to create a github token see the [documentation page here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)

```
export GITHUB_TOKEN={your_github_token}"
```

- Run the fetcher:

```bash
docsrag fetch-docs --config-path {path_to_config_file} --data-path {path_to_data_dir}
```

- Check to see that a docs directory has been created under {path_to_data_dir} with a pickle file storing your documents.


**Step 3** Parse nodes from the documents:


- Create a config yaml file for the node parser to use, here is a sample config:

```yaml
generate_nodes:
  inherit_metadata_from_doc: True
  construct_prev_next_relations: True
  text_chunker:
    chunk_size: 1024
    chunk_overlap: 20
    paragraph_separator: "\n\n\n"
    sentence_tokenizer:
      type: "tokenizers/punkt"
    secondary_chunking_regex: "[^,.;。]+[,.;。]?"
    tokenizer:
      encoding: "gpt2"
    word_seperator: " "
  metadata_pipeline:
    extractors:
      - file_path_extractor
      - text_hash_extractor
```

A node parser will take a document and chunk it, injecting metadata into each chunk.
The `chunk_size` specifies how many tokens each chunk should contain. Note that most LLM models 
have a maximum input length of 4096 tokens - so you should set the `chunk_size` to be less than this
to avoid truncating your prompt to the model.

- Run the node parser:
```bash
docsrag parse-ndoes --config-path {path_to_config_file} --data-path {path_to_data_dir}
```
- Check to see that a nodes directory has been created under {path_to_data_dir} with a pickle file storing your documents.

**Step 4** Build your embedding vector store index

- Create a config yaml file for the vector store builder to use, here is a sample config:

```yaml
build_vector_store:
  embedding_model_name: "BAAI/bge-small-en"
```

- Run the embedding vector store builder:

```bash
docsrag build-embedding-vector-store-index --config-path {path_to_config_file} --data-path {path_to_data_dir}
```

**Step 5** Generate a dataset of questions and answers from the node content


- Create a config yaml file for the dataset generator to use, here is a sample config:

```yaml
generate_evaluation_dataset:
  qa_generator_open_ai:
    model: "gpt-3.5-turbo"
    system_prompt: |
      You are a helpful assistant that generates questions and answers from a provided context.
      The context will be selected documents from a project's documentation.
      The questions you generate should be obvious on their own and should mimic what a developer might ask trying to work with the project, especially if they can't directly find the answer in the documentation.
      The answers should be factually correct, can be of a variable length and can contain code.
      If the provided context does not contain enough information to create a question and answer, you should respond with 'I can't generate a question and answer from this context'. 
      The following is an example of how the output should look:
      Q1: {question1}
      A1: {answer1}

      Q2: {question2}
      A2: {answer2}
    user_prompt_template: |
      Provide questions and answers from the following context:

      {context}
    max_tokens: 1024
    temperature: 1.0
    top_p: 0.85
    frequency_penalty: 0
    presence_penalty: 0
```

- Set up your OPENAI API key. For how to create an OPENAI API key see the [documentation page here](https://platform.openai.com/docs/developer-quickstart/your-api-keys)

```
export OPENAI_API_KEY={your_openai_api_key}"
```

- Run the evaluation dataset builder

```bash
docsrag generate-evaluation-dataset --config-path {path_to_config_file} --data-path {path_to_data_dir}
```

- Check to see that an `eval_data` directory has been created under {path_to_data_dir} with a parquet file storing your evaluation dataset.


**Step 6** Evaluate your Embedding vector Index

The evaluation will use our built embedding vector index to retrieve relevant nodes for a given quesiton in our evaluation dataset.
We then generate metrics by comparing the retrieved nodes with the source node that was used to generate the question.

- Run the evaluation 
```bash
docsrag evaluate-embedding-vector-store --config-path {path_to_config_file} --data-path {path_to_data_dir}
```

For instance a high recall@k score indicates that the embedding vector index is able to retrieve the correct node for a given question.


**Step 7** Having evaluated our embedding vector index, we can now move on to deploying an LLM model that is augmented with our embedding index with `ray.serve`


- Create a config yaml file for the LLM model we want to deploy

```yaml
deploy_model:
    model: gpt-3.5-turbo
    temperature: 0.1
    max_tokens: 1000
    max_retries: 10
    similarity_threshold: 0.8
```

- Set up your OPENAI API key. For how to create an OPENAI API key see the [documentation page here](https://platform.openai.com/docs/developer-quickstart/your-api-keys)

```
export OPENAI_API_KEY={your_openai_api_key}"
```

- Run the model deployer
```bash
docsrag deploy-model --config-path {path_to_config_file}
```

- This should now be running a `ray.serve` instance with a model that is augmented with our embedding vector index. You can test this by running the following command:

```bash
curl -X GET http://localhost:8000/raybot -d '{"question": "How do I use ray to train a model?"}'
```

**Step 8** 
TODO: Add instructions for deploying a slack bot

- Sign up for slack 
- Create a slack worskpace if you haven't already
- Add a slack bot
