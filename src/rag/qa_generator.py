import os
import pickle
from typing import Literal
import joblib
import openai
from pydantic import BaseModel, Field

openai.api_key = os.environ["OPENAI_API_KEY"]


class QuestionAnswerGenerator(BaseModel):
    """Question Answer Generator."""

    prompt_template: str = """
    Provide questions and answers from the following context:

    {context}
    """
    model: Literal["gpt-3.5-turbo"]
    temperature: float = Field(ge=0.0, le=1.0)
    max_tokens: int = Field(gt=0, le=2048)
    top_p: float = Field(ge=0.0, le=1.0)
    frequency_penalty: float = Field(ge=0.0, le=1.0)
    presence_penalty: float = Field(ge=0.0, le=1.0)

    def generate_questions_answers(self, context: str):
        prompt = self.prompt_template.format(context=context)
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates questions and answers from a provided context",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            n=1,
            stream=False,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )
        try:
            return response.choices[0].message["content"]
        except Exception:
            return None

    def generate_questions_answers_from_contexts(self, contexts):
        for context in contexts:
            yield self.generate_questions_answers(context)

    def __hash__(self) -> int:
        hash_hex = joblib.hash(self.model_dump())
        return int(hash_hex, 16)


def build_row_data(node):
    row_data = node.metadata
    context = node.text
    response = qa_generator.generate_questions_answers(context)
    row_data["generated_question_answers"] = response
    row_data["text_hash_bin"] = int(row_data["text_hash"], 16) % num_text_hash_bins
    return row_data

if __name__ == "__main__":
    from pathlib import Path
    import yaml
    import pandas as pd
    from rag.generate_nodes import NodeParser

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

    qa_generator = QuestionAnswerGenerator.model_validate(config["qa_generator"])
    qa_dir = Path(__file__).parent / "qa"
    qa_dir.mkdir(exist_ok=True)

    num_text_hash_bins = 10
    # with joblib.Parallel(n_jobs=4, backend="threading") as parallel:
    #     data = parallel(
    #         joblib.delayed(build_row_data)(node)
    #         for node in nodes[:200]
    #         if node.metadata["file_path"].endswith((".rst", ".md"))
    #     )

    # df = pd.DataFrame(data)
    # df["qa_generator_hash"] = hash(qa_generator)
    # df["node_parser_hash"] = hash(parser)
    # df = df.astype({"qa_generator_hash": "category", "node_parser_hash": "category"})
    # print(df)
    # df.to_parquet(
    #     qa_dir,
    #     partition_cols=["qa_generator_hash", "node_parser_hash", "text_hash_bin"],
    # )
