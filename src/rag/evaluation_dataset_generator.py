import math
import os
import re
from functools import cached_property, partial
from pathlib import Path
from typing import Any, Literal, Sequence, TYPE_CHECKING

import backoff
import joblib
import openai
import pandas as pd
from pydantic import BaseModel, Field, validator

if TYPE_CHECKING:
    from llama_index.schema import BaseNode


class QuestionAnswerGeneratorOpenAI(BaseModel):
    """Question Answer Generator."""

    model: Literal["gpt-3.5-turbo"]
    system_prompt: str
    user_prompt_template: str
    temperature: float = Field(ge=0.0, le=1.0)
    max_tokens: int = Field(gt=0, le=2048)
    top_p: float = Field(ge=0.0, le=1.0)
    frequency_penalty: float = Field(ge=0.0, le=1.0)
    presence_penalty: float = Field(ge=0.0, le=1.0)
    api_key: str = Field(exclude=True, repr=False, default=None)

    @validator("api_key", pre=True, always=True)
    def ensure_api_key_is_set(cls, key: Any) -> str:
        if key is None:
            key_to_use = os.environ["OPENAI_API_KEY"]
        else:
            if not isinstance(key, str):
                raise TypeError(f"api_key must be a string got {type(key)} instead")
            key_to_use = key
        openai.api_key = key_to_use
        return key_to_use

    def run(self, context: str, dry_run: bool = False) -> str:
        """Generate questions and answers."""
        prompt = self.user_prompt_template.format(context=context)
        params = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
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

        if dry_run:
            # logger.debug(f"would have run chat completion with {params}")
            return ""
        else:
            try:
                handler = backoff.on_exception(
                    exception=Exception,
                    wait_gen=partial(backoff.expo, base=3, factor=1, max_value=20),
                    jitter=partial(backoff.full_jitter, value=5),
                    max_tries=6,
                )
                response = handler(openai.ChatCompletion.create)(**params)
            except Exception as e:
                return f"Error: {str(e)}"
        try:
            return response.choices[0].message["content"]
        except Exception:
            return ""

    def __hash__(self) -> int:
        hash_hex = joblib.hash(self.dict())
        return int(hash_hex, 16)


class NoiseInjectorFromParquet(BaseModel):
    """Generates questions that don't have anything to do with our context."""

    class Config:
        arbitrary_types_allowed = True
        keep_untouched = (cached_property,)

    dataset_name: str
    dataset_dir: str = Field(default_factory=lambda: (Path(__file__).parent).resolve())

    def __hash__(self) -> int:
        hash_hex = joblib.hash(self.dict())
        return int(hash_hex, 16)

    @cached_property
    def noisy_questions_dataset(self) -> pd.DataFrame:
        path = Path(self.dataset_dir) / self.dataset_name
        return pd.read_parquet(path)

    def run(self, num_questions: int) -> pd.DataFrame:
        df = self.noisy_questions_dataset
        return df.sample(n=num_questions, replace=False)


class EvaluationDatasetBuilder(BaseModel):
    qa_generator_open_ai: QuestionAnswerGeneratorOpenAI
    noise_injector_from_parquet: NoiseInjectorFromParquet
    noise_proportion: float = 0.1
    max_concurrent_requests: int = Field(exclude=True, repr=False, default=10)

    def __hash__(self) -> int:
        hash_hex = joblib.hash(self.dict())
        return int(hash_hex, 16)

    def _build_dataset(
        self, nodes: Sequence["BaseNode"], dry_run: bool = False
    ) -> list[dict[str, Any]]:
        data = []

        for node in nodes:
            response = self.qa_generator_open_ai.run(context=node.text, dry_run=dry_run)
            if response.strip().startswith("Q"):
                lines = response.split("\n")
                questions = [
                    re.sub("^Q.+?:", "", question).strip()
                    for question in lines
                    if question.startswith("Q")
                ]
                answers = [
                    re.sub("^A.+?:", "", answer).strip()
                    for answer in lines
                    if answer.startswith("A")
                ]
                data.extend(
                    [
                        {
                            **node.metadata,
                            "qa_generator": hash(self.qa_generator_open_ai),
                            "question": question,
                            "answer": answer,
                            "error": "",
                        }
                        for question, answer in zip(questions, answers)
                    ]
                )
            else:
                data.append(
                    {
                        **node.metadata,
                        "qa_generator": hash(self.qa_generator_open_ai),
                        "question": "",
                        "answer": "",
                        "error": response,
                    }
                )
        return data

    def _inject_noise(self, df_meaningful_questions_answers):
        num_non_meaningful_questions = math.ceil(
            self.noise_proportion * df_meaningful_questions_answers.shape[0]
        )

        return pd.concat(
            [
                df_meaningful_questions_answers,
                pd.DataFrame(
                    self.noise_injector_from_parquet.run(num_non_meaningful_questions),
                    columns=["question"],
                ),
            ],
            axis=0,
        )

    def build(self, nodes):
        import ray

        ray.init(ignore_reinit_error=True)

        df_meaningful_questions_answers = pd.DataFrame(
            [
                data_partition
                for data_partition in (
                    ray.data.from_items(nodes)
                    .map_batches(self._build_dataset, batch_size=100)
                    .iter_rows()
                )
            ]
        )

        df_meaningful_questions_answers.to_parquet(
            Path(__file__).parent / "meaningful_questions.parquet"
        )

        df = self._inject_noise(df_meaningful_questions_answers)

        return df
