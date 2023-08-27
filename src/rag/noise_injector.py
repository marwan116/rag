# inject out-of-context questions and answers
import pandas as pd

if __name__ == "__main__":
    df = pd.read_json("~/Downloads/triviaqa-unfiltered/unfiltered-web-train.json")
    trivia_questions = [row["Question"] for row in df["Data"][:100]]
    pd.Series(trivia_questions).to_frame("question").to_parquet(
        "src/rag/qa_trivia_questions.parquet"
    )
