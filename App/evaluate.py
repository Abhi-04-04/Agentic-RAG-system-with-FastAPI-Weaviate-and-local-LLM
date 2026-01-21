import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    answer_relevancy,
    faithfulness,
)
import requests

API_URL = "http://127.0.0.1:8000/query"

def call_api(q):
    r = requests.post(API_URL, params={"question": q})
    data = r.json()
    return {
        "answer": data["answer"],
        "contexts": data["sub_questions"]
    }

def run_eval():
    with open("eval_data.json") as f:
        raw = json.load(f)

    rows = []
    for r in raw:
        res = call_api(r["question"])
        rows.append({
            "question": r["question"],
            "answer": res["answer"],
            "contexts": res["contexts"],
            "ground_truth": r["ground_truth"]
        })

    ds = Dataset.from_list(rows)

    result = evaluate(
        ds,
        metrics=[
            context_precision,
            context_recall,
            answer_relevancy,
            faithfulness,
        ],
    )

    print(result)

if __name__ == "__main__":
    run_eval()
