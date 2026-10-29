"""Run retrieval, SQL-result, and LLM-judge evaluations."""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.llm import run_prompt
from app.rag import hybrid_retrieve
from app.sql_agent import get_samples, get_schema, run_sql, validate_sql

MODES = ("bm25", "vector", "hybrid", "hybrid_rerank")
GOLDEN_PATH = ROOT / "evals" / "golden_set.json"
RESULT_PATH = ROOT / "evals" / "results.json"


def _same_result(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    def normalize(rows: list[dict[str, Any]]) -> list[tuple[tuple[str, Any], ...]]:
        normalized = []
        for row in rows:
            values = tuple(sorted((key, round(value, 6) if isinstance(value, float) else value)
                                  for key, value in row.items()))
            normalized.append(repr(values))
        return sorted(normalized)
    return normalize(left) == normalize(right)


def _judge(item: dict[str, Any], context: str, answer: str) -> dict[str, Any]:
    return run_prompt("judge", json_mode=True, question=item["question"], context=context,
                      answer=answer, reference=item.get("reference_answer", ""))


def run_evals() -> dict[str, Any]:
    items = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    retrieval_items = [x for x in items if x["type"] in {"doc", "hybrid"}]
    retrieval_metrics = {}
    mode_chunks: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for mode in MODES:
        hits, reciprocal_ranks = [], []
        mode_chunks[mode] = {}
        for item in retrieval_items:
            results = hybrid_retrieve(item["question"], k=5, mode=mode)
            mode_chunks[mode][item["id"]] = results
            relevant = item["relevant_source"].lower()
            rank = next((i for i, c in enumerate(results, 1)
                         if Path(c["source"]).name.lower() == relevant), None)
            hits.append(int(rank is not None))
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        retrieval_metrics[mode] = {
            "hit_at_5": statistics.mean(hits) if hits else 0.0,
            "mrr": statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        }

    sql_metrics = {"accuracy": None, "correct": 0, "total": 0, "unavailable": None}
    sql_items = [x for x in items if x["type"] in {"sql", "hybrid"}]
    correct = 0
    try:
        schema = get_schema()
        samples = get_samples()
        for item in sql_items:
            generated = run_prompt("sql_gen", question=item["question"], schema=schema, samples=samples)
            generated_result = run_sql(validate_sql(generated))
            reference_result = run_sql(validate_sql(item["reference_sql"]))
            correct += int(_same_result(generated_result, reference_result))
        sql_metrics.update({"accuracy": correct / len(sql_items), "correct": correct,
                            "total": len(sql_items)})
    except Exception as exc:
        sql_metrics.update({"accuracy": None, "correct": correct, "total": len(sql_items),
                            "unavailable": str(exc)})
        schema, samples = "", ""

    faithfulness, relevance, latencies = [], [], []
    judged = []
    for item in items:
        started = time.perf_counter()
        chunks = mode_chunks.get("hybrid_rerank", {}).get(item["id"], [])
        doc_context = "\n\n".join(c["text"] for c in chunks)
        sql = "None"
        sql_result: list[dict[str, Any]] = []
        if item["type"] in {"sql", "hybrid"} and sql_metrics["unavailable"] is None:
            sql = run_prompt("sql_gen", question=item["question"], schema=schema, samples=samples)
            try:
                sql_result = run_sql(validate_sql(sql))[:20]
            except Exception:
                sql_result = []
        answer = run_prompt("answer", question=item["question"],
                            doc_context=doc_context or "None", sql=sql,
                            sql_result=str(sql_result))
        context = f"Documents:\n{doc_context or 'None'}\nSQL: {sql}\nSQL result: {sql_result}"
        verdict = _judge(item, context, answer)
        faithfulness.append(float(verdict.get("faithfulness", 0)))
        relevance.append(float(verdict.get("relevance", 0)))
        latencies.append((time.perf_counter() - started) * 1000)
        judged.append({"id": item["id"], "faithfulness": verdict.get("faithfulness"),
                       "relevance": verdict.get("relevance"), "comment": verdict.get("comment", "")})

    result = {
        "retrieval": retrieval_metrics,
        "sql_execution_accuracy": sql_metrics,
        "faithfulness": statistics.mean(faithfulness) if faithfulness else None,
        "relevance": statistics.mean(relevance) if relevance else None,
        "average_latency_ms": statistics.mean(latencies) if latencies else None,
        "judgments": judged,
        "item_count": len(items),
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("| Retrieval mode | Hit@5 | MRR |\n|---|---:|---:|")
    for mode, metrics in retrieval_metrics.items():
        print(f"| {mode} | {metrics['hit_at_5']:.3f} | {metrics['mrr']:.3f} |")
    print(f"\nSQL execution accuracy: {sql_metrics['accuracy'] if sql_metrics['accuracy'] is not None else 'unavailable'}")
    print(f"Faithfulness: {result['faithfulness']}; relevance: {result['relevance']}; average latency: {result['average_latency_ms']:.1f} ms")
    print(f"Saved {RESULT_PATH}")
    return result


if __name__ == "__main__":
    run_evals()
