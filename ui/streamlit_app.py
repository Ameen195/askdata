"""Streamlit UI for the AskData FastAPI service."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("ASKDATA_API_URL", "http://localhost:8000").rstrip("/")


def api_get(path: str) -> Any:
    response = requests.get(f"{API_URL}{path}", timeout=60)
    response.raise_for_status()
    return response.json()


def api_post(path: str, **kwargs: Any) -> Any:
    response = requests.post(f"{API_URL}{path}", timeout=600, **kwargs)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="AskData", page_icon="📊", layout="wide")
st.title("AskData")
chat_tab, data_tab, eval_tab, trace_tab = st.tabs(
    ["Chat", "Data & Docs", "Evals", "Traces"]
)

with chat_tab:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("response"):
                result = message["response"]
                st.caption(
                    f"Intent: {result.get('intent', 'unknown')} · "
                    + ("Grounded" if result.get("grounded") else "Not grounded")
                )
                if result.get("sql"):
                    with st.expander("Generated SQL"):
                        st.code(result["sql"], language="sql")
                rows = result.get("rows") or []
                if rows:
                    frame = pd.DataFrame(rows)
                    st.dataframe(frame, use_container_width=True)
                    numeric = frame.select_dtypes(include="number").columns.tolist()
                    categories = [c for c in frame.columns if c not in numeric]
                    if numeric and categories:
                        st.bar_chart(frame.set_index(categories[0])[numeric[0]])
                chunks = result.get("chunks") or []
                if chunks:
                    st.markdown("**Sources**")
                    for chunk in chunks:
                        st.markdown(
                            f"- [{chunk['source']}:{chunk['page']}] {chunk['text'][:240]}"
                        )
    prompt = st.chat_input("Ask about your documents or Olist data")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]
        with st.chat_message("assistant"), st.spinner("Working..."):
            try:
                result = api_post("/ask", json={"question": prompt, "history": history})
                st.markdown(result.get("answer", ""))
                st.caption(
                    f"Intent: {result.get('intent', 'unknown')} · "
                    + ("Grounded" if result.get("grounded") else "Not grounded")
                )
                if result.get("sql"):
                    with st.expander("Generated SQL"):
                        st.code(result["sql"], language="sql")
                rows = result.get("rows") or []
                if rows:
                    frame = pd.DataFrame(rows)
                    st.dataframe(frame, use_container_width=True)
                    numeric = frame.select_dtypes(include="number").columns.tolist()
                    categories = [c for c in frame.columns if c not in numeric]
                    if numeric and categories:
                        st.bar_chart(frame.set_index(categories[0])[numeric[0]])
                for chunk in result.get("chunks") or []:
                    st.markdown(
                        f"- [{chunk['source']}:{chunk['page']}] {chunk['text'][:240]}"
                    )
            except requests.RequestException as exc:
                result = {"answer": f"Backend request failed: {exc}"}
                st.error(result["answer"])
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result.get("answer", ""),
                    "response": result,
                }
            )

with data_tab:
    st.subheader("Upload policy documents")
    uploaded = st.file_uploader("PDF or Markdown", type=["pdf", "md"])
    if uploaded and st.button("Upload and index"):
        try:
            result = api_post(
                "/ingest", files={"file": (uploaded.name, uploaded.getvalue())}
            )
            st.success(
                f"Indexed {result['chunks_indexed']} chunks from {uploaded.name}."
            )
        except requests.RequestException as exc:
            st.error(f"Indexing failed: {exc}")
    if st.button("Re-index all documents"):
        try:
            result = api_post("/ingest")
            st.success(f"Re-indexed {result['chunks_indexed']} chunks.")
        except requests.RequestException as exc:
            st.error(f"Re-index failed: {exc}")
    st.subheader("DuckDB schema")
    try:
        schema_data = api_get("/schema")
        if schema_data.get("error"):
            st.info(f"Database unavailable: {schema_data['error']}")
        else:
            st.code(schema_data.get("schema") or "No tables loaded.")
    except requests.RequestException as exc:
        st.error(f"Could not load schema: {exc}")

with eval_tab:
    st.subheader("Evaluation results")
    result_path = os.getenv("ASKDATA_RESULTS_PATH", "evals/results.json")
    try:
        with open(result_path, encoding="utf-8") as file:
            results = __import__("json").load(file)
        columns = st.columns(4)
        for col, (label, value) in zip(
            columns,
            [
                (
                    "SQL execution accuracy",
                    results.get("sql_execution_accuracy", {}).get("accuracy"),
                ),
                ("Faithfulness", results.get("faithfulness")),
                ("Relevance", results.get("relevance")),
                ("Average latency (ms)", results.get("average_latency_ms")),
            ],
        ):
            col.metric(label, "N/A" if value is None else f"{value:.3f}")
        retrieval = results.get("retrieval", {})
        if retrieval:
            chart = pd.DataFrame(retrieval).T[["hit_at_5", "mrr"]]
            st.bar_chart(chart)
    except FileNotFoundError:
        st.info("Run `python evals/run_evals.py` to create evals/results.json.")
    except (ValueError, KeyError) as exc:
        st.error(f"Could not read evaluation results: {exc}")

with trace_tab:
    st.subheader("Recent requests")
    try:
        traces = api_get("/traces?limit=50")
        if traces:
            frame = pd.DataFrame(
                [
                    {k: v for k, v in row.items() if k != "node_latency_ms"}
                    for row in traces
                ]
            )
            st.dataframe(frame, use_container_width=True)
            selected_id = st.selectbox("Query to inspect", [r["id"] for r in traces])
            selected = next(r for r in traces if r["id"] == selected_id)
            st.write(selected["question"])
            timings = selected.get("node_latency_ms", [])
            if timings:
                st.bar_chart(pd.DataFrame(timings).set_index("node")["ms"])
        else:
            st.info("No requests have been recorded yet.")
    except requests.RequestException as exc:
        st.error(f"Could not load traces: {exc}")
