import time
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.graph import graph
from app.rag import DOCS_DIR, ingest
from app.sql_agent import get_schema
from app.tracing import recent_traces, save_trace

app = FastAPI(title="AskData")


class Ask(BaseModel):
    question: str
    history: list = []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: Ask):
    started = time.perf_counter()
    try:
        out = graph.invoke({"question": req.question, "history": req.history, "trace": []})
    except Exception:
        save_trace(req.question, {}, (time.perf_counter() - started) * 1000)
        raise
    save_trace(req.question, out, (time.perf_counter() - started) * 1000)
    return {k: out.get(k) for k in
            ("answer", "intent", "sql", "rows", "chunks", "grounded", "trace")}


@app.get("/traces")
def traces(limit: int = 50):
    return recent_traces(limit)


@app.post("/ingest")
async def ingest_upload(file: Annotated[UploadFile | None, File()] = None):
    if file is not None:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".pdf", ".md"}:
            raise HTTPException(status_code=400, detail="Only PDF and Markdown files are supported")
    else:
        suffix = ""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    destination = None
    if file is not None:
        destination = DOCS_DIR / Path(file.filename or "upload").name
        destination.write_bytes(await file.read())
    try:
        count = ingest(DOCS_DIR)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc
    return {"chunks_indexed": count, "filename": destination.name if destination else None}


@app.get("/schema")
def schema():
    try:
        return {"schema": get_schema()}
    except Exception as exc:
        return {"schema": "", "error": str(exc)}
