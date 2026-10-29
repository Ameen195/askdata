"""LangGraph pipeline: guardrail -> router -> (DOC | SQL | HYBRID) -> answer -> output check."""
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.llm import run_prompt
from app.rag import hybrid_retrieve
from app.sql_agent import get_samples, get_schema, run_sql, validate_sql

MAX_REPAIRS = 2


class State(TypedDict, total=False):
    question: str
    history: list
    safe: bool
    intent: str            # DOC | SQL | HYBRID
    rewritten: str
    chunks: list
    sql: str
    rows: list
    sql_error: str
    repairs: int
    answer: str
    grounded: bool
    trace: list


def traced(name):
    """Decorator: logs node name + latency into state['trace']."""
    def deco(fn):
        def wrapper(state: State) -> State:
            t0 = time.time()
            out = fn(state)
            step = {"node": name, "ms": round((time.time() - t0) * 1000)}
            out["trace"] = state.get("trace", []) + [step]
            return out
        return wrapper
    return deco


@traced("guardrail_in")
def guardrail_in(s: State) -> State:
    r = run_prompt("guardrail", json_mode=True, question=s["question"])
    return {"safe": bool(r.get("safe", True))}


@traced("router")
def router(s: State) -> State:
    r = run_prompt("router", json_mode=True, question=s["question"],
                   history=s.get("history", []), schema_summary=get_schema())
    return {"intent": r.get("intent", "DOC")}


@traced("rewrite")
def rewrite(s: State) -> State:
    q = run_prompt("rewrite", question=s["question"], history=s.get("history", []))
    return {"rewritten": q}


@traced("doc_retrieve")
def doc_retrieve(s: State) -> State:
    return {"chunks": hybrid_retrieve(s.get("rewritten", s["question"]), k=5)}


@traced("sql_generate")
def sql_generate(s: State) -> State:
    sql = run_prompt("sql_gen", question=s["question"],
                     schema=get_schema(), samples=get_samples())
    return {"sql": sql, "repairs": 0, "sql_error": ""}


@traced("sql_execute")
def sql_execute(s: State) -> State:
    try:
        safe_sql = validate_sql(s["sql"])
        return {"sql": safe_sql, "rows": run_sql(safe_sql), "sql_error": ""}
    except Exception as e:
        return {"sql_error": str(e), "rows": []}


@traced("sql_repair")
def sql_repair(s: State) -> State:
    sql = run_prompt("sql_repair", question=s["question"], schema=get_schema(),
                     sql=s["sql"], error=s["sql_error"])
    return {"sql": sql, "repairs": s.get("repairs", 0) + 1}


@traced("synthesize")
def synthesize(s: State) -> State:
    doc_ctx = "\n\n".join(f"[{c['source']}:{c['page']}] {c['text']}"
                          for c in s.get("chunks", []))
    ans = run_prompt("answer", question=s["question"], doc_context=doc_ctx or "None",
                     sql=s.get("sql", "None"), sql_result=str(s.get("rows", [])[:20]))
    return {"answer": ans}


@traced("guardrail_out")
def guardrail_out(s: State) -> State:
    ctx = str(s.get("chunks", [])) + str(s.get("rows", [])[:20])
    r = run_prompt("output_check", json_mode=True, context=ctx, answer=s["answer"])
    ok = r.get("grounded", True) and not r.get("pii", False)
    if not ok:
        return {"grounded": False,
                "answer": "I couldn't produce a reliably grounded answer from the available data."}
    return {"grounded": True}


def refuse(s: State) -> State:
    return {"answer": "Sorry, I can't help with that request."}


# ---- routing functions ----
def after_guardrail(s: State) -> str:
    return "router" if s.get("safe") else "refuse"


def after_router(s: State) -> str:
    return "sql_generate" if s["intent"] == "SQL" else "rewrite"


def after_doc(s: State) -> str:
    return "sql_generate" if s["intent"] == "HYBRID" else "synthesize"


def after_sql_execute(s: State) -> str:
    if s.get("sql_error") and s.get("repairs", 0) < MAX_REPAIRS:
        return "sql_repair"
    return "synthesize"


def build_graph():
    g = StateGraph(State)
    for name, fn in [("guardrail_in", guardrail_in), ("router", router),
                     ("rewrite", rewrite), ("doc_retrieve", doc_retrieve),
                     ("sql_generate", sql_generate), ("sql_execute", sql_execute),
                     ("sql_repair", sql_repair), ("synthesize", synthesize),
                     ("guardrail_out", guardrail_out), ("refuse", refuse)]:
        g.add_node(name, fn)

    g.add_edge(START, "guardrail_in")
    g.add_conditional_edges("guardrail_in", after_guardrail, ["router", "refuse"])
    g.add_conditional_edges("router", after_router, ["rewrite", "sql_generate"])
    g.add_edge("rewrite", "doc_retrieve")
    g.add_conditional_edges("doc_retrieve", after_doc, ["sql_generate", "synthesize"])
    g.add_edge("sql_generate", "sql_execute")
    g.add_conditional_edges("sql_execute", after_sql_execute, ["sql_repair", "synthesize"])
    g.add_edge("sql_repair", "sql_execute")
    g.add_edge("synthesize", "guardrail_out")
    g.add_edge("guardrail_out", END)
    g.add_edge("refuse", END)
    return g.compile()


graph = build_graph()
