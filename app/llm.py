"""LLM wrapper: Groq (free tier) by default, Ollama as local fallback."""
import json
import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
load_dotenv(PROMPT_DIR.parent / ".env")


def load_prompt(name: str) -> dict:
    return yaml.safe_load((PROMPT_DIR / f"{name}.yaml").read_text())


def render(template: str, **vars) -> str:
    for k, v in vars.items():
        template = template.replace("{{" + k + "}}", str(v))
    return template


def chat(system: str, user: str, json_mode: bool = False) -> str:
    provider = os.getenv("LLM_PROVIDER", "groq")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    if provider == "groq":
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        kwargs = dict(model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
                      messages=messages, temperature=0)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return client.chat.completions.create(**kwargs).choices[0].message.content
    import ollama
    kwargs = dict(model=os.getenv("LLM_MODEL", "llama3.2"), messages=messages)
    if json_mode:
        kwargs["format"] = "json"
    return ollama.chat(**kwargs)["message"]["content"]


def run_prompt(name: str, json_mode: bool = False, **vars):
    p = load_prompt(name)
    out = chat(render(p["system"], **vars), render(p["user"], **vars), json_mode)
    if json_mode:
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", out, re.DOTALL)
            return json.loads(m.group(0)) if m else {}
    return re.sub(r"^```(?:sql)?|```$", "", out.strip(), flags=re.MULTILINE).strip()
