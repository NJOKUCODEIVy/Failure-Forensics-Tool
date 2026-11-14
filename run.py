"""Run documents through the traced pipeline.

    python run.py --mock                          # free: fake LLM, tests the plumbing
    python run.py                                 # every file in samples/ with OpenAI
    python run.py samples/FAIL_contract_no_dates.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from forensics.llm import BaseLLMClient, MockLLMClient, OpenAILLMClient
from forensics.models import DocumentFormat, RawDocument
from forensics.pipeline import run_pipeline
from forensics.store import TraceStore
from forensics.tracing import Trace

SAMPLES = Path(__file__).parent / "samples"
STATUS_ICON = {"success": "✅", "degraded": "⚠️ ", "failure": "❌"}


def load_document(path: Path) -> RawDocument:
    name = path.name.lower()
    if name.endswith(".pdf.txt"):
        fmt = DocumentFormat.PDF_TEXT
    elif name.endswith(".md"):
        fmt = DocumentFormat.MARKDOWN
    else:
        fmt = DocumentFormat.TEXT
    doc_id = path.name.split(".")[0]
    return RawDocument(doc_id=doc_id, format=fmt, content=path.read_text(encoding="utf-8"), source=str(path))


def print_trace(trace: Trace, label: str) -> None:
    score = f"{trace.final_score:.0f}" if trace.final_score is not None else "-"
    print(f"\n{STATUS_ICON[trace.status]} {label}  →  {trace.status.upper()}"
          f"   score {score} · {trace.total_tokens:,} tokens · {trace.latency_ms:,.0f} ms")
    for s in trace.spans:
        mark = "✗" if s.status == "error" else "✓"
        conf = f"conf {s.confidence}" if s.confidence is not None else "      "
        if s.error:
            note = f"{s.error.type}: {s.error.message}"
        elif s.warnings:
            note = "; ".join(s.warnings)
        else:
            note = s.confidence_reason or ""
        if len(note) > 80:
            note = note[:77] + "..."
        print(f"   {mark} {s.name:<15}{s.latency_ms:>8,.0f} ms   {conf}   {note}")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run documents through the traced pipeline.")
    parser.add_argument("paths", nargs="*", type=Path, help="Documents to run (default: everything in samples/)")
    parser.add_argument("--mock", action="store_true", help="Use the offline mock LLM (no API calls, no cost)")
    parser.add_argument("--model", help="OpenAI model (overrides OPENAI_MODEL)")
    parser.add_argument("--traces-dir", default="traces", help="Where traces are stored (default: traces/)")
    args = parser.parse_args()

    paths = args.paths or sorted(p for p in SAMPLES.iterdir() if p.is_file() and not p.name.startswith("."))
    if not paths:
        print("No documents found.", file=sys.stderr)
        return 1

    if args.mock:
        llm: BaseLLMClient = MockLLMClient()
    else:
        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY is not set. Add it to .env, or run with --mock.", file=sys.stderr)
            return 1
        llm = OpenAILLMClient(model=args.model)

    store = TraceStore(args.traces_dir)
    print(f"Running {len(paths)} document(s) with {llm.provider}:{llm.model}")

    traces = []
    for path in paths:
        trace = run_pipeline(load_document(path), llm, store)
        traces.append(trace)
        print_trace(trace, path.name)

    counts = {status: sum(t.status == status for t in traces) for status in ("success", "degraded", "failure")}
    print(f"\n{counts['success']} success · {counts['degraded']} degraded · {counts['failure']} failure"
          f"   →  traces saved to {store.root}/ (index: {store.db_path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
