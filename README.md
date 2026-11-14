# Failure Forensics Tool for AI Pipelines

An observability layer for multi-step AI pipelines. Every run produces a trace that records each step's input, output, LLM prompt, raw response, tokens, latency, errors and self-reported confidence, so when the final output is bad you can see which step went wrong.

**Status:** Phase 1 (multi-step pipeline) and Phase 2 (tracing layer) complete.

## The pipeline

A 4-step document pipeline, each step an isolated function with Pydantic-typed input and output:

1. **Intake**: normalizes raw text, markdown, or simulated PDF text (page markers, broken hyphenation). No LLM.
2. **Extraction**: LLM pulls out names, dates, amounts and key terms.
3. **Classification**: LLM labels the document as a contract, invoice, report or correspondence.
4. **Summarization**: LLM writes a structured summary whose schema depends on the document type.

`samples/` holds clean documents plus deliberately broken ones (prefixed `FAIL_`): a contract with no dates, an invoice mixing four currencies under one unlabeled total, a letter that is also a contract, and a blank OCR scan.

## Setup

Requires Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your OPENAI_API_KEY
```

## Run

```bash
python run.py --mock                               # offline fake LLM: free, tests the plumbing
python run.py                                      # all samples with OpenAI
python run.py samples/FAIL_contract_no_dates.txt   # one document
```

## How tracing works

Instrumenting a step is one line:

```python
@traced("extraction")
def extract(doc: IntakeResult, llm: BaseLLMClient) -> ExtractedEntities:
    return llm.parse(system=EXTRACTION_PROMPT, user=..., schema=ExtractedEntities)
```

The decorator opens a span, serializes the arguments as input, stores the return value as output, copies `confidence` and `warnings` from the output when present, and records any exception with its traceback. The LLM client attaches each call (prompt, raw response, tokens, latency) to whichever span is open, via `contextvars`, so steps never touch the tracer directly. Trace and span IDs use OpenTelemetry's format.

A trace ends with one of three statuses:

- **failure**: a step raised an error and the pipeline stopped.
- **degraded**: the pipeline finished, but a step reported confidence ≤ 2 or raised a warning.
- **success**: everything else.

## Storage

Each trace is written to `traces/<date>/<trace_id>.json` and indexed in `traces/index.db`:

```sql
-- Which step fails most?
SELECT failed_step, COUNT(*) FROM traces WHERE status = 'failure' GROUP BY failed_step;

-- Which step is least confident on average?
SELECT name, AVG(confidence) FROM spans WHERE confidence IS NOT NULL GROUP BY name;
```

## Layout

```
forensics/
  models.py     Pydantic models for every step's input and output
  tracing.py    Trace, Span, @traced decorator
  llm.py        OpenAI client + offline mock, both auto-recorded on spans
  pipeline.py   the four steps and run_pipeline()
  store.py      JSON files + SQLite index
run.py          CLI
samples/        test documents, clean and broken
```
