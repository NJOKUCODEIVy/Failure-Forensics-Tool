"""The 4-step document pipeline: intake -> extraction -> classification -> summarization.

Each step is a plain function with typed inputs and a typed output. The
@traced decorator is the only tracing code a step needs.
"""

from __future__ import annotations

import re

from .llm import BaseLLMClient
from .models import (
    SUMMARY_SCHEMAS,
    Classification,
    DocumentFormat,
    ExtractedEntities,
    IntakeResult,
    PipelineResult,
    RawDocument,
    Summary,
)
from .store import TraceStore
from .tracing import Trace, start_trace, traced, to_jsonable

MIN_WORDS = 40
MIN_READABLE_CHARS = 20

PAGE_MARKER = re.compile(r"\bPage\s+\d+\s+of\s+\d+\b", re.IGNORECASE)
HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
MD_SYNTAX = re.compile(r"(\*\*|__|^#{1,6}\s+)", re.MULTILINE)


# ------------------------------------------------------------- Step 1: intake

@traced("intake")
def intake(doc: RawDocument) -> IntakeResult:
    """Normalize raw text. No LLM: cheap, deterministic, easy to trust."""
    text = doc.content.replace("\r\n", "\n")
    warnings: list[str] = []

    if doc.format == DocumentFormat.PDF_TEXT:
        text = PAGE_MARKER.sub("", text)
        text = HYPHEN_BREAK.sub(r"\1\2", text)  # "agree-\nment" -> "agreement"
    elif doc.format == DocumentFormat.MARKDOWN:
        text = MD_SYNTAX.sub("", text)

    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if len(re.findall(r"\w", text)) < MIN_READABLE_CHARS:
        raise ValueError("Document has no readable text (empty file or failed OCR)")

    word_count = len(text.split())
    if word_count < MIN_WORDS:
        warnings.append(f"Very short document ({word_count} words)")

    return IntakeResult(doc_id=doc.doc_id, format=doc.format, text=text, word_count=word_count, warnings=warnings)


# --------------------------------------------------------- Step 2: extraction

EXTRACTION_PROMPT = """You extract structured entities from business documents.
Extract only what the document explicitly states. Never infer or invent values.
- Dates: use YYYY-MM-DD when the full date is stated; otherwise copy the text as written.
- Amounts: give the numeric value and the currency exactly as stated. If no currency is stated for an amount, use null.
Then rate your confidence that the extraction is complete and correct."""


@traced("extraction")
def extract(doc: IntakeResult, llm: BaseLLMClient) -> ExtractedEntities:
    return llm.parse(system=EXTRACTION_PROMPT, user=_doc_block(doc), schema=ExtractedEntities)


# ----------------------------------------------------- Step 3: classification

CLASSIFICATION_PROMPT = """You classify business documents as exactly one of:
- contract: an agreement that creates obligations between parties
- invoice: a request for payment for goods or services
- report: an analysis or summary of results, findings or metrics
- correspondence: a letter, email or memo between people
Also name the runner-up type. Then rate your confidence in the classification."""


@traced("classification")
def classify(doc: IntakeResult, entities: ExtractedEntities, llm: BaseLLMClient) -> Classification:
    user = f"{_doc_block(doc)}\n\nEntities already extracted:\n{entities.model_dump_json(exclude={'confidence', 'confidence_reason'})}"
    return llm.parse(system=CLASSIFICATION_PROMPT, user=user, schema=Classification)


# ------------------------------------------------------ Step 4: summarization

SUMMARY_PROMPT = """You write a structured summary of a {doc_type}.
Fill every field from the document itself. Use null when the document does not state a value; never guess.
Then rate your confidence that the summary is complete and correct."""


@traced("summarization")
def summarize(doc: IntakeResult, entities: ExtractedEntities, classification: Classification, llm: BaseLLMClient) -> Summary:
    schema = SUMMARY_SCHEMAS[classification.doc_type]
    user = f"{_doc_block(doc)}\n\nEntities already extracted:\n{entities.model_dump_json(exclude={'confidence', 'confidence_reason'})}"
    return llm.parse(system=SUMMARY_PROMPT.format(doc_type=classification.doc_type), user=user, schema=schema)


# ------------------------------------------------------------------ pipeline

def run_pipeline(raw: RawDocument, llm: BaseLLMClient, store: TraceStore | None = None) -> Trace:
    """Run all four steps inside one trace. Never raises: errors land in the trace."""
    with start_trace("document_pipeline", trace_input=raw, metadata={"model": llm.model, "provider": llm.provider}) as trace:
        try:
            doc = intake(raw)
            entities = extract(doc, llm=llm)
            classification = classify(doc, entities, llm=llm)
            summary = summarize(doc, entities, classification, llm=llm)
            trace.final_output = to_jsonable(
                PipelineResult(doc_id=raw.doc_id, doc_type=classification.doc_type, entities=entities, summary=summary)
            )
        except Exception as exc:  # already recorded on the failing span
            trace.error = f"{type(exc).__name__}: {exc}"

    if store is not None:
        store.save(trace)
    return trace


def _doc_block(doc: IntakeResult) -> str:
    return f"Document (format: {doc.format.value}):\n<<<\n{doc.text}\n>>>"
