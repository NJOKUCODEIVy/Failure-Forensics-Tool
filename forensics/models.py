"""Typed data structures passed between pipeline steps.

Models the LLM fills in (ExtractedEntities, Classification and the four
summaries) avoid defaults and numeric constraints so they stay valid for
OpenAI's strict structured-output mode. Each one ends with a self-reported
confidence (1-5) plus a one-line reason, which the tracer copies onto the span.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Union

from pydantic import BaseModel, Field

Confidence = Literal[1, 2, 3, 4, 5]
DocType = Literal["contract", "invoice", "report", "correspondence"]

CONFIDENCE_DESC = (
    "Your confidence that this output is complete and correct: "
    "1 = very unsure, 3 = some doubts, 5 = certain."
)
REASON_DESC = "One sentence explaining the rating. Name anything missing, ambiguous or inconsistent."


# ---------------------------------------------------------------- Step 1: intake

class DocumentFormat(str, Enum):
    TEXT = "text"
    MARKDOWN = "markdown"
    PDF_TEXT = "pdf_text"  # text extracted from a PDF: page markers, broken hyphenation


class RawDocument(BaseModel):
    doc_id: str
    format: DocumentFormat
    content: str
    source: str | None = None


class IntakeResult(BaseModel):
    doc_id: str
    format: DocumentFormat
    text: str
    word_count: int
    warnings: list[str] = Field(default_factory=list)


# ------------------------------------------------------------ Step 2: extraction

class DateMention(BaseModel):
    value: str = Field(description="ISO 8601 (YYYY-MM-DD) if the full date is stated, otherwise the text as written")
    context: str = Field(description="What the date refers to, e.g. 'effective date'")


class MoneyAmount(BaseModel):
    value: float
    currency: str | None = Field(description="ISO 4217 code as stated in the document, or null if no currency is given")
    context: str = Field(description="What the amount refers to, e.g. 'monthly fee'")


class ExtractedEntities(BaseModel):
    names: list[str] = Field(description="People and organizations named in the document")
    dates: list[DateMention]
    amounts: list[MoneyAmount]
    key_terms: list[str] = Field(description="Important terms, conditions or topics, a few words each")
    confidence: Confidence = Field(description=CONFIDENCE_DESC)
    confidence_reason: str = Field(description=REASON_DESC)


# -------------------------------------------------------- Step 3: classification

class Classification(BaseModel):
    doc_type: DocType
    runner_up: DocType = Field(description="The second most likely type")
    reasoning: str = Field(description="Which features of the document decided the type")
    confidence: Confidence = Field(description=CONFIDENCE_DESC)
    confidence_reason: str = Field(description=REASON_DESC)


# --------------------------------------------------------- Step 4: summarization

class ContractSummary(BaseModel):
    headline: str
    parties: list[str]
    effective_date: str | None
    end_date: str | None
    key_obligations: list[str]
    payment_terms: str | None
    termination_terms: str | None
    confidence: Confidence = Field(description=CONFIDENCE_DESC)
    confidence_reason: str = Field(description=REASON_DESC)


class LineItem(BaseModel):
    description: str
    amount: float
    currency: str | None


class InvoiceSummary(BaseModel):
    headline: str
    vendor: str | None
    customer: str | None
    invoice_number: str | None
    issue_date: str | None
    due_date: str | None
    line_items: list[LineItem]
    total: float | None
    currency: str | None
    confidence: Confidence = Field(description=CONFIDENCE_DESC)
    confidence_reason: str = Field(description=REASON_DESC)


class ReportSummary(BaseModel):
    headline: str
    title: str | None
    period: str | None
    key_findings: list[str]
    metrics: list[str]
    recommendations: list[str]
    confidence: Confidence = Field(description=CONFIDENCE_DESC)
    confidence_reason: str = Field(description=REASON_DESC)


class CorrespondenceSummary(BaseModel):
    headline: str
    sender: str | None
    recipient: str | None
    date: str | None
    purpose: str
    action_items: list[str]
    confidence: Confidence = Field(description=CONFIDENCE_DESC)
    confidence_reason: str = Field(description=REASON_DESC)


Summary = Union[ContractSummary, InvoiceSummary, ReportSummary, CorrespondenceSummary]

SUMMARY_SCHEMAS: dict[str, type[BaseModel]] = {
    "contract": ContractSummary,
    "invoice": InvoiceSummary,
    "report": ReportSummary,
    "correspondence": CorrespondenceSummary,
}


# ------------------------------------------------------------------ final output

class PipelineResult(BaseModel):
    doc_id: str
    doc_type: DocType
    entities: ExtractedEntities
    summary: Summary
