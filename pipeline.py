"""pipeline.py

Purpose
-------
Orchestrates the existing modules (pdf_input, retrieval, decision,
document) into one run: extract the job PDF, rank the profile against
it, make exactly one drafting API request, save the validated draft,
and render + page-count it. No new framework - this is glue code that
reuses what already exists.

One note on reuse: decision.draft_resume_content() recomputes evidence
(TF-IDF) internally, but this pipeline needs that evidence list
separately too, to display ranked titles/scores in the trace - and
must calculate TF-IDF only once per run. So this module calls
decision.draft_resume_content_with_evidence() directly instead, passing
the evidence list it already computed. There is only one drafting
implementation (in decision.py); this file does not duplicate it.

Where this fits in the application
-----------------------------------
run_pipeline() is what app.py's Generate button calls. It is a
generator: each step both prints to the terminal immediately and yields
an updated (trace_text, download_path_or_None) pair, so the same
messages appear live in the terminal and incrementally in the browser.
Only one run is allowed at a time (a module-level lock); overlapping
clicks are rejected with a clear message rather than queued or ignored.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, List, Optional, Tuple

from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from decision import (
    DraftingError,
    TokenUsage,
    _resolve_api_key,
    collect_full_evidence,
    decision_to_dict,
    draft_resume_content_with_evidence,
)
from document import DocumentGenerationError, render_and_verify
from pdf_input import extract_pdf_text
from retrieval import PROFILE_PATH, format_shared_terms, load_profile

ENV_PATH = Path(__file__).resolve().parent / ".env"
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"

# Verified against OpenAI's published standard (non-"Ultrafast",
# short-context) API pricing as of 2026-10-12. Only models listed here
# get a numeric estimate; anything else reports "estimate unavailable" -
# never an invented price.
MODEL_PRICING_PER_MILLION_TOKENS_USD = {
    "gpt-6.1-sol": {"input": 2.00, "output": 10.00},
}

_RUN_LOCK = threading.Lock()


_DATED_SNAPSHOT_SUFFIX_RE = re.compile(r"^-\d{4}-\d{2}-\d{2}$")


def _find_verified_pricing(model: str) -> Tuple[Optional[dict], Optional[str]]:
    """Exact match, or a dated-snapshot match (e.g. "gpt-6.1-sol-2026-10-01"
    against verified base name "gpt-6.1-sol") - OpenAI's own snapshot
    naming convention, not a guess at an unrelated model's price. The
    suffix after the base name must be exactly "-YYYY-MM-DD"; a variant
    name like "gpt-6.1-sol-mini" or "gpt-6.1-sol-turbo" does NOT match,
    since `startswith` alone would wrongly treat any such variant as
    the same priced model.
    Returns (pricing, matched_base_name) or (None, None)."""
    if model in MODEL_PRICING_PER_MILLION_TOKENS_USD:
        return MODEL_PRICING_PER_MILLION_TOKENS_USD[model], model
    for base_name, pricing in MODEL_PRICING_PER_MILLION_TOKENS_USD.items():
        if model.startswith(base_name) and _DATED_SNAPSHOT_SUFFIX_RE.match(model[len(base_name):]):
            return pricing, base_name
    return None, None


def estimate_cost_usd(model: Optional[str], usage: Optional[TokenUsage]) -> str:
    """A clearly-labeled estimate string, or an explicit
    "unavailable" note - never a fabricated number for an unverified
    model, and never silent about the assumptions behind the number."""
    if model is None or usage is None:
        return "Cost estimate unavailable (model or token usage not reported by the API)."
    pricing, matched_base_name = _find_verified_pricing(model)
    if pricing is None:
        return f"Cost estimate unavailable: no verified pricing on file for model {model!r}."
    amount = (usage.input_tokens * pricing["input"] + usage.output_tokens * pricing["output"]) / 1_000_000
    snapshot_note = f" (model reported: {model})" if matched_base_name != model else ""
    return (
        f"ESTIMATE: ${amount:.4f} using standard, short-context published rates for "
        f"{matched_base_name}{snapshot_note} (${pricing['input']:.2f}/M input + ${pricing['output']:.2f}/M "
        "output tokens, verified 2026-10-12). Assumes no cached-input discount, no long-context "
        "(>272k token) surcharge, and no Ultrafast-mode pricing - actual billing may differ."
    )


def run_pipeline(pdf_path: Optional[str], model: str) -> Generator[Tuple[str, Optional[str]], None, None]:
    """One run: extract -> rank -> draft (1 request) -> save -> render.

    Yields (trace_text, download_path) after every step - download_path
    is None until (and unless) rendering verifies exactly two pages.
    Rejects a second concurrent call outright rather than queuing it.
    """
    trace: List[str] = []

    def emit(message: str) -> str:
        print(message)
        trace.append(message)
        return "\n".join(trace)

    if not _RUN_LOCK.acquire(blocking=False):
        yield emit("A run is already in progress - please wait for it to finish before starting another."), None
        return

    try:
        yield emit("Starting new run (any previous download is now cleared)."), None

        if not pdf_path:
            yield emit("No job-description PDF was provided."), None
            return
        if not model or not model.strip():
            yield emit("No model name was provided."), None
            return

        yield emit(f"Extracting text from: {pdf_path}"), None
        try:
            extraction = extract_pdf_text(Path(pdf_path))
        except FileNotFoundError as exc:
            yield emit(f"Could not read the uploaded file: {exc}"), None
            return
        except ValueError as exc:
            yield emit(f"Could not extract text from the PDF: {exc}"), None
            return
        for warning in extraction.warnings:
            yield emit(f"WARNING: {warning}"), None
        if not extraction.full_text.strip():
            yield emit("No extractable text found in the PDF; stopping before any API request."), None
            return

        yield emit("Loading the profile and computing TF-IDF relevance (once for this run)..."), None
        _metadata, contact, chunks = load_profile(PROFILE_PATH)
        evidence = collect_full_evidence(extraction.full_text, chunks)

        yield emit("Ranked profile evidence (local lexical similarity only - not a relevance filter):"), None
        for item in sorted(evidence, key=lambda e: e.score, reverse=True):
            ranked_terms = sorted(item.shared_terms.items(), key=lambda kv: (-kv[1], kv[0]))
            yield emit(
                f"  [{item.chunk_id}] {item.title} - score={item.score:.3f} - "
                f"shared terms: {format_shared_terms(ranked_terms)}"
            ), None

        api_key = _resolve_api_key(ENV_PATH)
        if not api_key or not api_key.strip():
            yield emit("OPENAI_API_KEY is not set - add it to .env or the environment, then retry."), None
            return

        try:
            client = OpenAI(max_retries=0)
        except OpenAIError:
            yield emit("Could not set up the OpenAI client - check that OPENAI_API_KEY is set."), None
            return

        yield emit(f"Making 1 drafting API request to model '{model}' (retries disabled)..."), None
        try:
            outcome = draft_resume_content_with_evidence(evidence, extraction.full_text, model, client)
        except DraftingError as exc:
            yield emit(f"Drafting failed: {exc}"), None
            yield emit("Request count: 1."), None
            yield emit(f"  model reported by the API: {exc.model or '(not reported)'}"), None
            if exc.usage is not None:
                yield emit(
                    f"  token usage - input: {exc.usage.input_tokens}, output: {exc.usage.output_tokens}, "
                    f"total: {exc.usage.total_tokens}"
                ), None
            else:
                yield emit("  token usage: not reported by the API response."), None
            yield emit(f"  {estimate_cost_usd(exc.model, exc.usage)}"), None
            return
        except AuthenticationError:
            yield emit("Authentication failed - check that OPENAI_API_KEY is set and valid."), None
            return
        except RateLimitError:
            yield emit("Rate limit or quota exceeded - wait and retry later, or check your plan/billing."), None
            return
        except APIConnectionError:
            yield emit("Could not connect to the OpenAI API - check your network connection and try again."), None
            return
        except APIStatusError as exc:
            yield emit(f"The OpenAI API returned an error (HTTP {exc.status_code})."), None
            return
        except OpenAIError:
            yield emit("An OpenAI SDK error occurred while requesting a draft."), None
            return

        decision_obj = outcome.decision
        returned_model = outcome.model
        usage = outcome.usage

        yield emit(f"Request count: 1. Model reported by the API: {returned_model or '(not reported)'}"), None
        if usage is not None:
            yield emit(
                f"Token usage - input: {usage.input_tokens}, output: {usage.output_tokens}, "
                f"total: {usage.total_tokens}"
            ), None
        else:
            yield emit("Token usage: not reported by the API response."), None
        yield emit(estimate_cost_usd(returned_model, usage)), None

        yield emit(f"Summary evidence IDs: {decision_obj.summary_evidence_ids}"), None
        yield emit("Selected evidence:"), None
        for item in decision_obj.selected_evidence:
            yield emit(f"  {item.chunk_id}: {item.reason}"), None
        yield emit("Omitted evidence:"), None
        for item in decision_obj.omitted_evidence:
            yield emit(f"  {item.chunk_id}: {item.reason}"), None

        run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
        run_dir = OUTPUTS_DIR / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        yield emit(f"Saving validated draft under: {run_dir}"), None

        draft_payload = {
            "schema_version": 1,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "model": returned_model,
            "usage": (
                {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens, "total_tokens": usage.total_tokens}
                if usage is not None
                else None
            ),
            "draft": decision_to_dict(decision_obj),
        }
        (run_dir / "draft.json").write_text(json.dumps(draft_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        docx_path = run_dir / "resume.docx"
        yield emit("Rendering the Word document and checking page count..."), None
        try:
            render_result = render_and_verify(decision_obj, contact, chunks, docx_path)
        except DocumentGenerationError as exc:
            yield emit(f"Could not generate the document: {exc}"), None
            return

        for preset_name, page_count in render_result.attempts:
            shown = page_count if page_count is not None else "conversion error"
            yield emit(f"  preset '{preset_name}' -> {shown} page(s)"), None
        yield emit(f"Render status: {render_result.status}"), None
        yield emit(render_result.message), None

        if render_result.status == "verified_two_pages":
            yield emit("Download enabled: exactly two pages verified."), str(render_result.docx_path)
        elif render_result.status == "generated_target_not_met":
            yield emit(
                "Download enabled: document generated successfully, but the two-page target was not met "
                "(see page count above)."
            ), str(render_result.docx_path)
        else:
            yield emit("Download not enabled: document generation or rendering did not succeed."), None

    finally:
        _RUN_LOCK.release()
