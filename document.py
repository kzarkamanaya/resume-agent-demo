"""document.py

Purpose
-------
Turns a validated drafting decision (from decision.py) into a Word
resume, then checks how many pages it actually renders to - all without
calling the OpenAI API. One file, three jobs, because none of them is
big enough alone to justify splitting:
  1. Extract institution/dates and location/dates text straight from the
     profile's own wording for education and job entries (never from the
     model - see "Extraction rules" below).
  2. Build the .docx with python-docx.
  3. Convert it to PDF with LibreOffice headless and count pages with
     pypdf, trying a small, bounded set of formatting presets.

Where this fits in the application
-----------------------------------
Input: a decision.DraftResult, the contact dict and chunks from
retrieval.py's load_profile() (contact/chunks only - metadata is not
used here either). Output: a .docx file plus a RenderResult describing
whether exactly two pages were verified.

This module also provides a render-only CLI (see main()) that loads a
JSON file saved by decision.py's `--save-json`, re-validates its
evidence IDs against a freshly loaded profile, and renders it - without
ever importing OpenAI or constructing an API client. That is what makes
"render repeatedly without another API call" possible.

Extraction rules (deliberately narrow, not a Markdown framework)
------------------------------------------------------------------
retrieval.py's Chunk has no separate institution/location/date fields -
those live as plain prose inside `chunk.text`, and the drafted
decision.Bullet objects are not guaranteed to repeat them. So this
module reads them straight from the two bullet formats resume_profile.md
already uses:
  - Education: the record's first bullet line, shaped like
    "InstitutionName; completed|expected YEAR[; more]".
  - Jobs: a "- Location: ..." line and a "- Dates: ..." line, found
    anywhere in the record's bullets (order-independent, so a reordered
    profile still extracts correctly).
Both are matched with a couple of small regexes against exact profile
wording - no rewording, no synonyms, no generic bullet parser. Projects
and skills categories have no such field and are left alone. If a
required line is missing or doesn't match the expected shape for an
education/job chunk, that chunk's reason is collected and reported
clearly (see DocumentGenerationError) - nothing is guessed or silently
dropped.

A note on page counting
------------------------
LibreOffice and Microsoft Word do not always paginate an identical
document exactly the same way (font metrics, justification, and widow/
orphan handling can differ slightly). A LibreOffice-verified "exactly
two pages" result is the best local check available here, not an
absolute guarantee of what Word will show.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from pypdf import PdfReader

from decision import (
    DraftResult,
    ResumeEntry,
    _section_heading,
    build_decision,
    validate_decision,
)
from retrieval import Chunk, PROFILE_PATH, load_profile

MIN_BODY_FONT_PT = 10.5
MIN_MARGIN_IN = 0.6
TARGET_PAGE_COUNT = 2

DEFAULT_SOFFICE_PATH = Path(
    os.environ.get("LIBREOFFICE_PATH", r"C:\Program Files\LibreOffice\program\soffice.exe")
)

# Resume convention: a stable, predictable section order, regardless of
# what order resume_sections happened to arrive in from the model.
_SECTION_ORDER = ["job", "education_training", "project", "skill_category"]


class DocumentGenerationError(RuntimeError):
    """Raised before anything is written to disk, when one or more
    education/job entries' institution/dates or location/dates cannot
    be reliably extracted from the profile's own wording."""


class LibreOfficeNotFoundError(RuntimeError):
    """Raised when the configured soffice executable does not exist."""


class LibreOfficeConversionError(RuntimeError):
    """Raised when LibreOffice exits non-zero, times out, or does not
    produce the expected PDF file."""


@dataclass(frozen=True)
class EntryHeaderExtraction:
    """Result of extracting an education/job sub-heading line.

    ok: whether extraction succeeded.
    institution/location/dates: the exact substrings found (verbatim,
        only whitespace-trimmed) - only the fields relevant to this
        chunk's section are populated.
    reason: set only when ok is False, naming the chunk_id and what
        was expected but not found.
    """

    ok: bool
    institution: Optional[str] = None
    location: Optional[str] = None
    dates: Optional[str] = None
    reason: Optional[str] = None


_EDU_INSTITUTION_RE = re.compile(r"^-\s*(?P<institution>[^;]+?)\s*;")
_EDU_DATE_PHRASE_RE = re.compile(r"\b(?:completed|expected)\s+\d{4}\b", re.IGNORECASE)
_JOB_LOCATION_RE = re.compile(r"^-\s*Location:\s*(?P<value>.+)$", re.IGNORECASE)
_JOB_DATES_RE = re.compile(r"^-\s*Dates:\s*(?P<value>.+)$", re.IGNORECASE)


def _bullet_lines(chunk_text: str) -> List[str]:
    return [line.strip() for line in chunk_text.splitlines() if line.strip().startswith("-")]


def extract_education_header(chunk: Chunk) -> EntryHeaderExtraction:
    """Pull "institution" and "dates" from an education_training chunk's
    first bullet line: "InstitutionName; completed|expected YEAR[...]"."""
    bullets = _bullet_lines(chunk.text)
    if not bullets:
        return EntryHeaderExtraction(ok=False, reason=f"{chunk.chunk_id}: record has no bullet lines at all")

    first_bullet = bullets[0]
    institution_match = _EDU_INSTITUTION_RE.match(first_bullet)
    if not institution_match:
        return EntryHeaderExtraction(
            ok=False,
            reason=(
                f"{chunk.chunk_id}: first bullet does not match "
                f"'Institution; completed|expected YEAR' format: {first_bullet!r}"
            ),
        )

    date_match = _EDU_DATE_PHRASE_RE.search(first_bullet)
    if not date_match:
        return EntryHeaderExtraction(
            ok=False,
            reason=f"{chunk.chunk_id}: no 'completed/expected YEAR' phrase found in: {first_bullet!r}",
        )

    return EntryHeaderExtraction(
        ok=True,
        institution=institution_match.group("institution").strip(),
        dates=date_match.group(0),
    )


def extract_job_header(chunk: Chunk) -> EntryHeaderExtraction:
    """Pull "location" and "dates" from a job chunk's "- Location: ..."
    and "- Dates: ..." bullet lines. Scans every bullet rather than
    assuming a position, so a profile with those two lines reordered
    still extracts correctly."""
    location_value: Optional[str] = None
    dates_value: Optional[str] = None

    for bullet in _bullet_lines(chunk.text):
        if location_value is None:
            loc_match = _JOB_LOCATION_RE.match(bullet)
            if loc_match:
                location_value = loc_match.group("value").strip()
                continue
        if dates_value is None:
            dates_match = _JOB_DATES_RE.match(bullet)
            if dates_match:
                dates_value = dates_match.group("value").strip()

    missing = [label for label, value in (("Location:", location_value), ("Dates:", dates_value)) if value is None]
    if missing:
        return EntryHeaderExtraction(
            ok=False, reason=f"{chunk.chunk_id}: missing expected job line(s) {missing}"
        )

    return EntryHeaderExtraction(ok=True, location=location_value, dates=dates_value)


def extract_entry_header(chunk: Chunk) -> Optional[EntryHeaderExtraction]:
    """Dispatch by section: education/job get extraction; project and
    skill_category have no institution/location/date concept and return
    None (nothing extra to render below the title)."""
    if chunk.section == "education_training":
        return extract_education_header(chunk)
    if chunk.section == "job":
        return extract_job_header(chunk)
    return None


@dataclass(frozen=True)
class FormattingPreset:
    name: str
    body_font_pt: float
    margin_in: float
    line_spacing: float
    space_after_pt: float


FORMATTING_PRESETS = (
    FormattingPreset("comfortable", 11.0, 0.75, 1.08, 6.0),
    FormattingPreset("compact", 10.5, 0.7, 1.0, 4.0),
    FormattingPreset("tight", 10.5, 0.6, 1.0, 2.0),
)
for _preset in FORMATTING_PRESETS:
    assert _preset.body_font_pt >= MIN_BODY_FONT_PT, "preset below the readable font floor"
    assert _preset.margin_in >= MIN_MARGIN_IN, "preset below the margin floor"


def _ordered_sections(decision: DraftResult) -> List:
    order_index = {label: i for i, label in enumerate(_SECTION_ORDER)}
    return sorted(decision.resume_sections, key=lambda s: order_index.get(s.section, len(_SECTION_ORDER)))


def build_docx(
    decision: DraftResult,
    contact: Dict[str, str],
    chunks: List[Chunk],
    output_path: Path,
    preset: FormattingPreset,
) -> None:
    """Write one .docx from a validated decision. Raises
    DocumentGenerationError (before writing anything) if any
    education/job entry's header line cannot be extracted. Excludes
    evidence IDs and selection/omission reasons entirely - those never
    appear in the document."""
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    headers: Dict[str, Optional[EntryHeaderExtraction]] = {}
    failures: List[str] = []
    for section in decision.resume_sections:
        for entry in section.entries:
            chunk = chunk_by_id.get(entry.source_chunk_id)
            if chunk is None:
                failures.append(f"{entry.source_chunk_id}: not found among the loaded profile chunks")
                continue
            header = extract_entry_header(chunk)
            headers[entry.source_chunk_id] = header
            if header is not None and not header.ok:
                failures.append(header.reason or f"{chunk.chunk_id}: extraction failed")

    if failures:
        raise DocumentGenerationError(
            "could not reliably extract required fields for "
            f"{len(failures)} entry/entries: " + "; ".join(failures)
        )

    document = Document()
    section_fmt = document.sections[0]
    section_fmt.page_width = Inches(8.5)
    section_fmt.page_height = Inches(11.0)
    section_fmt.left_margin = Inches(preset.margin_in)
    section_fmt.right_margin = Inches(preset.margin_in)
    section_fmt.top_margin = Inches(preset.margin_in)
    section_fmt.bottom_margin = Inches(preset.margin_in)

    normal = document.styles["Normal"]
    normal.font.size = Pt(preset.body_font_pt)
    normal.font.name = "Calibri"
    normal.paragraph_format.line_spacing = preset.line_spacing
    normal.paragraph_format.space_after = Pt(preset.space_after_pt)
    normal.paragraph_format.space_before = Pt(0)

    list_bullet = document.styles["List Bullet"]
    list_bullet.font.size = Pt(preset.body_font_pt)
    list_bullet.font.name = "Calibri"
    list_bullet.paragraph_format.line_spacing = preset.line_spacing
    list_bullet.paragraph_format.space_after = Pt(preset.space_after_pt)
    list_bullet.paragraph_format.space_before = Pt(0)

    _add_contact_header(document, contact)

    _add_major_heading(document, "Professional Summary")
    document.add_paragraph(decision.professional_summary)

    for section in _ordered_sections(decision):
        _add_major_heading(document, _section_heading(section.section))

        for entry in section.entries:
            chunk = chunk_by_id[entry.source_chunk_id]
            title_paragraph = document.add_paragraph()
            title_paragraph.paragraph_format.keep_with_next = True
            title_run = title_paragraph.add_run(chunk.title)
            title_run.bold = True
            title_run.font.size = Pt(preset.body_font_pt)

            header = headers.get(entry.source_chunk_id)
            if header is not None and header.ok:
                if header.institution is not None:
                    detail_paragraph = document.add_paragraph(f"{header.institution} \u2014 {header.dates}")
                else:
                    detail_paragraph = document.add_paragraph(f"{header.location}  |  {header.dates}")
                detail_paragraph.paragraph_format.keep_with_next = True

            for bullet in entry.bullets:
                document.add_paragraph(bullet.text, style="List Bullet")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(output_path))


def _add_major_heading(document: "Document", text: str) -> None:
    """A major heading (Professional Summary, Professional Experience,
    etc.): 12pt bold black, with a little space before/after to stand
    out clearly from body text, while staying attached to the content
    under it (keep_with_next) - never pushed onto its own page."""
    heading = document.add_paragraph()
    heading.alignment = WD_ALIGN_PARAGRAPH.LEFT
    heading.paragraph_format.keep_with_next = True
    heading.paragraph_format.space_before = Pt(8)
    heading.paragraph_format.space_after = Pt(4)
    run = heading.add_run(text)
    run.bold = True
    run.font.size = Pt(12)
    run.font.color.rgb = RGBColor(0, 0, 0)


def _add_contact_header(document: "Document", contact: Dict[str, str]) -> None:
    """Local-only contact header: whatever fields are present, shown
    exactly as parsed - never a placeholder for a missing one. Name is
    left-aligned; phone/email share one line; the full address (if any)
    is its own, separate paragraph."""
    name = contact.get("Name")
    if name:
        heading = document.add_paragraph()
        heading.alignment = WD_ALIGN_PARAGRAPH.LEFT
        heading.paragraph_format.keep_with_next = True
        run = heading.add_run(name)
        run.bold = True
        run.font.size = Pt(16)

    phone_email_parts = [contact[key] for key in ("Phone", "Email") if contact.get(key)]
    if phone_email_parts:
        line = document.add_paragraph(" | ".join(phone_email_parts))
        line.alignment = WD_ALIGN_PARAGRAPH.LEFT
        line.paragraph_format.keep_with_next = True

    address = contact.get("Address")
    if address:
        address_paragraph = document.add_paragraph(address)
        address_paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT


def convert_docx_to_pdf(docx_path: Path, attempt_dir: Path, soffice_path: Path, timeout_seconds: int = 45) -> Path:
    """Convert docx_path to PDF via LibreOffice headless, writing into
    `attempt_dir`.

    `attempt_dir` must already be a fresh, unique directory (the caller
    is expected to create one per attempt, e.g. via
    tempfile.mkdtemp()) - this function does not delete or reuse an
    existing directory; it only creates the "lo_profile" subdirectory
    LibreOffice needs inside the one it was given. That is what
    guarantees a stale PDF from an earlier attempt can never be mistaken
    for this attempt's result: every attempt gets its own directory
    that never existed before.

    Uses the explicit "pdf:writer_pdf_Export" filter rather than the
    bare "pdf" target, which is the documented way to request a direct
    PDF export instead of a print-pipeline code path. This does NOT
    guarantee Windows printer enumeration never happens - that
    enumeration can occur during LibreOffice's own startup regardless of
    which export filter is requested, and depends on Windows/printer-
    driver behavior outside this function's control.

    Also points LibreOffice at a brand-new, private user profile for
    this attempt (`-env:UserInstallation=...`, built from
    attempt_dir.resolve().as_uri()), so a stuck or first-run profile
    from an earlier attempt can never block or alter this one.

    On a timeout, subprocess.run() kills the soffice process it directly
    started and this function raises LibreOfficeConversionError - it
    does NOT guarantee every LibreOffice-related process has exited; if
    soffice spawned additional helper processes, some could still be
    running after this function returns. This function itself never
    retries internally; render_and_verify() decides whether a different
    preset is worth trying next, and never relaunches the same blocked
    attempt in a loop.
    """
    if not soffice_path.exists():
        raise LibreOfficeNotFoundError(f"LibreOffice executable not found at {soffice_path}")

    profile_dir = attempt_dir / "lo_profile"
    profile_uri = profile_dir.resolve().as_uri()
    expected_pdf = attempt_dir / (docx_path.stem + ".pdf")

    try:
        result = subprocess.run(
            [
                str(soffice_path),
                "--headless",
                "--norestore",
                f"-env:UserInstallation={profile_uri}",
                "--convert-to",
                "pdf:writer_pdf_Export",
                "--outdir",
                str(attempt_dir),
                str(docx_path),
            ],
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise LibreOfficeConversionError(
            f"LibreOffice conversion timed out after {timeout_seconds}s (the directly-started process "
            "was killed; this does not guarantee every LibreOffice-related process exited - not retried)"
        ) from exc

    if result.returncode != 0 or not expected_pdf.exists():
        raise LibreOfficeConversionError(
            f"LibreOffice conversion failed (exit code {result.returncode}); expected {expected_pdf.name} was not produced"
        )

    return expected_pdf


def count_pdf_pages(pdf_path: Path) -> int:
    return len(PdfReader(str(pdf_path)).pages)


@dataclass(frozen=True)
class RenderResult:
    """docx_path: always written. pdf_path/page_count: set only when a
    LibreOffice conversion actually produced a page count (they reflect
    the LAST attempt made, not necessarily a "best" one - see
    `attempts` for the full, factual history). status: one of
    "verified_two_pages" (exact target hit - the only one eligible for
    a "verified" download message), "generated_target_not_met" (a
    successful render that did not land on exactly two pages - still a
    real, downloadable document), "unverified_no_libreoffice", or
    "failed_conversion_error" (these last two block the download -
    nothing was reliably produced/measured). attempts: every
    (preset_name, page_count) actually tried, in order, for honest
    reporting - never implies any one of them was selected as "best"
    unless status is "verified_two_pages" (in which case it's the last
    entry)."""

    docx_path: Path
    pdf_path: Optional[Path]
    page_count: Optional[int]
    preset_name: Optional[str]
    status: str
    message: str
    attempts: List[Tuple[str, Optional[int]]] = field(default_factory=list)


_PAGINATION_NOTE = (
    "Note: LibreOffice and Microsoft Word can paginate the same document "
    "slightly differently, so this is the best local check available, not "
    "an absolute guarantee of what Word will show."
)


def render_and_verify(
    decision: DraftResult,
    contact: Dict[str, str],
    chunks: List[Chunk],
    docx_path: Path,
    soffice_path: Path = DEFAULT_SOFFICE_PATH,
) -> RenderResult:
    """Build the .docx and, if LibreOffice is available, try up to
    len(FORMATTING_PRESETS) presets, from most spacious to tightest,
    looking for an exact two-page render. Never pads content, never
    deletes content, and never calls the API - this only measures what
    build_docx() already produced.

    Exactly TARGET_PAGE_COUNT pages is the *target*, not a hard
    requirement for offering a download: any successfully-rendered
    page count (1, 2, 3+) results in a usable .docx and a status that
    is never "failed" - only an actual document-generation error
    (DocumentGenerationError, raised before this returns) or a
    rendering/conversion error blocks the download (see
    "failed_conversion_error" / "unverified_no_libreoffice" below).

    - Exactly 2 pages on any attempt -> status "verified_two_pages"
      (the strongest outcome), and stops there.
    - An overlong render (more than 2 pages) continues on to the next,
      tighter preset, same as before.
    - An undersized render (fewer than 2 pages) stops immediately
      instead of trying tighter presets - a tighter preset can only
      shrink an already-short document further, so trying one would
      not help.
    - If every attempt renders successfully but none lands on exactly
      2 pages (undersized, or overlong even after the tightest preset),
      status is "generated_target_not_met": the document from the last
      attempt is still offered as a fallback download, with a plain
      "Generated N pages; two-page target not met." message. No content
      is added or removed to chase the target, and no shortening
      request is made.
    """
    attempts_root = docx_path.parent / f"_{docx_path.stem}_render_attempts"

    if not soffice_path.exists():
        build_docx(decision, contact, chunks, docx_path, FORMATTING_PRESETS[0])
        return RenderResult(
            docx_path=docx_path,
            pdf_path=None,
            page_count=None,
            preset_name=FORMATTING_PRESETS[0].name,
            status="unverified_no_libreoffice",
            message=f"LibreOffice was not found at {soffice_path}; DOCX was created but page count is unverified. "
            + _PAGINATION_NOTE,
            attempts=[],
        )

    attempts_root.mkdir(parents=True, exist_ok=True)
    attempts: List[Tuple[str, Optional[int]]] = []
    last_pdf_path: Optional[Path] = None

    for attempt_index, preset in enumerate(FORMATTING_PRESETS, start=1):
        build_docx(decision, contact, chunks, docx_path, preset)
        attempt_dir = Path(tempfile.mkdtemp(dir=attempts_root, prefix=f"attempt_{attempt_index}_"))
        try:
            pdf_path = convert_docx_to_pdf(docx_path, attempt_dir, soffice_path)
            page_count = count_pdf_pages(pdf_path)
        except (LibreOfficeConversionError, LibreOfficeNotFoundError) as exc:
            attempts.append((preset.name, None))
            return RenderResult(
                docx_path=docx_path,
                pdf_path=None,
                page_count=None,
                preset_name=preset.name,
                status="failed_conversion_error",
                message=f"{exc} Attempts so far: {attempts}.",
                attempts=attempts,
            )

        attempts.append((preset.name, page_count))
        last_pdf_path = pdf_path

        if page_count == TARGET_PAGE_COUNT:
            return RenderResult(
                docx_path=docx_path,
                pdf_path=pdf_path,
                page_count=page_count,
                preset_name=preset.name,
                status="verified_two_pages",
                message=f"Verified exactly {TARGET_PAGE_COUNT} rendered pages using the '{preset.name}' preset. "
                f"Attempts: {attempts}. " + _PAGINATION_NOTE,
                attempts=attempts,
            )

        if page_count < TARGET_PAGE_COUNT:
            return RenderResult(
                docx_path=docx_path,
                pdf_path=pdf_path,
                page_count=page_count,
                preset_name=preset.name,
                status="generated_target_not_met",
                message=(
                    f"Generated {page_count} page(s); two-page target not met. Rendered using the "
                    f"'{preset.name}' preset. A tighter preset would only shrink this further, so none "
                    "was tried. No content was added or removed, and no shortening request was made - "
                    f"the document is still available to download. Attempts so far: {attempts}."
                ),
                attempts=attempts,
            )

        # page_count > TARGET_PAGE_COUNT: overlong, so continue to the next, tighter preset.

    final_page_count = attempts[-1][1]
    return RenderResult(
        docx_path=docx_path,
        pdf_path=last_pdf_path,
        page_count=final_page_count,
        preset_name=attempts[-1][0],
        status="generated_target_not_met",
        message=(
            f"Generated {final_page_count} page(s); two-page target not met. Tried {len(attempts)} "
            f"formatting preset(s), all remained overlong. No content was added or removed, and no "
            f"shortening request was made - the document is still available to download. Attempts: {attempts}."
        ),
        attempts=attempts,
    )


def load_saved_draft(path: Path) -> Dict[str, Any]:
    """Load a JSON file written by decision.py's --save-json. Returns
    the raw dict as-is; validate_decision() must still be run on
    saved["draft"] before trusting it (the profile may have changed
    since the draft was saved)."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main(argv: Optional[List[str]] = None) -> int:
    """Render-only CLI: load a saved draft, validate its evidence IDs
    against a freshly loaded profile, and render it. Never imports or
    constructs an OpenAI client - no API call is possible from here."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Render a saved resume draft (from decision.py --save-json) to a page-counted .docx. Makes no API calls."
    )
    parser.add_argument("saved_json", type=Path, help="Path to a JSON file saved by decision.py --save-json.")
    parser.add_argument("--profile", type=Path, default=PROFILE_PATH, help="Path to resume_profile.md.")
    parser.add_argument("--out", type=Path, default=Path("outputs/resume.docx"), help="Where to write the .docx.")
    parser.add_argument("--soffice", type=Path, default=DEFAULT_SOFFICE_PATH, help="Path to the LibreOffice executable.")
    args = parser.parse_args(argv)

    saved = load_saved_draft(args.saved_json)
    draft_raw = saved.get("draft")
    if not isinstance(draft_raw, dict):
        print(f"{args.saved_json} has no 'draft' object; nothing to render.")
        return 1

    _metadata, contact, chunks = load_profile(args.profile)
    known_chunk_ids: Set[str] = {chunk.chunk_id for chunk in chunks}

    problems = validate_decision(draft_raw, known_chunk_ids)
    if problems:
        print("Saved draft failed validation against the current profile:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    decision = build_decision(draft_raw)

    try:
        result = render_and_verify(decision, contact, chunks, args.out, args.soffice)
    except DocumentGenerationError as exc:
        print(f"Could not generate the document: {exc}")
        return 1

    print(f"DOCX written to: {result.docx_path}")
    print(f"Status: {result.status}")
    print(result.message)
    return 0 if result.status in ("verified_two_pages", "generated_target_not_met") else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
