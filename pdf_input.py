"""pdf_input.py

Purpose
-------
Local, offline text extraction for the resume-tailoring pipeline's
job-description input. Given the path to an uploaded PDF, extract_pdf_text()
returns the text pypdf can find on each page, the pages joined into one
string, and plain-language warnings about anything that looks wrong or
unusable. It calls no network/API and performs no OCR - a page with no
embedded text layer (a scanned image, for example) simply has no text to
return, and this module says so rather than pretending otherwise.

Where this fits in the application
-----------------------------------
extract_pdf_text() is the first stage for a job description: its
`full_text` output is what would be passed to retrieve_relevant_chunks()
(in retrieval.py) as `job_text`. Candidate-profile loading (retrieval.py's
load_profile()), the OpenAI drafting call, Word document generation, and
the Gradio UI are separate stages and are not implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from pypdf import PdfReader
from pypdf.errors import PdfReadError

PathLike = Union[str, Path]


class PdfPasswordProtectedError(ValueError):
    """Raised when a PDF is encrypted and does not open with an empty
    password. This module never prompts for or guesses a real password -
    callers needing that would have to supply one through a different
    path, which is out of scope here."""


@dataclass(frozen=True)
class PdfExtractionResult:
    """Result of extracting text from one local PDF.

    pages: per-page extracted text, one entry per page, in page order.
        A page with no extractable text (a scanned/image-only page, or a
        genuinely blank page) comes back as "". This module cannot tell
        those two cases apart - that would require OCR or image
        analysis, and neither is implemented.
    full_text: every page's text joined with blank lines, then stripped;
        this is the string meant to be handed to retrieve_relevant_chunks()
        as job_text.
    warnings: human-readable notices about degraded-but-handled
        conditions (never exceptions - see extract_pdf_text's Raises
        section for the cases that *do* raise). If `full_text` ends up
        empty, one of these warnings always says so explicitly, so a
        caller cannot mistake an unusable result for a successful one.
    """

    pages: List[str]
    full_text: str
    warnings: List[str]


def extract_pdf_text(path: PathLike) -> PdfExtractionResult:
    r"""Extract text from a local PDF file using pypdf only (no OCR).

    Input
    -----
    path: path to a PDF file (e.g. an uploaded job description).

    Output
    ------
    A PdfExtractionResult. Pages with no extractable text, or a
    whole document with none, are reported through `warnings` and an
    empty `full_text`/`pages` entries - not raised as errors, since a
    scanned or blank page is valid PDF content, just not text this
    module can read.

    Raises
    ------
    FileNotFoundError: `path` does not exist. This is a caller mistake
        (e.g. a broken upload path), not a property of the PDF itself.
    ValueError: the file cannot be parsed as a PDF at all, or pypdf
        raises a PdfReadError (e.g. its PdfStreamError subclass) while
        opening it, counting its pages, or reading a specific page's
        content; this module re-raises that as a plain ValueError
        naming the file (and page number, where known), with the
        original PdfReadError preserved as `__cause__`.
    PdfPasswordProtectedError (a ValueError subclass): the PDF is
        encrypted and an empty password does not open it.

    Only those expected, pypdf-specific failure modes are caught. Any
    other exception (a bug in this function, a memory error, etc.) is
    deliberately left to propagate rather than being hidden.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    try:
        reader = PdfReader(str(path))
    except PdfReadError as exc:
        raise ValueError(f"not a valid PDF: {path} ({exc})") from exc

    if reader.is_encrypted:
        try:
            # Many "encrypted" PDFs set only an empty user password to
            # restrict permissions (printing, copying) without hiding
            # content; decrypt("") unlocks exactly those. A return value
            # of 0 means even that failed, i.e. a real password is set.
            decrypt_result = reader.decrypt("")
        except PdfReadError as exc:
            raise PdfPasswordProtectedError(
                f"PDF is password-protected and could not be opened: {path}"
            ) from exc
        if decrypt_result == 0:
            raise PdfPasswordProtectedError(
                f"PDF requires a real password that was not provided: {path}"
            )

    warnings: List[str] = []

    try:
        page_count = len(reader.pages)
    except PdfReadError as exc:
        raise ValueError(f"failed to read page count of {path}: {exc}") from exc

    if page_count == 0:
        warnings.append(
            "PDF has zero pages; there is no text to extract, so "
            "retrieval/drafting cannot proceed from this file as-is."
        )
        return PdfExtractionResult(pages=[], full_text="", warnings=warnings)

    pages: List[str] = []
    for index in range(1, page_count + 1):
        try:
            page = reader.pages[index - 1]
            pages.append(page.extract_text() or "")
        except PdfReadError as exc:
            raise ValueError(f"failed to read page {index} of {path}: {exc}") from exc

    blank_page_numbers = [i for i, text in enumerate(pages, start=1) if not text.strip()]
    if blank_page_numbers:
        if len(blank_page_numbers) == len(pages):
            warnings.append(
                "No extractable text was found on any page. This PDF may be "
                "a scanned image or otherwise non-text content; OCR is not "
                "implemented, so retrieval/drafting cannot proceed from "
                "this file as-is."
            )
        else:
            warnings.append(
                f"{len(blank_page_numbers)} of {len(pages)} page(s) had no "
                f"extractable text (page numbers: {blank_page_numbers}). "
                "This module cannot tell a blank page from a scanned image "
                "- it only reports that no text was found."
            )

    full_text = "\n\n".join(pages).strip()
    if not full_text and not blank_page_numbers:
        # Defensive: every page had some non-whitespace text per the
        # per-page check above, yet the joined/stripped result is empty.
        # Not expected to happen, but don't let it pass as "successful".
        warnings.append(
            "Extracted text is empty; retrieval/drafting cannot proceed "
            "from this file as-is."
        )

    return PdfExtractionResult(pages=pages, full_text=full_text, warnings=warnings)
