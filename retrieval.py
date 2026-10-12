"""retrieval.py

Purpose
-------
Heading-based loader, plus local TF-IDF retrieval, for resume_profile.md.
load_profile() turns the single Markdown profile into a list of small,
retrievable "chunks"; retrieve_relevant_chunks() then ranks those chunks
against a job description's text using scikit-learn TF-IDF, entirely
locally (no network/API calls). This is the "local fetch" stage of the
resume-tailoring pipeline - its output is what a later drafting step
would hand to the LLM as candidate evidence.

This module reads resume_profile.md but never modifies it, and never
calls any external API.

Document shape this parser expects
-----------------------------------
- A YAML-style frontmatter block delimited by "---" lines, holding
  metadata (candidate_name, use_mode, etc.).
- A "## Contact header" section with "- Key: value" bullets. This is kept
  separate from retrieval chunks and is never surfaced by the CLI output
  below, since contact details should not be injected into tailored
  retrieval context by accident.
- "## Education", "## Professional experience", and "## Projects"
  sections, each made of one or more "### Title" records. Everything
  under a "### Title" line (dates, descriptions, tools, coursework) until
  the next "### " or "## " heading belongs to that one record/chunk.
- A "## Skills" section made of bullet lines shaped like
  "- **Category label:** comma separated items". Each such bullet is one
  chunk.

Where this fits in the application
-----------------------------------
load_profile() is the function later pipeline stages are expected to call
first. It returns (metadata, contact, chunks): metadata/contact support
header fields like name/phone/email elsewhere in the app, while chunks is
the list that retrieve_relevant_chunks() indexes and searches. The next
stage (not implemented here) is a single OpenAI call that drafts resume
content from whatever chunks retrieval returns; PDF extraction of the job
description, Word document generation, and the Gradio UI are also later
stages, not part of this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer

PROFILE_PATH = Path(__file__).parent / "resume_profile.md"

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
H2_RE = re.compile(r"^##\s+(.+?)\s*$")
H3_RE = re.compile(r"^###\s+(.+?)\s*$")
SKILL_BULLET_RE = re.compile(r"^-\s+\*\*(.+?):\*\*\s*(.*)$")

# Maps a "## " section heading in the profile to the (section label used on
# chunks, chunk-id prefix) pair for that record type. Education/jobs/projects
# are all "one chunk per ### heading" sections, so they share one splitter.
RECORD_SECTIONS = {
    "Education": ("education_training", "edu"),
    "Professional experience": ("job", "job"),
    "Projects": ("project", "proj"),
}


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of the profile.

    chunk_id: deterministic unique id, e.g. "proj-05" (stable across runs
        as long as resume_profile.md's record order doesn't change).
    section: record type - "education_training", "job", "project", or
        "skill_category".
    title: the record's heading (degree/employer-role/project name) or,
        for skills, the bold category label.
    text: the complete original record text (heading plus every bullet
        under it - dates, descriptions, tools, coursework all stay
        together), used verbatim downstream so nothing is lost.
    """

    chunk_id: str
    section: str
    title: str
    text: str


def read_profile_text(path: Path = PROFILE_PATH) -> str:
    """Read resume_profile.md as-is. Input: path to the profile. Output:
    the raw file text. Never writes to the file."""
    return path.read_text(encoding="utf-8")


def split_frontmatter(raw_text: str) -> Tuple[Dict[str, str], str]:
    """Split the leading '---' YAML-style block from the markdown body.

    Input: the full raw profile text.
    Output: (metadata dict, remaining body text with frontmatter removed).
    """
    match = FRONTMATTER_RE.match(raw_text)
    if not match:
        return {}, raw_text

    metadata: Dict[str, str] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        metadata[key.strip()] = value.strip().strip('"')

    body = raw_text[match.end():]
    return metadata, body


def split_top_level_sections(body: str) -> Dict[str, str]:
    """Split the body into '## Heading' sections.

    Input: markdown body (frontmatter already removed).
    Output: dict mapping each '## ' heading's text to the raw text found
    under it, up to (but not including) the next '## ' heading.
    """
    sections: Dict[str, str] = {}
    current_title: str | None = None
    current_lines: List[str] = []

    for line in body.splitlines():
        heading = H2_RE.match(line)
        if heading:
            if current_title is not None:
                sections[current_title] = "\n".join(current_lines).strip("\n")
            current_title = heading.group(1).strip()
            current_lines = []
        elif current_title is not None:
            current_lines.append(line)

    if current_title is not None:
        sections[current_title] = "\n".join(current_lines).strip("\n")

    return sections


def parse_contact_header(section_text: str) -> Dict[str, str]:
    """Parse the Contact header's '- Key: value' bullets.

    Input: raw text of the '## Contact header' section.
    Output: dict of contact fields (Name/Phone/Email/Address). This is
    kept separate from the chunk list returned by load_profile() and the
    CLI output below never prints it, so contact details cannot leak into
    retrieval context by accident.
    """
    contact: Dict[str, str] = {}
    for line in section_text.splitlines():
        line = line.strip()
        if not line.startswith("-") or ":" not in line:
            continue
        key, _, value = line[1:].strip().partition(":")
        contact[key.strip()] = value.strip()
    return contact


def split_record_chunks(section_text: str, section_label: str, id_prefix: str) -> List[Chunk]:
    """Split one '## ' section's body into one chunk per '### ' record.

    Used for Education, Professional experience, and Projects: each
    heading plus everything under it (dates, descriptions, tools,
    coursework) until the next '### ' heading becomes one Chunk.
    """
    chunks: List[Chunk] = []
    current_title: str | None = None
    current_body: List[str] = []
    counter = 0

    def finalize(title: str, body_lines: List[str], index: int) -> Chunk:
        text = "\n".join([f"### {title}"] + body_lines).strip("\n")
        return Chunk(f"{id_prefix}-{index:02d}", section_label, title, text)

    for line in section_text.splitlines():
        heading = H3_RE.match(line)
        if heading:
            if current_title is not None:
                counter += 1
                chunks.append(finalize(current_title, current_body, counter))
            current_title = heading.group(1).strip()
            current_body = []
        elif current_title is not None:
            current_body.append(line)

    if current_title is not None:
        counter += 1
        chunks.append(finalize(current_title, current_body, counter))

    return chunks


def split_skill_chunks(section_text: str, id_prefix: str = "skill") -> List[Chunk]:
    """Split the '## Skills' section into one chunk per bold-labeled bullet.

    Each '- **Category:** items...' bullet (plus any wrapped continuation
    lines that follow it before the next bulleted category) becomes one
    Chunk, titled with the category label.
    """
    chunks: List[Chunk] = []
    current_label: str | None = None
    current_body: List[str] = []
    counter = 0

    def finalize(label: str, body_lines: List[str], index: int) -> Chunk:
        text = "\n".join(body_lines).strip("\n")
        return Chunk(f"{id_prefix}-{index:02d}", "skill_category", label, text)

    for raw_line in section_text.splitlines():
        line = raw_line.strip()
        bullet = SKILL_BULLET_RE.match(line)
        if bullet:
            if current_label is not None:
                counter += 1
                chunks.append(finalize(current_label, current_body, counter))
            current_label = bullet.group(1).strip()
            current_body = [line]
        elif current_label is not None and line:
            current_body.append(line)

    if current_label is not None:
        counter += 1
        chunks.append(finalize(current_label, current_body, counter))

    return chunks


def load_profile(path: Path = PROFILE_PATH) -> Tuple[Dict[str, str], Dict[str, str], List[Chunk]]:
    """Load and structure resume_profile.md.

    Input: path to the profile (defaults to resume_profile.md next to
    this module).
    Output: (metadata, contact, chunks) -
        metadata - frontmatter key/value pairs.
        contact - Contact header fields, kept out of chunks.
        chunks - one Chunk per education/training record, job, project,
            and skills category (28 for the supplied demo profile).

    This is the function the later TF-IDF retrieval stage is expected to
    call to get its searchable corpus.
    """
    raw_text = read_profile_text(path)
    metadata, body = split_frontmatter(raw_text)
    sections = split_top_level_sections(body)

    contact = parse_contact_header(sections.get("Contact header", ""))

    chunks: List[Chunk] = []
    for heading, (section_label, id_prefix) in RECORD_SECTIONS.items():
        chunks.extend(split_record_chunks(sections.get(heading, ""), section_label, id_prefix))
    chunks.extend(split_skill_chunks(sections.get("Skills", "")))

    return metadata, contact, chunks


@dataclass(frozen=True)
class RetrievalMatch:
    """One chunk's TF-IDF match against a job description.

    chunk: the matched Chunk, unmodified - retrieval never edits chunk
        text or splits/merges records.
    score: cosine similarity between the job text's and this chunk's
        L2-normalized TF-IDF vectors (both built from a vocabulary fit
        only on the profile's chunks). This is a measure of lexical
        overlap only - not a confidence score, and not proof the
        candidate is qualified for anything the job text mentions.
    term_contributions: every shared word/two-word phrase mapped to its
        contribution to `score`, computed as
        job_weight[term] * chunk_weight[term]. Summing every value in
        this dict reproduces `score` exactly (within floating-point
        precision), because cosine similarity of two L2-normalized
        vectors is just the dot product of their shared, nonzero terms.
    """

    chunk: Chunk
    score: float
    term_contributions: Dict[str, float]


def retrieve_relevant_chunks(
    job_text: str,
    chunks: List[Chunk],
    top_k: int = 8,
) -> List[RetrievalMatch]:
    r"""Rank profile chunks against a job description using local TF-IDF.

    Purpose / place in the application
    ------------------------------------
    This is the pipeline's "local fetch" step: it measures lexical
    overlap between a job description and each profile chunk completely
    offline, with no network or API call. Its ranked output is what a
    later single OpenAI request would receive as candidate evidence to
    draft from - this function does not judge qualification, it only
    scores word/phrase overlap.

    How it scores
    --------------
    - A TfidfVectorizer is fit ONLY on `chunks` (never on `job_text`), so
      the vocabulary and IDF weights come solely from the candidate's own
      profile; `job_text` is then transformed through that fitted
      vocabulary, so job-only terms are simply ignored rather than
      distorting the comparison.
    - Both unigrams and bigrams are indexed (ngram_range=(1, 2)), so
      two-word phrases such as "machine learning" count alongside single
      words.
    - Text is lowercased, and the token pattern is widened to
      r"(?u)\b\w+\b" so single-letter tool names (e.g. "R") survive
      tokenization - scikit-learn's default pattern requires 2+
      characters per token and would silently drop them.
    - Vectors are L2-normalized (norm="l2"), so the dot product of the
      job vector and a chunk vector already IS their cosine similarity;
      that dot product is computed directly (instead of via a separate
      similarity call) so the per-term contributions below are
      guaranteed to sum to the same score.

    Inputs
    ------
    job_text: raw job-description text, any length, may be blank.
    chunks: the chunks to search - normally load_profile()'s chunk list,
        which is already contact-free. This function does not itself
        inspect chunk text for contact information: it trusts whatever
        `chunks` it is given and will happily score/return a chunk that
        contains contact details if one is passed in directly rather
        than via load_profile().
    top_k: maximum number of matches to return (default 8). Must be a
        positive int (bool is rejected even though bool is technically
        an int subclass).

    Output
    ------
    A list of RetrievalMatch, best match first, truncated to `top_k`.
    Only chunks with a strictly positive score are included. Ties
    (equal score) are broken by chunk_id so ordering is deterministic
    across runs.

    Edge cases
    ----------
    - `top_k` that is not a positive int (including 0, negative values,
      non-integers, and bool) raises ValueError immediately - before
      job_text/chunks are even inspected, so an invalid top_k is never
      silently ignored just because the other inputs are empty.
    - Blank/whitespace-only `job_text` -> returns [].
    - Empty `chunks` -> returns [].
    - Every chunk has no usable tokens (e.g., punctuation-only text), so
      the fitted vocabulary is empty -> returns []. Only that specific
      scikit-learn "empty vocabulary" ValueError is caught; any other
      error from fitting is left to propagate.
    - `job_text` shares no vocabulary at all with any chunk (for example,
      a synonym-only rewrite like "attorney" for a chunk that only says
      "lawyer") -> every score is 0, so [] is returned; this reflects a
      real limitation of lexical TF-IDF matching, not a bug.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k!r}")

    if not chunks:
        return []
    if not job_text or not job_text.strip():
        return []

    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        norm="l2",
    )

    corpus = [chunk.text for chunk in chunks]
    try:
        chunk_matrix = vectorizer.fit_transform(corpus)
    except ValueError as exc:
        if "empty vocabulary" in str(exc).lower():
            # None of the chunks had any token matching token_pattern.
            return []
        raise
    job_vector = vectorizer.transform([job_text])

    if job_vector.nnz == 0:
        # job_text has no vocabulary overlap with the profile at all.
        return []

    feature_names = vectorizer.get_feature_names_out()
    job_weights = dict(zip(job_vector.indices, job_vector.data))

    # Both vectors are L2-normalized, so this dot product is the cosine
    # similarity directly - the same quantity the contributions below sum to.
    scores = (job_vector @ chunk_matrix.T).toarray().ravel()

    matches: List[RetrievalMatch] = []
    for row_index, chunk in enumerate(chunks):
        score = float(scores[row_index])
        if score <= 0:
            continue

        chunk_row = chunk_matrix.getrow(row_index)
        contributions: Dict[str, float] = {}
        for col_index, chunk_weight in zip(chunk_row.indices, chunk_row.data):
            job_weight = job_weights.get(col_index)
            if job_weight is None:
                continue
            contributions[feature_names[col_index]] = job_weight * chunk_weight

        contributions = dict(sorted(contributions.items(), key=lambda kv: (-kv[1], kv[0])))
        matches.append(RetrievalMatch(chunk=chunk, score=score, term_contributions=contributions))

    matches.sort(key=lambda match: (-match.score, match.chunk.chunk_id))
    return matches[:top_k]


def format_shared_terms(term_contributions: Dict[str, float], limit: int = 5) -> str:
    """Render the highest-contributing shared terms for display only.

    Purpose: lets CLI/demo output show *why* a chunk matched without
    dumping every shared n-gram; it never feeds back into scoring.
    Terms are ranked by contribution (ties broken by term name) and only
    the top `limit` are shown, with a trailing note naming how many more
    were left out of the display.

    Returns e.g. "python (0.182), sql (0.091) (+6 more shared terms not
    shown)", or "(no shared terms)" if the dict is empty.
    """
    if not term_contributions:
        return "(no shared terms)"

    ranked = sorted(term_contributions.items(), key=lambda kv: (-kv[1], kv[0]))
    shown = ranked[:limit]
    remaining = len(ranked) - len(shown)

    rendered = ", ".join(f"{term} ({weight:.3f})" for term, weight in shown)
    if remaining > 0:
        noun = "term" if remaining == 1 else "terms"
        rendered += f" (+{remaining} more shared {noun} not shown)"
    return rendered


def _print_summary(chunks: List[Chunk]) -> None:
    """CLI reporting only: chunk ids, titles, and per-section counts.
    Deliberately never prints chunk text or contact fields."""
    for chunk in chunks:
        print(f"[{chunk.chunk_id}] ({chunk.section}) {chunk.title}")

    counts: Dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.section] = counts.get(chunk.section, 0) + 1

    print("\nCounts by section:")
    for section, count in counts.items():
        print(f"  {section}: {count}")
    print(f"\nTotal chunks: {len(chunks)}")


if __name__ == "__main__":
    _metadata, _contact, loaded_chunks = load_profile()
    _print_summary(loaded_chunks)
