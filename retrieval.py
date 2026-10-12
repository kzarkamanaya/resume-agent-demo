"""retrieval.py

Purpose
-------
Heading-based loader for resume_profile.md. This is the first stage of the
resume-tailoring pipeline: it turns the single Markdown profile into a list
of small, retrievable "chunks" that a later TF-IDF step can search against
a job description, and that an LLM step can later draft from.

This module does not do retrieval (no TF-IDF/scoring here) and does not
call any API. It only parses and structures resume_profile.md, which it
reads but never modifies.

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
load_profile() is the function later pipeline stages (TF-IDF retrieval,
then the OpenAI drafting call) are expected to import and call. It returns
(metadata, contact, chunks): metadata/contact support header fields like
name/phone/email elsewhere in the app, while chunks is the list that
retrieval will index and search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
