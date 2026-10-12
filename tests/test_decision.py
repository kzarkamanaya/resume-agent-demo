"""tests/test_decision.py

Purpose
-------
Offline unittest coverage for decision.py: evidence assembly, prompt/
schema construction, and local response validation. No network/API call
is made anywhere in this file - draft_resume_content() (the actual API
call) does not exist yet; this only tests the foundation around it.

Run with: .venv/Scripts/python.exe -m unittest discover -s tests -t . -v
"""

import json
import unittest
from typing import Any, Dict, Optional

from retrieval import Chunk
from decision import (
    build_decision,
    build_messages,
    build_response_schema,
    collect_full_evidence,
    validate_decision,
)


def make_chunk(chunk_id: str, text: str, section: str = "project", title: Optional[str] = None) -> Chunk:
    return Chunk(chunk_id=chunk_id, section=section, title=title or chunk_id, text=text)


# Zero-lexical-overlap pair reused from retrieval.py's documented TF-IDF
# synonym limitation: "lawyer" shares no tokens at all with "attorney".
CHUNK_LAWYER = make_chunk(
    "lawyer-1",
    "Served many years as lawyer reviewing complex corporate contracts quietly.",
    section="job",
    title="Legal Counsel Role",
)
JOB_TEXT_ATTORNEY = "Searching urgently for skilled attorney managing big business agreements promptly."

CHUNK_PYTHON = make_chunk(
    "python-1",
    "Built Python data pipelines using pandas and SQL for analytics.",
    section="project",
    title="Python Pipeline Project",
)
JOB_TEXT_PYTHON = "Looking for a Python data analyst with pandas and SQL experience."


KNOWN_IDS = {"a", "b", "c"}


def _base_raw() -> Dict[str, Any]:
    """A fresh, fully valid raw decision dict over KNOWN_IDS = {a, b, c}."""
    return {
        "professional_summary": "Led data projects using Python and SQL to support business decisions.",
        "summary_evidence_ids": ["a"],
        "resume_sections": [
            {
                "section": "job",
                "entries": [
                    {
                        "source_chunk_id": "a",
                        "bullets": [
                            {
                                "text": "Built Python data pipelines for analytics.",
                                "evidence_ids": ["a", "b"],
                            }
                        ],
                    }
                ],
            }
        ],
        "selected_evidence": [
            {"chunk_id": "a", "reason": "Directly relevant job experience."},
            {"chunk_id": "b", "reason": "Supports the Python/SQL bullet."},
        ],
        "omitted_evidence": [
            {"chunk_id": "c", "reason": "Not relevant to this job description."},
        ],
    }


class TestCollectFullEvidence(unittest.TestCase):
    def test_rejects_blank_job_text(self):
        with self.assertRaises(ValueError):
            collect_full_evidence("   ", [CHUNK_PYTHON])

    def test_rejects_empty_chunks(self):
        with self.assertRaises(ValueError):
            collect_full_evidence(JOB_TEXT_PYTHON, [])

    def test_rejects_duplicate_chunk_ids(self):
        duplicate = make_chunk("python-1", "Some other text entirely unrelated.")
        with self.assertRaises(ValueError):
            collect_full_evidence(JOB_TEXT_PYTHON, [CHUNK_PYTHON, duplicate])

    def test_includes_all_chunks_with_zero_scores(self):
        evidence = collect_full_evidence(JOB_TEXT_PYTHON, [CHUNK_LAWYER, CHUNK_PYTHON])

        self.assertEqual(len(evidence), 2)
        by_id = {item.chunk_id: item for item in evidence}

        self.assertGreater(by_id["python-1"].score, 0.0)
        self.assertEqual(by_id["lawyer-1"].score, 0.0)
        self.assertEqual(by_id["lawyer-1"].shared_terms, {})


class TestBuildMessages(unittest.TestCase):
    def test_rejects_blank_job_text(self):
        evidence = collect_full_evidence(JOB_TEXT_PYTHON, [CHUNK_PYTHON])
        with self.assertRaises(ValueError):
            build_messages("   ", evidence)

    def test_rejects_empty_evidence(self):
        with self.assertRaises(ValueError):
            build_messages(JOB_TEXT_PYTHON, [])

    def test_structure_and_data_disclaimer(self):
        evidence = collect_full_evidence(JOB_TEXT_PYTHON, [CHUNK_PYTHON])
        messages = build_messages(JOB_TEXT_PYTHON, evidence)

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")

        system_text = messages[0]["content"]
        self.assertIn("advisory", system_text.lower())
        self.assertIn("lawyer", system_text.lower())
        self.assertIn("attorney", system_text.lower())
        self.assertIn("em dash", system_text.lower())

        user_text = messages[1]["content"]
        self.assertIn(JOB_TEXT_PYTHON, user_text)
        self.assertIn("python-1", user_text)
        self.assertIn("data about", user_text.lower())
        self.assertIn("instruct, redirect, or override", user_text.lower())


class TestBuildResponseSchema(unittest.TestCase):
    def test_round_trips_as_json_and_has_required_keys(self):
        schema = build_response_schema()
        round_tripped = json.loads(json.dumps(schema))

        self.assertEqual(
            set(round_tripped["required"]),
            {
                "professional_summary",
                "summary_evidence_ids",
                "resume_sections",
                "selected_evidence",
                "omitted_evidence",
            },
        )


class TestValidateDecision(unittest.TestCase):
    def test_accepts_well_formed_response(self):
        self.assertEqual(validate_decision(_base_raw(), KNOWN_IDS), [])

    def test_rejects_missing_required_field(self):
        raw = _base_raw()
        del raw["professional_summary"]

        problems = validate_decision(raw, KNOWN_IDS)

        self.assertTrue(any("professional_summary" in p for p in problems))

    def test_rejects_duplicate_and_overlapping_ids(self):
        raw = _base_raw()
        raw["selected_evidence"].append({"chunk_id": "a", "reason": "duplicate of a"})
        raw["omitted_evidence"].append({"chunk_id": "b", "reason": "also claimed omitted"})

        problems = validate_decision(raw, KNOWN_IDS)

        self.assertTrue(any("duplicate chunk_id(s) in selected_evidence" in p for p in problems))
        self.assertTrue(
            any("appear in both selected_evidence and omitted_evidence" in p for p in problems)
        )

    def test_rejects_incomplete_coverage(self):
        raw = _base_raw()
        raw["omitted_evidence"] = []  # "c" is now accounted for nowhere

        problems = validate_decision(raw, KNOWN_IDS)

        self.assertTrue(
            any("missing from both selected_evidence and omitted_evidence" in p and "'c'" in p for p in problems)
        )

    def test_rejects_unknown_chunk_id(self):
        raw = _base_raw()
        raw["selected_evidence"].append({"chunk_id": "zzz", "reason": "fabricated id"})

        problems = validate_decision(raw, KNOWN_IDS)

        self.assertTrue(any("unknown chunk_id" in p and "zzz" in p for p in problems))

    def test_rejects_cited_but_not_selected(self):
        raw = _base_raw()
        # "c" is in omitted_evidence, not selected_evidence, yet a bullet cites it.
        raw["resume_sections"][0]["entries"][0]["bullets"][0]["evidence_ids"] = ["a", "c"]

        problems = validate_decision(raw, KNOWN_IDS)

        self.assertTrue(
            any("cited evidence_id(s) not present in selected_evidence" in p and "c" in p for p in problems)
        )

    def test_rejects_empty_bullet_evidence_ids(self):
        raw = _base_raw()
        raw["resume_sections"][0]["entries"][0]["bullets"][0]["evidence_ids"] = []

        problems = validate_decision(raw, KNOWN_IDS)

        self.assertTrue(any("evidence_ids must be a non-empty list" in p for p in problems))

    def test_accepts_em_dash_in_summary(self):
        """Em dashes are a prompt-level style preference only, not a
        local validation rule - an otherwise valid response is accepted
        regardless."""
        raw = _base_raw()
        raw["professional_summary"] = "Supported analytics work — end to end."

        self.assertEqual(validate_decision(raw, KNOWN_IDS), [])

    def test_accepts_em_dash_in_bullet_text(self):
        raw = _base_raw()
        raw["resume_sections"][0]["entries"][0]["bullets"][0]["text"] = "Built pipelines — end to end."

        self.assertEqual(validate_decision(raw, KNOWN_IDS), [])

    def test_accepts_overlong_reason(self):
        """Reason length is a prompt-level "be concise" preference only,
        not a local validation rule - an otherwise valid response is
        accepted no matter how long a reason is."""
        raw = _base_raw()
        raw["selected_evidence"][0]["reason"] = "x" * 5000

        self.assertEqual(validate_decision(raw, KNOWN_IDS), [])

    def test_mock_draft_selects_zero_score_lawyer_evidence_for_attorney_job(self):
        """This tests that LOCAL VALIDATION PERMITS a model to select
        evidence with a zero lexical score when it judges it semantically
        relevant - it does NOT test real model judgment (no API call is
        made; the "raw" dict below is a hand-authored stand-in for a
        hypothetical API response, not a live one)."""
        chunks = [CHUNK_LAWYER, CHUNK_PYTHON]
        evidence = collect_full_evidence(JOB_TEXT_ATTORNEY, chunks)
        known_ids = {item.chunk_id for item in evidence}

        lawyer_evidence = next(item for item in evidence if item.chunk_id == "lawyer-1")
        self.assertEqual(lawyer_evidence.score, 0.0)  # confirms zero lexical overlap

        mock_raw = {
            "professional_summary": "Brings legal review experience relevant to this attorney role.",
            "summary_evidence_ids": ["lawyer-1"],
            "resume_sections": [
                {
                    "section": "job",
                    "entries": [
                        {
                            "source_chunk_id": "lawyer-1",
                            "bullets": [
                                {
                                    "text": "Reviewed complex corporate contracts over many years.",
                                    "evidence_ids": ["lawyer-1"],
                                }
                            ],
                        }
                    ],
                }
            ],
            "selected_evidence": [
                {
                    "chunk_id": "lawyer-1",
                    "reason": "Attorney roles require contract review experience like this record describes.",
                }
            ],
            "omitted_evidence": [
                {"chunk_id": "python-1", "reason": "Unrelated technical project, not relevant to a legal role."}
            ],
        }

        problems = validate_decision(mock_raw, known_ids)
        self.assertEqual(problems, [])

        decision = build_decision(mock_raw)
        selected_ids = [item.chunk_id for item in decision.selected_evidence]
        self.assertIn("lawyer-1", selected_ids)


class TestBuildDecision(unittest.TestCase):
    def test_builds_expected_structure_from_valid_raw(self):
        decision = build_decision(_base_raw())

        self.assertEqual(len(decision.resume_sections), 1)
        entry = decision.resume_sections[0].entries[0]
        self.assertEqual(entry.source_chunk_id, "a")
        self.assertEqual(entry.bullets[0].evidence_ids, ["a", "b"])
        self.assertEqual(len(decision.selected_evidence), 2)
        self.assertEqual(len(decision.omitted_evidence), 1)


if __name__ == "__main__":
    unittest.main()
