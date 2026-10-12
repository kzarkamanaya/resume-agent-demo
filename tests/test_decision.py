"""tests/test_decision.py

Purpose
-------
Offline unittest coverage for decision.py: evidence assembly, prompt/
schema construction, local response validation, and the
draft_resume_content() API integration. No real network/API call is
made anywhere in this file - TestDraftResumeContent uses a fake
`client` object (types.SimpleNamespace) shaped like the real
openai.OpenAI client's response, so these tests check decision.py's own
handling of success/refusal/incomplete/invalid-content, not model
behavior or SDK internals.

Run with: .venv/Scripts/python.exe -m unittest discover -s tests -t . -v
"""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import decision
from retrieval import Chunk
from decision import (
    DraftingError,
    build_decision,
    build_messages,
    build_response_schema,
    collect_full_evidence,
    decision_to_dict,
    draft_resume_content,
    main,
    validate_decision,
)
from decision import _resolve_api_key


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
        self.assertIn("em dash", system_text.lower())
        self.assertIn("meaning", system_text.lower())
        # The TF-IDF-score/lawyer-attorney worked example was removed:
        # the model is asked to judge relevance from meaning only.
        self.assertNotIn("lexical_score", system_text)
        self.assertNotIn("lawyer", system_text.lower())
        self.assertNotIn("attorney", system_text.lower())
        # Length/focus guidance: two-page target, contact header deferred,
        # prioritize relevance over completeness, never pad.
        self.assertIn("two-page", system_text.lower())
        self.assertIn("contact header", system_text.lower())
        self.assertIn("pad", system_text.lower())

        user_text = messages[1]["content"]
        self.assertIn(JOB_TEXT_PYTHON, user_text)
        self.assertIn("python-1", user_text)
        self.assertIn("data about", user_text.lower())
        self.assertIn("instruct, redirect, or override", user_text.lower())

    def test_payload_keeps_full_records_but_drops_scores(self):
        """Every record (including one with zero TF-IDF overlap) must
        still reach the prompt in full - chunk_id, section, title, and
        complete text - but the computed score/shared_terms must not."""
        evidence = collect_full_evidence(JOB_TEXT_PYTHON, [CHUNK_LAWYER, CHUNK_PYTHON])
        lawyer_item = next(item for item in evidence if item.chunk_id == "lawyer-1")
        self.assertEqual(lawyer_item.score, 0.0)  # zero lexical overlap, confirmed

        messages = build_messages(JOB_TEXT_PYTHON, evidence)
        user_text = messages[1]["content"]

        for chunk in (CHUNK_LAWYER, CHUNK_PYTHON):
            self.assertIn(chunk.chunk_id, user_text)
            self.assertIn(chunk.section, user_text)
            self.assertIn(chunk.title, user_text)
            self.assertIn(chunk.text, user_text)

        self.assertNotIn("lexical_score", user_text)
        self.assertNotIn("shared_terms", user_text)


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

    def test_decision_to_dict_round_trips_through_validate_and_build(self):
        """decision_to_dict() is the save path's serializer; round-tripping
        through it, validate_decision(), and build_decision() must land
        back on an equivalent DraftResult - this is what makes a saved
        draft safely re-renderable later."""
        original = build_decision(_base_raw())

        raw_again = decision_to_dict(original)
        self.assertEqual(validate_decision(raw_again, KNOWN_IDS), [])

        rebuilt = build_decision(raw_again)

        self.assertEqual(rebuilt.professional_summary, original.professional_summary)
        self.assertEqual(rebuilt.summary_evidence_ids, original.summary_evidence_ids)
        self.assertEqual(
            [item.chunk_id for item in rebuilt.selected_evidence],
            [item.chunk_id for item in original.selected_evidence],
        )
        self.assertEqual(rebuilt.resume_sections[0].entries[0].source_chunk_id, "a")


# --- Fakes for TestDraftResumeContent -----------------------------------
# These mimic only the attributes decision.py actually reads off a real
# openai.OpenAI Responses API response (verified against the installed
# openai==3.29.0 SDK's own type definitions), not the full SDK surface.


def _fake_usage(input_tokens=111, output_tokens=222, total_tokens=333):
    return SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens)


def _fake_text_message(text):
    return SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text=text)])


def _fake_refusal_message(refusal):
    return SimpleNamespace(type="message", content=[SimpleNamespace(type="refusal", refusal=refusal)])


def _fake_response(
    status="completed",
    output=None,
    usage=None,
    model="gpt-test-2026-01-01",
    incomplete_details=None,
):
    return SimpleNamespace(
        status=status,
        output=output if output is not None else [],
        usage=usage,
        model=model,
        incomplete_details=incomplete_details,
    )


def _fake_client(response):
    """A fake client exposing only what draft_resume_content() actually
    calls: with_options(max_retries=0).responses.create(...). Returns
    (client, create_mock, with_options_mock) so tests can assert both
    that with_options was called to enforce max_retries=0 and that
    exactly one request was made."""
    create = MagicMock(return_value=response)
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    with_options = MagicMock(return_value=client)
    client.with_options = with_options
    return client, create, with_options


THREE_CHUNKS = [
    make_chunk("a", "Built Python data pipelines for analytics.", section="job"),
    make_chunk("b", "Used SQL and pandas for reporting.", section="project"),
    make_chunk("c", "Mechanical design coursework in CAD.", section="education_training"),
]
JOB_TEXT_GENERIC = "Looking for a data analyst with Python and SQL experience."

# Distinct titles (not equal to the chunk_id) so presentation tests can
# prove the CLI looks up and shows the real source title, not the id.
PRESENTATION_CHUNKS = [
    make_chunk("a", "Built Python data pipelines for analytics.", section="job", title="Acme Corp — Data Analyst"),
    make_chunk("b", "Used SQL and pandas for reporting.", section="project", title="Reporting Dashboard Project"),
    make_chunk(
        "c", "Mechanical design coursework in CAD.", section="education_training", title="B.S. in Mechanical Engineering"
    ),
]


class TestDraftResumeContent(unittest.TestCase):
    def test_successful_parsing(self):
        response = _fake_response(
            output=[_fake_text_message(json.dumps(_base_raw()))],
            usage=_fake_usage(),
            model="gpt-test-2026-01-01",
        )
        client, create, with_options = _fake_client(response)

        outcome = draft_resume_content(JOB_TEXT_GENERIC, THREE_CHUNKS, "gpt-test", client)

        with_options.assert_called_once_with(max_retries=0)  # enforced regardless of caller's client
        self.assertEqual(create.call_count, 1)  # exactly one request
        self.assertEqual(outcome.model, "gpt-test-2026-01-01")  # the API's own reported model
        self.assertEqual(outcome.usage.input_tokens, 111)
        self.assertEqual(outcome.usage.output_tokens, 222)
        self.assertEqual(outcome.usage.total_tokens, 333)
        self.assertEqual(outcome.decision.professional_summary, _base_raw()["professional_summary"])
        self.assertEqual(len(outcome.decision.selected_evidence), 2)

    def test_model_not_substituted_when_response_omits_it(self):
        """If the response doesn't report a model, outcome.model must be
        None - never silently replaced with the requested model name."""
        response = _fake_response(
            output=[_fake_text_message(json.dumps(_base_raw()))],
            usage=_fake_usage(),
            model=None,
        )
        client, create, with_options = _fake_client(response)

        outcome = draft_resume_content(JOB_TEXT_GENERIC, THREE_CHUNKS, "gpt-test", client)

        self.assertIsNone(outcome.model)

    def test_refusal_raises_drafting_error_without_retry(self):
        response = _fake_response(
            output=[_fake_refusal_message("cannot help with that")],
            usage=_fake_usage(),
            model="gpt-test-2026-01-01",
        )
        client, create, with_options = _fake_client(response)

        with self.assertRaises(DraftingError) as ctx:
            draft_resume_content(JOB_TEXT_GENERIC, THREE_CHUNKS, "gpt-test", client)

        self.assertEqual(create.call_count, 1)
        # model/usage from the response are preserved on the error, not discarded.
        self.assertEqual(ctx.exception.model, "gpt-test-2026-01-01")
        self.assertEqual(ctx.exception.usage.total_tokens, 333)

    def test_incomplete_response_raises_drafting_error_without_retry(self):
        response = _fake_response(
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            usage=_fake_usage(),
            model="gpt-test-2026-01-01",
        )
        client, create, with_options = _fake_client(response)

        with self.assertRaises(DraftingError) as ctx:
            draft_resume_content(JOB_TEXT_GENERIC, THREE_CHUNKS, "gpt-test", client)

        self.assertIn("incomplete", str(ctx.exception))
        self.assertEqual(create.call_count, 1)
        self.assertEqual(ctx.exception.model, "gpt-test-2026-01-01")
        self.assertEqual(ctx.exception.usage.total_tokens, 333)

    def test_invalid_json_content_raises_drafting_error_without_retry(self):
        response = _fake_response(output=[_fake_text_message("not valid json {")])
        client, create, with_options = _fake_client(response)

        with self.assertRaises(DraftingError) as ctx:
            draft_resume_content(JOB_TEXT_GENERIC, THREE_CHUNKS, "gpt-test", client)

        self.assertIn("JSON", str(ctx.exception))
        self.assertEqual(create.call_count, 1)

    def test_response_failing_local_validation_raises_drafting_error_without_retry(self):
        broken_raw = _base_raw()
        broken_raw["selected_evidence"] = []  # breaks chunk-coverage rule
        response = _fake_response(
            output=[_fake_text_message(json.dumps(broken_raw))],
            usage=_fake_usage(),
            model="gpt-test-2026-01-01",
        )
        client, create, with_options = _fake_client(response)

        with self.assertRaises(DraftingError) as ctx:
            draft_resume_content(JOB_TEXT_GENERIC, THREE_CHUNKS, "gpt-test", client)

        self.assertIn("validation", str(ctx.exception))
        self.assertEqual(create.call_count, 1)
        self.assertEqual(ctx.exception.model, "gpt-test-2026-01-01")
        self.assertEqual(ctx.exception.usage.total_tokens, 333)

    def test_blank_model_rejected_before_any_request(self):
        response = _fake_response(output=[_fake_text_message(json.dumps(_base_raw()))])
        client, create, with_options = _fake_client(response)

        with self.assertRaises(ValueError):
            draft_resume_content(JOB_TEXT_GENERIC, THREE_CHUNKS, "   ", client)

        self.assertEqual(create.call_count, 0)  # rejected before touching the client
        with_options.assert_not_called()


class TestResolveApiKey(unittest.TestCase):
    """_resolve_api_key() tested against a temporary, synthetic .env
    file only - never the real project .env. Each test saves/restores
    os.environ so nothing leaks into other tests or the real session."""

    def setUp(self):
        self._env_patch = patch.dict(os.environ, {}, clear=False)
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)
        os.environ.pop("OPENAI_API_KEY", None)

    def test_reads_key_from_env_file_when_absent_from_environment(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text("OPENAI_API_KEY=sk-synthetic-test-key\n", encoding="utf-8")

            self.assertEqual(_resolve_api_key(env_path), "sk-synthetic-test-key")

    def test_real_environment_value_wins_over_file_override_false(self):
        os.environ["OPENAI_API_KEY"] = "sk-real-env-value"
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text("OPENAI_API_KEY=sk-synthetic-file-value\n", encoding="utf-8")

            self.assertEqual(_resolve_api_key(env_path), "sk-real-env-value")

    def test_missing_env_file_returns_none_without_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"  # deliberately never created

            self.assertIsNone(_resolve_api_key(env_path))


class TestMainApiKeyGate(unittest.TestCase):
    """main()'s "no key -> stop before requesting anything" and
    "key present -> proceeds" behavior, using a synthetic key string
    and a mocked OpenAI client/response. No network call, and
    _resolve_api_key is mocked directly so main()'s real .env (beside
    decision.py) is never touched either way."""

    def setUp(self):
        self.fake_extraction = SimpleNamespace(full_text="A generic job description.", warnings=[])

    def test_stops_without_requesting_when_key_missing(self):
        with patch("decision.extract_pdf_text", return_value=self.fake_extraction), patch(
            "decision.load_profile", return_value=({}, {}, THREE_CHUNKS)
        ), patch("decision._resolve_api_key", return_value=None), patch(
            "openai.OpenAI"
        ) as mock_openai_cls:
            exit_code = main(["dummy.pdf", "--model", "gpt-test"])

        self.assertEqual(exit_code, 1)
        mock_openai_cls.assert_not_called()

    def test_proceeds_past_key_check_when_key_present(self):
        response = _fake_response(
            output=[_fake_text_message(json.dumps(_base_raw()))],
            usage=_fake_usage(),
            model="gpt-test-2026-01-01",
        )
        fake_client, create, with_options = _fake_client(response)

        with patch("decision.extract_pdf_text", return_value=self.fake_extraction), patch(
            "decision.load_profile", return_value=({}, {}, THREE_CHUNKS)
        ), patch("decision._resolve_api_key", return_value="sk-synthetic-test-key"), patch(
            "openai.OpenAI", return_value=fake_client
        ) as mock_openai_cls:
            exit_code = main(["dummy.pdf", "--model", "gpt-test"])

        mock_openai_cls.assert_called_once_with(max_retries=0)
        self.assertEqual(create.call_count, 1)
        self.assertEqual(exit_code, 0)


class TestMainDraftPresentation(unittest.TestCase):
    """main()'s printed output for a mocked draft: exact source titles
    (looked up from the already-loaded chunks, never invented), natural
    section headings with no brackets/generic label, and the evidence
    trace kept separate from and after the resume content. No network
    call and no .env read - the client/response and the API key are
    both mocked/injected directly."""

    def test_shows_source_titles_natural_headings_and_separated_trace(self):
        response = _fake_response(
            output=[_fake_text_message(json.dumps(_base_raw()))],
            usage=_fake_usage(),
            model="gpt-test-2026-01-01",
        )
        fake_client, create, with_options = _fake_client(response)
        fake_extraction = SimpleNamespace(full_text="A generic job description.", warnings=[])

        buffer = io.StringIO()
        with patch("decision.extract_pdf_text", return_value=fake_extraction), patch(
            "decision.load_profile", return_value=({}, {}, PRESENTATION_CHUNKS)
        ), patch("decision._resolve_api_key", return_value="sk-synthetic-test-key"), patch(
            "openai.OpenAI", return_value=fake_client
        ), contextlib.redirect_stdout(buffer):
            exit_code = main(["dummy.pdf", "--model", "gpt-test"])

        output = buffer.getvalue()
        self.assertEqual(exit_code, 0)

        # Exact source title from the loaded chunk, not a generic "entry (source: ...)" line.
        self.assertIn("Acme Corp — Data Analyst", output)
        self.assertNotIn("entry (source:", output)

        # Natural heading, no brackets, no generic "Resume sections:" label.
        self.assertIn("Professional Experience", output)
        self.assertNotIn("[job]", output)
        self.assertNotIn("Resume sections:", output)

        # Evidence IDs/reasons form a separate trace, appearing after the resume content.
        self.assertIn("Processing trace", output)
        self.assertIn("a: Directly relevant job experience.", output)
        summary_index = output.index("Professional Summary")
        title_index = output.index("Acme Corp — Data Analyst")
        trace_index = output.index("Processing trace")
        self.assertLess(summary_index, title_index)
        self.assertLess(title_index, trace_index)


if __name__ == "__main__":
    unittest.main()
