"""tests/test_pipeline.py

Purpose
-------
Offline unittest coverage for pipeline.run_pipeline(): orchestration,
overlap prevention, cost-estimate labeling, and download gating. No
real network/API call and no real .env read anywhere in this file -
extract_pdf_text, load_profile, _resolve_api_key, OpenAI, and
render_and_verify are all patched with synthetic fakes.

Run with: .venv/Scripts/python.exe -m unittest discover -s tests -t . -v
"""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pipeline
from decision import DraftingError
from document import RenderResult
from retrieval import Chunk


def make_chunk(chunk_id: str, text: str, section: str = "job", title: str = None) -> Chunk:
    return Chunk(chunk_id=chunk_id, section=section, title=title or chunk_id, text=text)


CHUNKS = [
    make_chunk("job-1", "Built Python data pipelines for analytics.", title="Acme Corp — Analyst"),
    make_chunk("skill-1", "Python, SQL.", section="skill_category", title="Programming languages"),
]
CONTACT = {"Name": "Jordan Example"}
JOB_TEXT = "Looking for a Python data analyst."

GOOD_RAW_DRAFT = {
    "professional_summary": "Experienced analyst with strong Python skills.",
    "summary_evidence_ids": ["job-1"],
    "resume_sections": [
        {
            "section": "job",
            "entries": [
                {
                    "source_chunk_id": "job-1",
                    "bullets": [{"text": "Built Python data pipelines for analytics.", "evidence_ids": ["job-1"]}],
                }
            ],
        }
    ],
    "selected_evidence": [{"chunk_id": "job-1", "reason": "Directly relevant."}],
    "omitted_evidence": [{"chunk_id": "skill-1", "reason": "Not needed for this job."}],
}


def _fake_usage(input_tokens=100, output_tokens=50, total_tokens=150):
    return SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens)


def _fake_text_message(text):
    return SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text=text)])


def _fake_refusal_message(refusal):
    return SimpleNamespace(type="message", content=[SimpleNamespace(type="refusal", refusal=refusal)])


def _fake_response(status="completed", output=None, usage=None, model="gpt-6.1-sol-2026-10-01"):
    return SimpleNamespace(status=status, output=output or [], usage=usage, model=model, incomplete_details=None)


def _fake_client(response):
    create = MagicMock(return_value=response)
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    client.with_options = MagicMock(return_value=client)
    return client, create


class TestEstimateCostUsd(unittest.TestCase):
    def test_exact_model_match_gives_a_number(self):
        note = pipeline.estimate_cost_usd("gpt-6.1-sol", _fake_usage(100, 50, 150))
        self.assertIn("ESTIMATE: $0.0007", note)

    def test_dated_snapshot_matches_verified_base_name(self):
        note = pipeline.estimate_cost_usd("gpt-6.1-sol-2026-10-01", _fake_usage(100, 50, 150))
        self.assertIn("ESTIMATE: $0.0007", note)
        self.assertIn("gpt-6.1-sol-2026-10-01", note)  # the actual reported snapshot is named too

    def test_unrelated_model_name_is_not_matched(self):
        note = pipeline.estimate_cost_usd("gpt-6.1-solar", _fake_usage(100, 50, 150))
        self.assertIn("unavailable", note)

    def test_model_variant_with_non_date_suffix_is_not_matched(self):
        """A plain startswith() check would wrongly treat "gpt-6.1-sol-mini"
        as the verified "gpt-6.1-sol" since it starts with that prefix -
        only a genuine "-YYYY-MM-DD" dated-snapshot suffix should match."""
        for variant in ("gpt-6.1-sol-mini", "gpt-6.1-sol-turbo", "gpt-6.1-sol-preview"):
            note = pipeline.estimate_cost_usd(variant, _fake_usage(100, 50, 150))
            self.assertIn("unavailable", note, msg=f"{variant!r} should not match verified pricing")

    def test_unknown_model_never_invents_a_price(self):
        note = pipeline.estimate_cost_usd("totally-unverified-model", _fake_usage())
        self.assertIn("Cost estimate unavailable", note)
        self.assertIn("totally-unverified-model", note)


class TestRunPipelineSuccess(unittest.TestCase):
    def setUp(self):
        import tempfile

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_dir = Path(tmp.name)

    def test_full_run_reports_everything_and_enables_download(self):
        fake_extraction = SimpleNamespace(full_text=JOB_TEXT, warnings=["page 2 had no extractable text"])
        response = _fake_response(output=[_fake_text_message(json.dumps(GOOD_RAW_DRAFT))], usage=_fake_usage())
        fake_client, create = _fake_client(response)

        with patch("pipeline.extract_pdf_text", return_value=fake_extraction), patch(
            "pipeline.load_profile", return_value=({}, CONTACT, CHUNKS)
        ), patch("pipeline._resolve_api_key", return_value="sk-synthetic-test-key"), patch(
            "pipeline.OpenAI", return_value=fake_client
        ), patch(
            "pipeline.render_and_verify",
            return_value=RenderResult(
                docx_path=self.tmp_dir / "run" / "resume.docx",
                pdf_path=None,
                page_count=2,
                preset_name="comfortable",
                status="verified_two_pages",
                message="Verified exactly 2 pages.",
                attempts=[("comfortable", 2)],
            ),
        ), patch("pipeline.OUTPUTS_DIR", self.tmp_dir):
            results = list(pipeline.run_pipeline("job.pdf", "gpt-6.1-sol"))

        self.assertEqual(create.call_count, 1)  # exactly one request
        final_text, final_download = results[-1]

        self.assertIn("page 2 had no extractable text", final_text)
        self.assertIn("Request count: 1", final_text)
        self.assertIn("input: 100, output: 50, total: 150", final_text)
        self.assertIn("ESTIMATE: $0.0007", final_text)  # (100*$2 + 50*$10) / 1e6 tokens
        self.assertIn("Summary evidence IDs: ['job-1']", final_text)
        self.assertIn("job-1: Directly relevant.", final_text)
        self.assertIn("skill-1: Not needed for this job.", final_text)
        self.assertIn("preset 'comfortable' -> 2 page(s)", final_text)
        self.assertIsNotNone(final_download)

        # No raw source dump: chunk body text must not appear in the trace.
        self.assertNotIn("Built Python data pipelines for analytics.", final_text)

    def test_unknown_model_reports_estimate_unavailable_not_a_number(self):
        fake_extraction = SimpleNamespace(full_text=JOB_TEXT, warnings=[])
        response = _fake_response(
            output=[_fake_text_message(json.dumps(GOOD_RAW_DRAFT))], usage=_fake_usage(), model="some-other-model"
        )
        fake_client, create = _fake_client(response)

        with patch("pipeline.extract_pdf_text", return_value=fake_extraction), patch(
            "pipeline.load_profile", return_value=({}, CONTACT, CHUNKS)
        ), patch("pipeline._resolve_api_key", return_value="sk-synthetic-test-key"), patch(
            "pipeline.OpenAI", return_value=fake_client
        ), patch(
            "pipeline.render_and_verify",
            return_value=RenderResult(
                docx_path=self.tmp_dir / "resume.docx", pdf_path=None, page_count=2, preset_name="comfortable",
                status="verified_two_pages", message="ok", attempts=[("comfortable", 2)],
            ),
        ), patch("pipeline.OUTPUTS_DIR", self.tmp_dir):
            results = list(pipeline.run_pipeline("job.pdf", "some-other-model"))

        final_text, _ = results[-1]
        self.assertIn("Cost estimate unavailable", final_text)
        self.assertIn("some-other-model", final_text)

    def test_render_failure_clears_download(self):
        """A genuine rendering failure (e.g. LibreOffice conversion
        error) must still block the download."""
        fake_extraction = SimpleNamespace(full_text=JOB_TEXT, warnings=[])
        response = _fake_response(output=[_fake_text_message(json.dumps(GOOD_RAW_DRAFT))], usage=_fake_usage())
        fake_client, create = _fake_client(response)

        with patch("pipeline.extract_pdf_text", return_value=fake_extraction), patch(
            "pipeline.load_profile", return_value=({}, CONTACT, CHUNKS)
        ), patch("pipeline._resolve_api_key", return_value="sk-synthetic-test-key"), patch(
            "pipeline.OpenAI", return_value=fake_client
        ), patch(
            "pipeline.render_and_verify",
            return_value=RenderResult(
                docx_path=self.tmp_dir / "resume.docx", pdf_path=None, page_count=None, preset_name="comfortable",
                status="failed_conversion_error", message="LibreOffice conversion failed.", attempts=[("comfortable", None)],
            ),
        ), patch("pipeline.OUTPUTS_DIR", self.tmp_dir):
            results = list(pipeline.run_pipeline("job.pdf", "gpt-6.1-sol"))

        final_text, final_download = results[-1]
        self.assertIsNone(final_download)
        self.assertIn("not enabled", final_text.lower())

    def test_target_not_met_still_offers_download(self):
        """Two pages is a target, not a requirement: a successful
        render that lands on 1 or 3+ pages must still offer a
        download, clearly labeled as not having met the target."""
        fake_extraction = SimpleNamespace(full_text=JOB_TEXT, warnings=[])
        response = _fake_response(output=[_fake_text_message(json.dumps(GOOD_RAW_DRAFT))], usage=_fake_usage())
        fake_client, create = _fake_client(response)

        with patch("pipeline.extract_pdf_text", return_value=fake_extraction), patch(
            "pipeline.load_profile", return_value=({}, CONTACT, CHUNKS)
        ), patch("pipeline._resolve_api_key", return_value="sk-synthetic-test-key"), patch(
            "pipeline.OpenAI", return_value=fake_client
        ), patch(
            "pipeline.render_and_verify",
            return_value=RenderResult(
                docx_path=self.tmp_dir / "resume.docx", pdf_path=None, page_count=3, preset_name="tight",
                status="generated_target_not_met",
                message="Generated 3 page(s); two-page target not met.",
                attempts=[("comfortable", 3), ("compact", 3), ("tight", 3)],
            ),
        ), patch("pipeline.OUTPUTS_DIR", self.tmp_dir):
            results = list(pipeline.run_pipeline("job.pdf", "gpt-6.1-sol"))

        final_text, final_download = results[-1]
        self.assertIsNotNone(final_download)
        self.assertIn("Generated 3 page(s); two-page target not met.", final_text)
        self.assertIn("download enabled", final_text.lower())

    def test_blank_extracted_text_stops_before_any_api_call(self):
        fake_extraction = SimpleNamespace(full_text="   ", warnings=["no extractable text"])

        with patch("pipeline.extract_pdf_text", return_value=fake_extraction), patch(
            "pipeline.OpenAI"
        ) as mock_openai_cls:
            results = list(pipeline.run_pipeline("job.pdf", "gpt-6.1-sol"))

        mock_openai_cls.assert_not_called()
        final_text, final_download = results[-1]
        self.assertIn("stopping before any API request", final_text)
        self.assertIsNone(final_download)

    def test_model_refusal_reports_request_count_and_cost_estimate(self):
        fake_extraction = SimpleNamespace(full_text=JOB_TEXT, warnings=[])
        response = _fake_response(output=[_fake_refusal_message("cannot help with that")], usage=_fake_usage())
        fake_client, create = _fake_client(response)

        with patch("pipeline.extract_pdf_text", return_value=fake_extraction), patch(
            "pipeline.load_profile", return_value=({}, CONTACT, CHUNKS)
        ), patch("pipeline._resolve_api_key", return_value="sk-synthetic-test-key"), patch(
            "pipeline.OpenAI", return_value=fake_client
        ):
            results = list(pipeline.run_pipeline("job.pdf", "gpt-6.1-sol"))

        self.assertEqual(create.call_count, 1)
        final_text = "\n".join(text for text, _ in results)
        self.assertIn("Drafting failed", final_text)
        self.assertIn("Request count: 1.", final_text)
        self.assertIn("input: 100, output: 50, total: 150", final_text)
        self.assertIn("ESTIMATE: $0.0007", final_text)  # model/usage were both available
        self.assertIsNone(results[-1][1])

    def test_pdf_extraction_failure_reports_clearly_and_releases_lock(self):
        """FileNotFoundError/ValueError from extract_pdf_text (missing
        file, corrupt/malformed PDF) must be caught, reported in plain
        terms, clear the download, and release the run lock - not
        propagate as a raw traceback."""
        with patch("pipeline.extract_pdf_text", side_effect=FileNotFoundError("PDF not found: missing.pdf")), patch(
            "pipeline.OpenAI"
        ) as mock_openai_cls:
            results = list(pipeline.run_pipeline("missing.pdf", "gpt-6.1-sol"))

        mock_openai_cls.assert_not_called()
        final_text, final_download = results[-1]
        self.assertIn("Could not read the uploaded file", final_text)
        self.assertIsNone(final_download)
        self.assertFalse(pipeline._RUN_LOCK.locked())  # released, so a next run can proceed

        with patch("pipeline.extract_pdf_text", side_effect=ValueError("not a valid PDF: bad.pdf (boom)")), patch(
            "pipeline.OpenAI"
        ) as mock_openai_cls2:
            results2 = list(pipeline.run_pipeline("bad.pdf", "gpt-6.1-sol"))

        mock_openai_cls2.assert_not_called()
        final_text2, final_download2 = results2[-1]
        self.assertIn("Could not extract text from the PDF", final_text2)
        self.assertIsNone(final_download2)
        self.assertFalse(pipeline._RUN_LOCK.locked())

    def test_real_evidence_ranking_and_shared_terms_formatting(self):
        """Uses the real collect_full_evidence() and real
        format_shared_terms() against synthetic chunks - only the API
        client and renderer are mocked."""
        fake_extraction = SimpleNamespace(full_text=JOB_TEXT, warnings=[])
        response = _fake_response(output=[_fake_text_message(json.dumps(GOOD_RAW_DRAFT))], usage=_fake_usage())
        fake_client, create = _fake_client(response)

        with patch("pipeline.extract_pdf_text", return_value=fake_extraction), patch(
            "pipeline.load_profile", return_value=({}, CONTACT, CHUNKS)
        ), patch("pipeline._resolve_api_key", return_value="sk-synthetic-test-key"), patch(
            "pipeline.OpenAI", return_value=fake_client
        ), patch(
            "pipeline.render_and_verify",
            return_value=RenderResult(
                docx_path=Path("resume.docx"), pdf_path=None, page_count=2, preset_name="comfortable",
                status="verified_two_pages", message="ok", attempts=[("comfortable", 2)],
            ),
        ):
            results = list(pipeline.run_pipeline("job.pdf", "gpt-6.1-sol"))

        full_text = "\n".join(text for text, _ in results)
        # job-1 shares "python"/"data"/"for" etc. with JOB_TEXT -> higher score, ranked first.
        job_line_index = full_text.index("[job-1]")
        skill_line_index = full_text.index("[skill-1]")
        self.assertLess(job_line_index, skill_line_index)
        self.assertIn("shared terms: python (", full_text)  # real, correctly-formatted output

    def test_overlapping_run_is_rejected(self):
        acquired = pipeline._RUN_LOCK.acquire(blocking=False)
        self.assertTrue(acquired)
        try:
            results = list(pipeline.run_pipeline("job.pdf", "gpt-6.1-sol"))
        finally:
            pipeline._RUN_LOCK.release()

        self.assertEqual(len(results), 1)
        text, download = results[0]
        self.assertIn("already in progress", text)
        self.assertIsNone(download)


if __name__ == "__main__":
    unittest.main()
