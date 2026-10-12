"""tests/test_retrieval.py

Purpose
-------
Focused, repeatable unittest coverage for retrieve_relevant_chunks() in
retrieval.py. All records here are small synthetic Chunk objects, never
the real resume_profile.md, so these tests stay fast, deterministic, and
independent of any future edits to the actual profile content.

Run with: .venv/Scripts/python.exe -m unittest discover -s tests -t .
"""

import unittest

from retrieval import Chunk, format_shared_terms, retrieve_relevant_chunks


def make_chunk(chunk_id: str, text: str, section: str = "project", title: str | None = None) -> Chunk:
    return Chunk(chunk_id=chunk_id, section=section, title=title or chunk_id, text=text)


class TestRetrieveRelevantChunks(unittest.TestCase):
    def test_ranks_more_overlapping_chunk_first(self):
        chunks = [
            make_chunk("a", "Built Python data pipelines using pandas and SQL for analytics."),
            make_chunk("b", "Designed HVAC duct layouts mechanical equipment schedules field surveys."),
        ]
        job_text = "Looking for a Python data analyst with pandas and SQL experience."

        matches = retrieve_relevant_chunks(job_text, chunks)

        # Chunk b shares no real vocabulary with the job text, so only a
        # (the Python/pandas/SQL overlap) should come back as a match.
        self.assertEqual([m.chunk.chunk_id for m in matches], ["a"])
        self.assertGreater(matches[0].score, 0)

    def test_contributions_sum_to_score(self):
        chunks = [
            make_chunk("a", "Python pandas SQL data analysis reporting."),
            make_chunk("b", "AutoCAD HVAC mechanical design drafting."),
        ]
        job_text = "Python pandas SQL analysis role."

        matches = retrieve_relevant_chunks(job_text, chunks)

        self.assertTrue(matches)
        top = matches[0]
        total_contribution = sum(top.term_contributions.values())
        self.assertAlmostEqual(total_contribution, top.score, places=9)

    def test_no_vocabulary_overlap_returns_empty(self):
        chunks = [make_chunk("a", "Python pandas SQL data analysis.")]
        job_text = "Completely unrelated gardening landscaping duties."

        matches = retrieve_relevant_chunks(job_text, chunks)

        self.assertEqual(matches, [])

    def test_blank_job_text_returns_empty(self):
        chunks = [make_chunk("a", "Python pandas SQL data analysis.")]

        self.assertEqual(retrieve_relevant_chunks("", chunks), [])
        self.assertEqual(retrieve_relevant_chunks("   \n\t  ", chunks), [])

    def test_empty_chunk_list_returns_empty(self):
        self.assertEqual(retrieve_relevant_chunks("Python developer wanted.", []), [])

    def test_single_letter_tool_name_r_is_preserved(self):
        chunks = [make_chunk("a", "Statistical modeling in R, Python, and SQL.")]
        job_text = "Seeking a candidate skilled in R for statistical analysis."

        matches = retrieve_relevant_chunks(job_text, chunks)

        self.assertEqual(len(matches), 1)
        self.assertIn("r", matches[0].term_contributions)
        self.assertGreater(matches[0].term_contributions["r"], 0)

    def test_deterministic_tie_ordering(self):
        chunks = [
            make_chunk("z-chunk", "Python pandas SQL data analysis."),
            make_chunk("a-chunk", "Python pandas SQL data analysis."),
        ]
        job_text = "Python pandas SQL data analysis role."

        matches = retrieve_relevant_chunks(job_text, chunks)

        self.assertEqual(len(matches), 2)
        self.assertAlmostEqual(matches[0].score, matches[1].score, places=9)
        # Equal scores must break ties by chunk_id, not input order.
        self.assertEqual([m.chunk.chunk_id for m in matches], ["a-chunk", "z-chunk"])

    def test_top_k_rejects_bool(self):
        chunks = [make_chunk("a", "Python pandas SQL data analysis.")]

        with self.assertRaises(ValueError):
            retrieve_relevant_chunks("Python developer wanted.", chunks, top_k=True)

    def test_top_k_rejects_non_integer(self):
        chunks = [make_chunk("a", "Python pandas SQL data analysis.")]

        with self.assertRaises(ValueError):
            retrieve_relevant_chunks("Python developer wanted.", chunks, top_k=2.5)
        with self.assertRaises(ValueError):
            retrieve_relevant_chunks("Python developer wanted.", chunks, top_k="3")

    def test_top_k_rejects_zero_and_negative(self):
        chunks = [make_chunk("a", "Python pandas SQL data analysis.")]

        with self.assertRaises(ValueError):
            retrieve_relevant_chunks("Python developer wanted.", chunks, top_k=0)
        with self.assertRaises(ValueError):
            retrieve_relevant_chunks("Python developer wanted.", chunks, top_k=-1)

    def test_invalid_top_k_raises_even_with_blank_job_text_and_empty_chunks(self):
        # top_k validation must happen before the blank-job-text/empty-chunks
        # early returns, so an invalid top_k is never silently swallowed.
        with self.assertRaises(ValueError):
            retrieve_relevant_chunks("", [], top_k=0)
        with self.assertRaises(ValueError):
            retrieve_relevant_chunks("", [], top_k=-5)
        with self.assertRaises(ValueError):
            retrieve_relevant_chunks("some job text", [], top_k=False)

    def test_chunks_with_no_usable_tokens_returns_empty(self):
        chunks = [
            make_chunk("a", "--- ... !!! ???"),
            make_chunk("b", "***  ---  ...."),
        ]

        matches = retrieve_relevant_chunks("Python developer wanted.", chunks)

        self.assertEqual(matches, [])

    def test_synonym_only_overlap_is_not_detected(self):
        """TF-IDF matches literal tokens only. A job description written
        entirely with synonyms of the chunk's wording ("attorney" for
        "lawyer", "business agreements" for "corporate contracts") and no
        shared tokens at all scores zero, even though the two describe the
        same kind of role. This is an expected limitation of lexical
        retrieval, not a bug in retrieve_relevant_chunks()."""
        chunks = [
            make_chunk(
                "a",
                "Served many years as lawyer reviewing complex corporate contracts quietly.",
            )
        ]
        job_text = "Searching urgently for skilled attorney managing big business agreements promptly."

        matches = retrieve_relevant_chunks(job_text, chunks)

        self.assertEqual(matches, [])


class TestFormatSharedTerms(unittest.TestCase):
    """format_shared_terms() takes an already-sorted list of (term,
    contribution) pairs - it only truncates/formats, it does not sort.
    A raw, unsorted dict (or its unsorted .items()) is NOT the expected
    input; the caller must sort first (see retrieve_relevant_chunks's
    own RetrievalMatch.term_contributions construction for the standard
    sort key: descending contribution, ties broken by term name)."""

    def test_formats_pre_sorted_pairs_with_truncation_note(self):
        pairs = [("python", 0.182), ("sql", 0.091), ("pandas", 0.05), ("r", 0.02), ("sql server", 0.01), ("excel", 0.005)]

        rendered = format_shared_terms(pairs, limit=3)

        self.assertEqual(
            rendered, "python (0.182), sql (0.091), pandas (0.050) (+3 more shared terms not shown)"
        )

    def test_does_not_sort_unsorted_input_itself(self):
        """Documents that format_shared_terms trusts caller ordering -
        passing pairs out of order renders them out of order too."""
        unsorted_pairs = [("sql", 0.091), ("python", 0.182)]

        rendered = format_shared_terms(unsorted_pairs)

        self.assertEqual(rendered, "sql (0.091), python (0.182)")

    def test_empty_list_reports_no_shared_terms(self):
        self.assertEqual(format_shared_terms([]), "(no shared terms)")


if __name__ == "__main__":
    unittest.main()
