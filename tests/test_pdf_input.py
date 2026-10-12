"""tests/test_pdf_input.py

Purpose
-------
Focused, repeatable unittest coverage for extract_pdf_text() in
pdf_input.py. Every fixture PDF used here is built on the fly, in a
temporary directory, using pypdf's own writer API (no reportlab/fpdf and
no checked-in binary files) - pypdf has no high-level "draw text" call,
so _add_text_page() below assembles a minimal content stream and a
standard Helvetica font resource directly through pypdf's object model
(ContentStream/DictionaryObject/NameObject), then hands the result back
to pypdf's own reader to confirm pypdf can round-trip what it wrote.

Run with: .venv/Scripts/python.exe -m unittest discover -s tests -t . -v
"""

import io
import tempfile
import unittest
from pathlib import Path
from typing import List, Optional
from unittest.mock import PropertyMock, patch

from pypdf import PdfReader, PdfWriter
from pypdf.errors import PdfReadError
from pypdf.generic import ContentStream, DictionaryObject, NameObject

from pdf_input import PdfPasswordProtectedError, extract_pdf_text


def _add_text_page(writer: PdfWriter, text: Optional[str], width: int = 200, height: int = 200) -> None:
    """Add one page to `writer`. `text=None` adds a genuinely blank page
    (no content stream at all); any other value adds a page whose content
    stream shows that exact text with the standard Helvetica font."""
    page = writer.add_blank_page(width=width, height=height)
    if text is None:
        return

    ops = f"BT /F1 12 Tf 10 {height - 20} Td ({text}) Tj ET".encode("latin-1")
    content_stream = ContentStream(None, writer)
    content_stream.set_data(ops)
    page.replace_contents(content_stream)

    font = DictionaryObject()
    font[NameObject("/Type")] = NameObject("/Font")
    font[NameObject("/Subtype")] = NameObject("/Type1")
    font[NameObject("/BaseFont")] = NameObject("/Helvetica")
    font_ref = writer._add_object(font)

    fonts = DictionaryObject()
    fonts[NameObject("/F1")] = font_ref
    resources = DictionaryObject()
    resources[NameObject("/Font")] = fonts
    page[NameObject("/Resources")] = resources


def _write_pdf(
    directory: Path,
    filename: str,
    page_texts: List[Optional[str]],
    user_password: Optional[str] = None,
    owner_password: Optional[str] = None,
) -> Path:
    """Build a synthetic PDF with one page per entry in `page_texts`
    (None -> blank page, str -> real extractable text) and write it to
    `directory / filename`. If `user_password` is given, encrypt the
    file with it (empty string "" is a valid, deliberately-weak user
    password many real PDFs use just to restrict permissions)."""
    writer = PdfWriter()
    for text in page_texts:
        _add_text_page(writer, text)

    if user_password is not None:
        writer.encrypt(user_password=user_password, owner_password=owner_password or "owner-secret")

    path = directory / filename
    with open(path, "wb") as fh:
        writer.write(fh)
    return path


class TestExtractPdfText(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)

    def test_extracts_multi_page_text(self):
        pdf_path = _write_pdf(
            self.tmp_path,
            "two_pages.pdf",
            ["Hello World Page One", "Second Page Content Here"],
        )

        result = extract_pdf_text(pdf_path)

        self.assertEqual(len(result.pages), 2)
        self.assertEqual(result.pages[0].strip(), "Hello World Page One")
        self.assertEqual(result.pages[1].strip(), "Second Page Content Here")
        self.assertIn("Hello World Page One", result.full_text)
        self.assertIn("Second Page Content Here", result.full_text)
        self.assertEqual(result.warnings, [])

    def test_mixed_text_and_blank_pages(self):
        pdf_path = _write_pdf(
            self.tmp_path,
            "mixed.pdf",
            ["First real page", None, "Third real page"],
        )

        result = extract_pdf_text(pdf_path)

        self.assertEqual(len(result.pages), 3)
        self.assertEqual(result.pages[1], "")
        self.assertIn("First real page", result.full_text)
        self.assertIn("Third real page", result.full_text)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("1 of 3", result.warnings[0])
        self.assertIn("[2]", result.warnings[0].replace(" ", ""))

    def test_all_blank_pages_warns_clearly(self):
        pdf_path = _write_pdf(self.tmp_path, "all_blank.pdf", [None, None])

        result = extract_pdf_text(pdf_path)

        self.assertEqual(result.pages, ["", ""])
        self.assertEqual(result.full_text, "")
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("No extractable text was found on any page", result.warnings[0])
        self.assertIn("OCR is not implemented", result.warnings[0])

    def test_zero_page_pdf_warns_and_returns_empty(self):
        pdf_path = _write_pdf(self.tmp_path, "zero_pages.pdf", [])

        result = extract_pdf_text(pdf_path)

        self.assertEqual(result.pages, [])
        self.assertEqual(result.full_text, "")
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("zero pages", result.warnings[0])
        self.assertIn("retrieval/drafting cannot proceed", result.warnings[0])

    def test_page_access_read_error_is_converted_to_value_error(self):
        """A PdfReadError raised while counting or accessing pages (not
        just while extracting a page's text) must still come back as a
        plain ValueError with the original error preserved as its cause,
        and must never be mistaken for PdfPasswordProtectedError."""
        pdf_path = _write_pdf(self.tmp_path, "normal.pdf", ["Some text"])

        with patch.object(
            PdfReader,
            "pages",
            new_callable=PropertyMock,
            side_effect=PdfReadError("simulated malformed page tree"),
        ):
            with self.assertRaises(ValueError) as ctx:
                extract_pdf_text(pdf_path)

        self.assertNotIsInstance(ctx.exception, PdfPasswordProtectedError)
        self.assertIsInstance(ctx.exception.__cause__, PdfReadError)

    def test_corrupt_file_raises_value_error(self):
        bad_path = self.tmp_path / "corrupt.pdf"
        bad_path.write_bytes(b"not a pdf at all, just garbage bytes 12345")

        with self.assertRaises(ValueError):
            extract_pdf_text(bad_path)

    def test_missing_file_raises_file_not_found(self):
        missing_path = self.tmp_path / "does_not_exist.pdf"

        with self.assertRaises(FileNotFoundError):
            extract_pdf_text(missing_path)

    def test_encrypted_with_empty_password_is_read_successfully(self):
        pdf_path = _write_pdf(
            self.tmp_path,
            "encrypted_empty_password.pdf",
            ["Secret-free content behind an empty password"],
            user_password="",
            owner_password="owner-secret",
        )

        result = extract_pdf_text(pdf_path)

        self.assertEqual(len(result.pages), 1)
        self.assertIn("Secret-free content behind an empty password", result.full_text)
        self.assertEqual(result.warnings, [])

    def test_encrypted_with_real_password_is_rejected(self):
        pdf_path = _write_pdf(
            self.tmp_path,
            "encrypted_real_password.pdf",
            ["Content behind a real password"],
            user_password="correct-password",
            owner_password="owner-secret",
        )

        with self.assertRaises(PdfPasswordProtectedError):
            extract_pdf_text(pdf_path)


if __name__ == "__main__":
    unittest.main()
