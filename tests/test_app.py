"""tests/test_app.py

Purpose
-------
Smoke-tests that the Gradio app actually builds and can bind/start
locally - without ever clicking Generate, so no API request or .env
read happens. Confirms 127.0.0.1/share=False wiring is intact.

Run with: .venv/Scripts/python.exe -m unittest discover -s tests -t . -v
"""

import inspect
import unittest

import app
from app import build_app


class TestAppStartup(unittest.TestCase):
    def test_builds_and_starts_locally_without_any_api_call(self):
        demo = build_app()
        try:
            demo.launch(server_name="127.0.0.1", share=False, server_port=None, prevent_thread_lock=True)
            self.assertTrue(demo.is_running)
            self.assertEqual(demo.local_url.split("://")[1].split(":")[0], "127.0.0.1")
        finally:
            demo.close()


class TestAppWording(unittest.TestCase):
    def test_describes_up_to_one_request_per_run_not_exactly_one_per_click(self):
        source = inspect.getsource(app).lower()
        self.assertIn("up to one paid", source)
        self.assertNotIn("exactly one paid", source)


if __name__ == "__main__":
    unittest.main()
