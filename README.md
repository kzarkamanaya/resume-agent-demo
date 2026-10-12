# Resume Agent Demo

A school project exploring a small AI-assisted resume tailoring workflow: given a local Markdown candidate profile and an uploaded job-description PDF, the tool is intended to produce a tailored, exactly-two-page Word resume. The plan is for local Python code to handle PDF extraction, heading-based chunking, TF-IDF retrieval, Word document creation, and rendered page-count validation, with a single OpenAI API request used to select supported evidence and draft content (including a short professional summary). A Gradio browser interface is intended to show the processing trace and offer a download.

## Demo data disclosure

Any candidate data in this repository, including `resume_profile.md`, is fictionalized demo data created for this project and does not describe a real person.

## Status

Implementation is **not complete**.

Done so far:
- `retrieval.py` loads `resume_profile.md` into 28 intact chunks (4 education/training records, 3 jobs, 12 projects, 9 skills categories), keeping contact info separate.
- `retrieval.py` also does local TF-IDF retrieval: given job-description text, it ranks those chunks by cosine similarity and reports each match's score and shared-term contributions.

Not implemented yet: PDF extraction of the job description, the OpenAI drafting call, Word document generation and two-page verification, and the Gradio browser UI.

### Try it (PowerShell, from the project root)

```powershell
.\.venv\Scripts\python.exe retrieval.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
```

`retrieval.py` run directly lists the 28 profile chunks (ids, titles, counts) - it does not run retrieval itself. The test suite (`tests/test_retrieval.py`) is what exercises TF-IDF retrieval, including ranking, score/contribution math, and edge cases.

### A note on the retrieval scores

Scores measure lexical (word/phrase) similarity between a chunk and a job description - they are **not** a measure of qualification or of how fully a chunk covers a job's requirements. Synonyms (e.g. "attorney" vs. "lawyer") can be missed entirely, and job-description terms that never appear anywhere in the profile are simply ignored rather than penalized.
