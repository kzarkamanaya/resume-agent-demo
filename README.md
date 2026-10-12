# Resume Agent Demo

This application turns a job-description PDF and a local candidate profile into an evidence-based, tailored Word resume, while showing its decisions and verifying a two-page layout.

## Demo data disclosure

Any candidate data in this repository, including `resume_profile.md`, is fictionalized demo data created for this project and does not describe a real person. A real profile, if you ever use one, belongs in the local-only, Git-ignored `private/` folder - never in `resume_profile.md` as committed here.

## Setup (Windows, PowerShell)

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

This installs the pinned versions in `requirements.txt` (scikit-learn, pypdf, openai, python-dotenv, python-docx, gradio) into the project's own `.venv`, not system-wide.

You also need **LibreOffice** installed locally (used headless to render the `.docx` to PDF for page counting; the code defaults to `C:\Program Files\LibreOffice\program\soffice.exe`, overridable via the `LIBREOFFICE_PATH` environment variable or a `--soffice` flag).

Then create your own `.env` next to this README (it is Git-ignored and never committed):

```powershell
Copy-Item .env.example .env
notepad .env
```

Put your real key in `.env` as `OPENAI_API_KEY=sk-...`. Never put a real key in `.env.example` - it must stay blank, and never paste a real key into chat, a screenshot, or a commit.

## Running it

All commands below are PowerShell, run from the project root.

**Launch the browser UI:**
```powershell
.\.venv\Scripts\python.exe app.py
```
Open the printed `http://127.0.0.1:7860` URL, upload a job-description PDF, and click Generate.

**Run the automated test suite:**
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
```

**Draft from the command line and save the validated result** (so it can be re-rendered later without another API call):
```powershell
.\.venv\Scripts\python.exe decision.py path\to\job.pdf --model gpt-6.1-sol --save-json outputs\draft.json
```

**Render a previously saved draft to a page-counted `.docx` - no API call:**
```powershell
.\.venv\Scripts\python.exe document.py outputs\draft.json --out outputs\resume.docx
```

## Code flow

`pdf_input.py` extracts the job description's text from the uploaded PDF. `retrieval.py` loads `resume_profile.md` into chunks (jobs, education/training, projects, skills categories) and computes local TF-IDF scores against that text. `decision.py` builds the drafting prompt, makes the one OpenAI request, and validates the structured response. `document.py` builds the `.docx` and renders it with LibreOffice to verify the page count. `pipeline.py` orchestrates all four and streams a live trace; `app.py` is the thin Gradio UI on top of `pipeline.py`.

## How evidence and scoring actually work

**All** profile chunks are sent to the model as evidence - never a pre-filtered subset. The local TF-IDF scores and their shared-term contributions are **trace information only**: they're shown in the UI/terminal so you can see what the local retrieval step found, but they are never part of the API request and never used to filter or select which chunks the model sees.

The **model** is what selects, prioritizes, and drafts from that evidence - deciding what's relevant, writing the summary and bullets, and citing which chunks it used. A simpler rule-based template could assemble a resume from the same chunks without any model call; what the model adds is semantic matching (recognizing relevant evidence even without matching keywords) and tailored, readable writing - not just mechanical insertion.

This is a **bounded AI workflow**: one well-defined request per run, validated structured output, and local verification steps around it - not a fully autonomous agent making open-ended decisions or further tool calls on its own.

## Cost and requests

Each run makes **up to one** paid drafting request (not "exactly one" - invalid input, like a PDF with no extractable text, stops before any request is made). Retries are disabled. The trace reports the actual token usage the API returns, and separately an **explicitly labeled estimate** of cost, based on verified published pricing for the model actually used - never a guessed price for an unverified model.

## Download gating

A download is only offered once LibreOffice renders the document to **exactly two pages**. Microsoft Word can paginate the same document slightly differently than LibreOffice, so this is the best local check available, not an absolute guarantee of what Word will show you.

## Known limits

- No OCR: a scanned/image-only PDF yields no extractable text, and the run stops before requesting anything.
- Evidence IDs confirm a claim points at a real profile chunk - they do not prove the model's paraphrase of that chunk is faithful to it.
- The institution/dates and location/dates extracted for each resume entry follow this demo profile's own bullet conventions; a differently formatted profile may not extract cleanly.
- Per-preset rendering results (which formatting preset was tried, how many pages it produced) are reported together right after rendering finishes - not as live, one-at-a-time updates during rendering.

## Not implemented: automatic shortening

If a render comes out overlong or underlength, the run stops and reports that clearly - it does **not** automatically request a shorter draft, and it never pads or cuts content to force a page count. No download is offered in that case.

## Before a live class demo

- Install dependencies (`pip install -r requirements.txt`) and confirm LibreOffice is installed *before* the demo, not live.
- Test with a job description the app hasn't seen before, so the result isn't memorized/cached behavior.
- Keep a previously-verified resume output and a few screenshots as a fallback in case live rendering or the API has an issue. Never screenshot `.env` or any API key.

## Testing notes

The automated test suite (`tests/`) mocks all OpenAI API calls and LibreOffice rendering - it verifies the code's own logic (validation, orchestration, formatting, error handling), not real model behavior or real rendering. Real browser and rendering verification is a separate, manual step you run yourself (see "Running it" above); testing against a genuinely unseen job description has not already been done as part of this automated suite.
