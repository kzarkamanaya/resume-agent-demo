# Resume Agent Demo

A school project exploring a small AI-assisted resume tailoring workflow: given a local Markdown candidate profile and an uploaded job-description PDF, the tool is intended to produce a tailored, exactly-two-page Word resume. The plan is for local Python code to handle PDF extraction, heading-based chunking, TF-IDF retrieval, Word document creation, and rendered page-count validation, with a single OpenAI API request used to select supported evidence and draft content (including a short professional summary). A Gradio browser interface is intended to show the processing trace and offer a download.

## Demo data disclosure

Any candidate data in this repository, including `resume_profile.md`, is fictionalized demo data created for this project and does not describe a real person.

## Status

Implementation is **not complete**. This repository currently contains only a safety scaffold (`.gitignore`, `.env.example`, this README) and the demo profile. No application code, dependencies, or processing pipeline exist yet.
