"""app.py

Minimal local Gradio UI for pipeline.run_pipeline(). Binds to 127.0.0.1
only (share=False) - never exposed outside this machine. Building and
launching this app makes no API call by itself; the Generate button is
the only control that can trigger one, and it makes up to one paid
request per run (invalid input - missing PDF, no extractable text,
etc. - stops before any request; pipeline.py enforces the "at most
one" bound, not this file).
"""

import gradio as gr

from pipeline import run_pipeline

DEFAULT_MODEL = "gpt-6.1-sol"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Resume Drafting Demo") as demo:
        gr.Markdown(
            "# Resume Drafting Demo\n"
            "Upload a job-description PDF, then click Generate.\n\n"
            "**Generate makes up to one paid OpenAI API request per run** "
            "(it stops before requesting anything if the input is invalid)."
        )
        pdf_input = gr.File(label="Job description PDF", file_types=[".pdf"], type="filepath")
        model_input = gr.Textbox(label="Model", value=DEFAULT_MODEL)
        generate_button = gr.Button("Generate resume (up to 1 paid API request)", variant="primary")
        trace_output = gr.Textbox(
            label="Processing trace", lines=20, max_lines=50, interactive=False, autoscroll=True
        )
        download_output = gr.File(
            label="Download resume.docx (two pages is the target; a generated document at another page count is still offered)",
            interactive=False,
        )

        generate_button.click(
            fn=run_pipeline,
            inputs=[pdf_input, model_input],
            outputs=[trace_output, download_output],
        )

    return demo


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", share=False)
