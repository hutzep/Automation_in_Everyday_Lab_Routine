#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "dspy",
#     "httpx",
#     "pymupdf",
#     "pymupdf4llm",
#     "typer",
# ]
# [tool.uv]
# exclude-newer = "2025-07-30T21:41:12Z"
# ///

import os
from contextlib import contextmanager
from typing import Annotated, Union, assert_never
from urllib.parse import ParseResult, urlparse

import dspy
import httpx
import pymupdf
import pymupdf4llm
import typer
from rich.console import Console
from rich.json import JSON
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt


def open_pdf(path_or_url: str) -> pymupdf.Document:
    url: ParseResult = urlparse(path_or_url)
    if url.scheme == "file" or not url.scheme:
        return pymupdf.Document(url.path, filetype="pdf")
    else:
        response = httpx.get(path_or_url, follow_redirects=True)
        response.raise_for_status()
        return pymupdf.Document(stream=response.content, filetype="pdf")


def format_as_markdown(sections: dict[str, Union[str, list[str]]]) -> str:
    markdown = ""

    for title, content in sections.items():
        markdown += f"## {title.capitalize()}\n\n"

        if isinstance(content, str):
            markdown += content
        elif isinstance(content, list):
            markdown += "\n".join(f"- {item}" for item in content)
        else:
            raise assert_never(content)

        markdown += "\n"

    markdown += "\n"

    return markdown


app = typer.Typer(
    pretty_exceptions_show_locals=False,
    add_completion=False,
)

console = Console()


@contextmanager
def Spinner(*, title: str):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task(description=title, total=None)
        yield


@app.command()
def main(
    path_or_url: Annotated[
        str,
        typer.Argument(help="Path or URL to paper"),
    ],
    model: Annotated[
        str,
        typer.Option(help="LLM provider and model to use."),
    ] = "anthropic/claude-sonnet-4-0",
    extract: Annotated[
        list[str],
        typer.Option(help="Information to extract from paper."),
    ] = ["summary", "keywords[]"],
    json: Annotated[
        bool,
        typer.Option("--json", help="Formats the extracted information as JSON."),
    ] = False,
    noninteractive: Annotated[
        bool,
        typer.Option("--noninteractive", help="Disables prompting for API keys."),
    ] = False,
):
    """Extracts information from local or web-hosted papers using LLMs.

    It is possible to provide the `--extract` option multiple times to extract
    separate pieces of information from the provided paper. Extract keys with
    the suffix [] will be extracted as lists.

    """

    try:
        api_key = os.environ["PAPER_EXTRACTOR_API_KEY"]
    except KeyError:
        if not noninteractive:
            console.print(
                "[yellow bold]![/] Environment variable `PAPER_EXTRACTOR_API_KEY` is not defined."
            )
            api_key = Prompt.ask(f"[red bold]?[/] API key for {model}", console=console)
        else:
            raise RuntimeError(
                "Missing environment variable `PAPER_EXTRACTOR_API_KEY` to connect to API."
            )

    lm = dspy.LM(model, api_key=api_key)
    dspy.configure(lm=lm)

    output_fields = {
        field.removesuffix("[]"): (
            list[str] if field.endswith("[]") else str,
            dspy.OutputField(),
        )
        for field in extract
    }

    PaperExtractorSignature = dspy.make_signature(
        {"context": (str, dspy.InputField())} | output_fields,
        instructions=f"""

        You are given the contents of a scientific paper as context. Your task
        is to extract information corresponding to the given fields from the
        context of your paper. You are doing this for an experienced reseacher
        who wants exact results.

        1. Think about how to interpret the fields the user asked for.
        2. Use the context given to you to extract these fields from the paper.

        Given the fields ['context'], produce the fields {list(output_fields.keys())}.

        """,
        signature_name="PaperExtractorSignature",
    )

    cot = dspy.ChainOfThought(PaperExtractorSignature)

    with Spinner(title="Downloading PDF file"):
        document = open_pdf(path_or_url)

    with Spinner(title="Processing PDF file"):
        pdf_content = pymupdf4llm.to_markdown(document)

    with Spinner(title="Querying LLM agent"):
        response = cot(context=pdf_content)

    output = {key: response[key] for key in output_fields.keys()}

    if json:
        console.print(JSON.from_data(output), overflow="ignore", crop=False)
    else:
        console.print(Markdown(format_as_markdown(output)))
        console.print()


if __name__ == "__main__":
    app()
