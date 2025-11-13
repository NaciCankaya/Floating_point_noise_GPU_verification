#!/usr/bin/env python3
"""
Prompt generation utilities for ablation experiments

Extracts text from PDFs and prepares prompts with ~6k tokens for consistent experiments.
"""

import PyPDF2
from pathlib import Path
from typing import Optional


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text


def load_prompt_from_pdf(
    pdf_path: str,
    tokenizer,
    target_tokens: int = 6000,
    task_instruction: str = "Please provide a comprehensive summary of the following text:",
) -> tuple[str, int]:
    """
    Load text from PDF, truncate to target token count, and prepare prompt.

    Args:
        pdf_path: Path to PDF file
        tokenizer: HuggingFace tokenizer to count tokens
        target_tokens: Target number of tokens for the document text (default: 6000)
        task_instruction: Instruction to prepend to the document

    Returns:
        tuple: (prompt_text, actual_token_count)
    """
    # Extract text from PDF
    pdf_text = extract_text_from_pdf(pdf_path)

    # Clean up text (remove excessive whitespace, newlines)
    pdf_text = " ".join(pdf_text.split())

    # Tokenize and truncate to target length
    tokens = tokenizer.encode(pdf_text, add_special_tokens=False)

    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
        pdf_text = tokenizer.decode(tokens, skip_special_tokens=True)

    # Construct final prompt with task instruction
    prompt = f"{task_instruction}\n\n{pdf_text}\n\nSummary:"

    # Count total tokens in final prompt
    total_tokens = len(tokenizer.encode(prompt, add_special_tokens=True))

    return prompt, total_tokens


def load_multiple_prompts_from_pdfs(
    pdf_paths: list[str],
    tokenizer,
    target_tokens: int = 6000,
    task_instruction: str = "Please provide a comprehensive summary of the following text:",
) -> list[tuple[str, int]]:
    """
    Load prompts from multiple PDFs for batch experiments.

    Args:
        pdf_paths: List of paths to PDF files
        tokenizer: HuggingFace tokenizer
        target_tokens: Target number of tokens per document
        task_instruction: Instruction to prepend

    Returns:
        list of (prompt_text, token_count) tuples
    """
    prompts = []
    for pdf_path in pdf_paths:
        prompt, token_count = load_prompt_from_pdf(
            pdf_path, tokenizer, target_tokens, task_instruction
        )
        prompts.append((prompt, token_count))
    return prompts


# Default PDF for experiments (can be overridden)
DEFAULT_PDF = Path(__file__).parent.parent / "ablation_cross_hardware" / "Qwen3_Technical_Report.pdf"
