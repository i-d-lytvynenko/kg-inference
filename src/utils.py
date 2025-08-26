import os
from textwrap import dedent

import PyPDF2


def format_prompt(prompt: str) -> str:
    return dedent(prompt.strip())


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text from the PDF
    """
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n\n"

    return text


def process_pdf_files(
    pdf_paths: list[str],
    max_pages: int | None = None,
    page_limit_per_file: int | None = None,
) -> str:
    """
    Process multiple PDF files and extract their text.

    Args:
        pdf_paths: list of paths to PDF files
        max_pages: Maximum total number of pages to process across all files
        page_limit_per_file: Maximum number of pages to process per file

    Returns:
        Extracted text from all PDFs, concatenated
    """
    all_text = ""
    total_pages_processed = 0

    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"Warning: PDF file not found: {pdf_path}")
            continue

        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)

            # Apply per-file page limit if specified
            if page_limit_per_file:
                pages_to_process = min(num_pages, page_limit_per_file)
            else:
                pages_to_process = num_pages

            # Check if we've hit the maximum total pages
            if max_pages and total_pages_processed + pages_to_process > max_pages:
                pages_to_process = max_pages - total_pages_processed

            if pages_to_process <= 0:
                break

            # Extract text from pages
            file_text = ""
            for page_num in range(pages_to_process):
                page = pdf_reader.pages[page_num]
                file_text += page.extract_text() + "\n\n"

            # Add filename as header
            file_basename = os.path.basename(pdf_path)
            all_text += f"\n\n--- PDF: {file_basename} ---\n\n{file_text}"

            total_pages_processed += pages_to_process

            # Check if we've hit the maximum total pages
            if max_pages and total_pages_processed >= max_pages:
                break

    return all_text.strip()
