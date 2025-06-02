# document_ingestion.py

"""
This module provides functions for ingesting documents from various sources
such as text files, PDF files, and web pages.
"""

import typing
import pdfplumber
import requests
from bs4 import BeautifulSoup

def read_text_file(file_path: str) -> str:
    """
    Reads a plain text file and returns its content as a string.

    Args:
        file_path: The path to the text file.

    Returns:
        The content of the text file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        return ""  # Or raise an exception
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return ""  # Or raise an exception

def read_pdf_file(file_path: str) -> str:
    """
    Reads a PDF file and extracts its text content as a string.
    Note: This is a placeholder and will require a PDF processing library.

    Args:
        file_path: The path to the PDF file.

    Returns:
        The extracted text content from the PDF file.
    """
    extracted_text = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:  # Ensure text was extracted
                    extracted_text.append(text)
        return "\n".join(extracted_text)
    except FileNotFoundError:
        print(f"Error: PDF file not found at path: {file_path}")
        return ""
    except Exception as e:  # Catch other potential pdfplumber errors
        print(f"Error reading PDF file {file_path}: {e}")
        return ""

def fetch_web_page_content(url: str) -> str:
    """
    Fetches the main textual content from a given URL.

    Args:
        url: The URL of the web page.

    Returns:
        The main textual content of the web page, or an empty string on error.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)

        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract text from the body, stripping extra whitespace and using space as separator
        # For more targeted extraction, one might look for <article>, <main>, or specific class/id.
        body = soup.body
        if body:
            return body.get_text(separator=' ', strip=True)
        else:
            # Fallback if body tag is not found (unlikely for valid HTML)
            return soup.get_text(separator=' ', strip=True)

    except requests.exceptions.Timeout:
        print(f"Error: Request timed out for URL: {url}")
        return ""
    except requests.exceptions.HTTPError as e:
        print(f"Error: HTTP error {e.response.status_code} for URL: {url}")
        return ""
    except requests.exceptions.RequestException as e:
        print(f"Error: Could not fetch content from URL: {url}. Error: {e}")
        return ""
    except Exception as e: # Catch other potential BeautifulSoup errors or unexpected issues
        print(f"Error processing web page content from {url}: {e}")
        return ""

if __name__ == '__main__':
    # Example usage (for testing purposes)
    print("--- Testing read_text_file ---")
    # Test with an existing file
    text_content_sample = read_text_file("sample.txt")
    if text_content_sample:
        print("Content of sample.txt:")
        print(text_content_sample)
    print("\n--- Testing read_text_file with non-existent file ---")
    text_content_non_existent = read_text_file("non_existent_file.txt")
    print(f"Content of non_existent_file.txt: '{text_content_non_existent}'\n")

    print("--- Testing read_pdf_file ---")
    # Note: This test requires a 'sample.pdf' file to be present.
    # For now, it will likely print an error or empty content if the file doesn't exist.
    pdf_content_sample = read_pdf_file("sample.pdf")
    if pdf_content_sample:
        print("Extracted text from sample.pdf:")
        print(pdf_content_sample)
    else:
        print("Could not extract text from sample.pdf (it might be missing or not a valid PDF).")
    print("\n--- Testing read_pdf_file with non-existent file ---")
    pdf_content_non_existent = read_pdf_file("non_existent_sample.pdf")
    print(f"Content of non_existent_sample.pdf: '{pdf_content_non_existent}'\n")

    web_content = fetch_web_page_content("http://example.com")
    print(f"Web content: {web_content}\n")
