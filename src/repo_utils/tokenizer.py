import os
import json
import pandas as pd
from pathlib import Path
from docx import Document
from PyPDF2 import PdfReader
import tiktoken
import logging

# Suppress PDF read warnings, when reading corrupted PDFs
logger = logging.getLogger("PyPDF2")
logger.setLevel(logging.ERROR)


def extract_text_from_file(filepath):
    """
    Extract text content from a file based on its extension.

    Supports multiple file formats including PDF, DOCX, Jupyter notebooks,
    CSV, JSON, and plain text files. Binary files are detected and return
    an empty string.

    :param filepath: Path to the file to extract text from.
    :type filepath: str or Path
    :return: The extracted text content from the file. Returns an empty
        string if the file cannot be read or contains binary content.
    :rtype: str

    :Example:

    >>> content = extract_text_from_file("script.py")
    >>> content = extract_text_from_file("notebook.ipynb")
    """
    ext = Path(filepath).suffix.lower()
    try:
        if ext == ".pdf":
            reader = PdfReader(filepath)
            return "\n".join(page.extract_text() or "" for page in reader.pages)

        elif ext == ".docx":
            doc = Document(filepath)
            return "\n".join(p.text for p in doc.paragraphs)

        elif ext == ".ipynb":
            # Read Jupyter notebook as executable Python script
            # Extract code cells and markdown cells (as comments), ignore metadata/outputs
            with open(filepath, "r", encoding="utf-8") as f:
                notebook = json.load(f)

            parts = []
            for cell in notebook.get("cells", []):
                cell_type = cell.get("cell_type", "")
                source = cell.get("source", [])

                # Handle source as list of lines or single string
                if isinstance(source, list):
                    content = "".join(source)
                else:
                    content = source

                if cell_type == "code":
                    parts.append(content)
                elif cell_type == "markdown":
                    # Convert markdown to Python comments
                    commented = "\n".join(f"# {line}" for line in content.split("\n"))
                    parts.append(commented)

            return "\n\n".join(parts)

        elif ext == ".csv":
            df = pd.read_csv(filepath)
            return df.to_string()

        elif ext in [".json", ".jsonl"]:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()

        else:
            content = Path(filepath).read_text(encoding="utf-8", errors="ignore")
            if "\x00" in content:
                return ""
            return content

    except Exception as e:
        # print(f"Error reading {filepath}: {e}")
        return ""


def tokenize_text(text, encoding_name="cl100k_base"):
    """
    Tokenize text using a tiktoken encoding.

    :param text: The text string to tokenize.
    :type text: str
    :param encoding_name: The name of the tiktoken encoding to use.
        Defaults to "cl100k_base"
    :type encoding_name: str
    :return: A tuple containing the list of token IDs and the encoding object.
    :rtype: tuple[list[int], tiktoken.Encoding]

    :Example:

    >>> tokens, enc = tokenize_text("Hello, world!")
    >>> len(tokens)
    4
    >>> enc.decode(tokens)
    'Hello, world!'
    """
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    return tokens, enc
