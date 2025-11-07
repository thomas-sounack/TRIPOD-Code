import os
import json
import pandas as pd
from pathlib import Path
from docx import Document
from PyPDF2 import PdfReader
import tiktoken

def extract_text_from_file(filepath):
    """
    Returns the contents of the processed files in the repository
    """
    ext = Path(filepath).suffix.lower()
    code_extensions = {
        ".py", ".r", ".java", ".js", ".ts", ".tsx", ".jsx",
        ".php", ".html", ".css", ".cpp", ".c", ".h", ".hpp",
        ".cs", ".go", ".rs", ".swift", ".kt", ".scala", ".rb",
        ".pl", ".lua", ".sh", ".md", ".toml", ".sample", ".groovy", ".php",
        ".ipynb", ".yaml", ".dvc", ".ttl", ".yml"
    }
    if ext in code_extensions:
        return Path(filepath).read_text(encoding="utf-8", errors="ignore")
    
    if ext == "":
        try:
            content = Path(filepath).read_text(encoding="utf-8")
            if "\x00" in content:
                return ""
            return content
        except Exception:
            return ""

    elif ext == ".txt":
        return Path(filepath).read_text(encoding="utf-8", errors="ignore")

    elif ext == ".pdf":
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    elif ext == ".docx":
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs)

    elif ext == ".csv":
        df = pd.read_csv(filepath)
        return df.to_string()

    elif ext in [".json", ".jsonl"]:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
def tokenize_text(text, encoding_name="cl100k_base"):
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    return tokens, enc