from pathlib import Path
import re
from pydantic import BaseModel, HttpUrl
from enum import Enum
from typing import List, Union, Optional
from urllib.parse import urlparse
import os
from .repo_providers import get_repo_cloner, RepoNotSupportedError
from .tokenizer import extract_text_from_file, tokenize_text

TOKEN_CUTOFF_PER_FILE = 3000


class RepoStatus(str, Enum):
    OK = "ok"
    EMPTY = "empty"
    INACCESSIBLE = "inaccessible"
    NOT_SUPPORTED = "not_supported"


class RepoExtractionResult(BaseModel):
    repo_url: str
    status: RepoStatus
    repo_path: Optional[str] = None
    output: Optional[str] = None
    error: Optional[str] = None


class RepoTree(BaseModel):
    name: str
    children: Union[List["RepoTree"], None] = None


RepoTree.model_rebuild()


class RepoInfo(BaseModel):
    url: HttpUrl
    tree: List[RepoTree]


def get_tree(path: Path) -> List[RepoTree]:
    """Recursively builds the file/directory tree structure."""
    tree = []
    for item in path.iterdir():
        if item.name.startswith("."):
            continue
        if item.is_dir():
            tree.append(RepoTree(name=item.name, children=get_tree(item)))
        else:
            tree.append(
                RepoTree(
                    name=item.name,
                )
            )

    return tree


# file that will be excluded from the text string but are reported in the repo tree
block_list = {
    ".pt",
    ".pth",
    ".pkl",
    ".h5",
    ".onnx",  # model checkpoints
    ".pack",  # compressed binary file for commits, blob, etc
    ".lock",  # lock files for dependency management
    ".log",
    ".bin",
    ".db",
    ".sqlite",  # large logs/databases
    ".zip",
    ".tar",
    ".gz",
    ".7z",  # compressed archives
    ".csv",
    ".tsv",
    ".parquet",
    ".feather",
    ".arrow",
    ".xls",
    ".xlsx",
    ".sav",
    ".dta",
    ".sas7bdat",
    ".jsonl",
    ".hdf5",
    ".mat",
    ".npy",
    ".npz",
    ".rds",
    ".rdata",
    ".avro",
    ".orc",
    ".mdb",
    ".accdb",
    ".dbf",
    ".pqt",
    ".rec",
    ".arff",
    ".psv",
    ".br",
}

# file included without any token cutoff
allow_list = {
    ".py",
    ".ipynb",
    ".r",
    ".java",
    ".jl",
    ".kt",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".go",
    ".rs",
    ".swift",
    ".php",
    ".rb",
    ".cs",
    ".scala",
    ".hs",
    ".dart",
    ".lua",
    ".pl",
    ".fs",
    ".asm",
    ".toml",
}


def read_all_files(base_path: Path, verbose: bool, context_window: int) -> str:
    """
    Reads files in the repo. Those in the blacklist are excluded,
    those in the allowlist are taken in full, and the rest are taken
    with a per-file token cutoff.

    Additionally, we track a global token budget (`context_window`).
    For ANY file (allowlisted or not), if adding that file's header+content
    would exceed the context window, we only add the header and a
    truncation marker, then stop reading further files.
    """
    file_contents = []
    total_tokens = 0

    for file_path in base_path.rglob("*"):
        if not file_path.is_file():
            continue

        rel_path = file_path.relative_to(base_path)
        rel_path_str = str(rel_path)

        is_readme = bool(re.match(r"readme(\.md|\.txt)?", rel_path_str, re.IGNORECASE))

        # Skip ignored suffixes entirely
        if file_path.suffix.lower() in block_list:
            continue

        header = f"\n\nFILE: {rel_path_str}\n"

        try:
            # Get file content via your existing extractor
            content = extract_text_from_file(file_path)

            # Apply *per-file* cutoff only for non-allowlisted & non-readme files
            if not (is_readme or file_path.suffix.lower() in allow_list):
                tokens, enc = tokenize_text(content)
                if len(tokens) > TOKEN_CUTOFF_PER_FILE:
                    original_len = len(tokens)
                    truncated_text = enc.decode(tokens[:TOKEN_CUTOFF_PER_FILE])
                    content = (
                        truncated_text
                        + f"\n\n[... CONTENT TRUNCATED: {original_len:,} tokens > "
                        f"{TOKEN_CUTOFF_PER_FILE:,} token limit ]"
                    )

            # Now compute tokens for what we plan to add (header + content)
            segment = header + content
            segment_tokens, _ = tokenize_text(segment)
            segment_token_len = len(segment_tokens)

            # If adding this file would exceed the global context window:
            # only add the header and a repository-level truncation note, then stop.
            if total_tokens + segment_token_len > context_window:
                truncation_notice = "(REST OF THE REPOSITORY IS TRUNCATED)"
                file_contents.append(header + truncation_notice)
                break

            # Otherwise, add it and update total
            file_contents.append(segment)
            total_tokens += segment_token_len

        except Exception as e:
            if verbose:
                print(f"Skipping {file_path}: {e}")

    return "\n".join(file_contents)


def clone_and_extract_tree(
    repo_url: str,
    context_window: int,
    clone_dir: str,
    verbose: bool = False,
) -> RepoExtractionResult:

    try:
        cloner = get_repo_cloner(repo_url)
        base = Path(clone_dir)
        base.mkdir(parents=True, exist_ok=True)

        repo_path = cloner.clone(repo_url, base)

        text = read_all_files(repo_path, verbose, context_window)

        if not text.strip():
            return RepoExtractionResult(
                repo_url=repo_url,
                status=RepoStatus.EMPTY,
                repo_path=str(repo_path),
            )

        tree = get_tree(repo_path)
        repo_info = RepoInfo(url=repo_url, tree=tree)

        output = (
            f"Repository tree\n{repo_info.model_dump_json(indent=2)}"
            f"\nFiles content\n{text}"
        )

        return RepoExtractionResult(
            repo_url=repo_url,
            status=RepoStatus.OK,
            repo_path=str(repo_path),
            output=output,
        )

    except RepoNotSupportedError as e:
        return RepoExtractionResult(
            repo_url=repo_url,
            status=RepoStatus.NOT_SUPPORTED,
            error=str(e),
        )

    except Exception as e:
        if verbose:
            print(f"ERROR processing {repo_url}: {e}")
        return RepoExtractionResult(
            repo_url=repo_url,
            status=RepoStatus.INACCESSIBLE,
            error=str(e),
        )
