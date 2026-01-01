from pathlib import Path
import re
from pydantic import BaseModel, HttpUrl
from typing import List, Union, Tuple
from urllib.parse import urlparse
import os
from .repo_providers import (
    get_repo_cloner,
    RepoStatus,
    RepoExtractionResult,
    RepoNotSupportedError,
)
from .tokenizer import extract_text_from_file, tokenize_text


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

token_cutoff = 3000


def read_all_files(base_path: Path) -> str:
    """
    Reads files in the repo. Those in the blacklist are excluded,
    those in the whitelist are taken in full, and the rest are taken until the token cutoff is reached.
    """
    file_contents = []
    for file_path in base_path.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(base_path)
            rel_path_str = str(rel_path)
            is_readme = bool(
                re.match(r"readme(\.md|\.txt)?", str(rel_path), re.IGNORECASE)
            )
            if file_path.suffix.lower() in block_list:
                continue
            header = f"\n\nFILE: {rel_path_str}\n"
            if is_readme or file_path.suffix.lower() in allow_list:
                content = extract_text_from_file(file_path)
                file_contents.append(header + content)
                continue
            try:
                content = extract_text_from_file(file_path)
                tokens, enc = tokenize_text(content)
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                if len(tokens) > token_cutoff:
                    original_len = len(tokens)
                    content = content[:token_cutoff] + (
                        f"\n\n[... CONTENT TRUNCATED: {original_len:,} chars > {token_cutoff:,} limit ]"
                    )
                    # print(f"-> TRUNCATED: {rel_path} ({original_len:,} chars)")
                file_contents.append(header + content)

            except Exception as e:
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
