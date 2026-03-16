import requests
from pathlib import Path
from abc import ABC, abstractmethod
from urllib.parse import urlparse
import os
import re
import git
import shutil
import subprocess
from typing import List
import zipfile
import tarfile


class RepoNotSupportedError(Exception):
    """
    Exception raised when a repository URL is from an unsupported platform.

    :param message: Description of why the repository is not supported.
    :type message: str
    """

    pass


def _ensure_dir(path: Path):
    """
    Ensure a directory exists, creating it and any parents if necessary.

    :param path: The directory path to create.
    :type path: Path
    """
    path.mkdir(parents=True, exist_ok=True)


def extract_file_tree(repo_path: Path) -> List[str]:
    """
    Extract a sorted list of all file paths in a repository.

    :param repo_path: Root path of the repository to scan.
    :type repo_path: Path
    :return: Sorted list of relative file paths as strings.
    :rtype: list[str]

    :Example:

    >>> files = extract_file_tree(Path("/path/to/repo"))
    >>> print(files)
    ['README.md', 'src/main.py', 'tests/test_main.py']
    """
    files = []
    for p in repo_path.rglob("*"):
        if p.is_file():
            files.append(str(p.relative_to(repo_path)))
    return sorted(files)


class RepoCloner(ABC):
    """
    Abstract base class for repository cloners.

    Subclasses must implement the :meth:`clone` method to handle
    repository cloning for specific platforms (e.g., GitHub, Zenodo).
    """

    @abstractmethod
    def clone(self, repo_url: str, base_path: Path) -> Path:
        """
        Clone a repository to a local directory.

        :param repo_url: URL of the repository to clone.
        :type repo_url: str
        :param base_path: Base directory where the repository will be cloned.
        :type base_path: Path
        :return: Path to the cloned repository directory.
        :rtype: Path
        :raises Exception: If cloning fails.
        """
        pass


class DefaultGitCloner(RepoCloner):
    """
    Cloner for standard Git repositories (GitHub, GitLab, Gitee, etc.).

    Uses shallow cloning (depth=1) to minimize download time and disk usage.
    Skips cloning if the repository already exists locally.
    """

    def clone(self, repo_url: str, base_path: Path) -> Path:
        """
        Clone a Git repository using GitPython.

        :param repo_url: URL of the Git repository.
        :type repo_url: str
        :param base_path: Base directory for cloning.
        :type base_path: Path
        :return: Path to the cloned repository.
        :rtype: Path
        """
        parsed = urlparse(repo_url)
        parts = [p for p in parsed.path.split("/") if p]
        repo_name = os.path.splitext(parts[-1])[0] if parts else "unknown_repo"

        repo_path = base_path / repo_name

        if repo_path.exists() and any(repo_path.iterdir()):
            return repo_path

        base_path.mkdir(parents=True, exist_ok=True)
        git.Repo.clone_from(repo_url, repo_path, depth=1)
        return repo_path


class ZenodoCloner(RepoCloner):
    """
    Cloner for Zenodo records.

    Uses the ``zenodo_get`` CLI tool to download record files.
    Automatically extracts ZIP archives and handles nested directory structures.
    """

    def clone(self, repo_url: str, base_path: Path) -> Path:
        """
        Download a Zenodo record.

        :param repo_url: URL of the Zenodo record (e.g., https://zenodo.org/record/12345).
        :type repo_url: str
        :param base_path: Base directory for downloading.
        :type base_path: Path
        :return: Path to the downloaded repository.
        :rtype: Path
        :raises RuntimeError: If zenodo_get fails to download the record.
        """
        record_id = repo_url.rstrip("/").split("/")[-1]
        repo_path = base_path / f"zenodo_{record_id}"

        # If already cloned, do nothing
        if repo_path.exists() and any(repo_path.iterdir()):
            return repo_path

        repo_path.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(
                ["zenodo_get", "-r", record_id, "-o", "."],
                cwd=repo_path,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            subdirs = [p for p in repo_path.iterdir() if p.is_dir()]
            if len(subdirs) == 1 and any(
                f.suffix == ".zip" for f in subdirs[0].iterdir()
            ):
                pass
            else:
                raise RuntimeError(
                    f"zenodo_get failed for record {record_id}: {e.stderr.strip()}"
                )

        # when repo downloaded in subfolder, adjust path:
        subdirs = [p for p in repo_path.iterdir() if p.is_dir()]
        if len(subdirs) == 1:
            repo_path = subdirs[0]

        for zip_path in repo_path.glob("*.zip"):
            with zipfile.ZipFile(zip_path) as zf:
                members = zf.namelist()
                top_levels = {
                    m.split("/")[0]
                    for m in members
                    if "/" in m and not m.startswith("__MACOSX")
                }
                zf.extractall(repo_path)

            if len(top_levels) == 1:
                root = repo_path / next(iter(top_levels))
                if root.exists() and root.is_dir():
                    for item in root.iterdir():
                        shutil.move(str(item), repo_path)
                    shutil.rmtree(root, ignore_errors=True)

            zip_path.unlink()

        return repo_path


class FigshareCloner(RepoCloner):
    """
    Cloner for Figshare articles.

    Uses the Figshare API to fetch article metadata and download all files.
    """

    API_BASE = "https://api.figshare.com/v2/articles/"

    def clone(self, repo_url: str, base_path: Path) -> Path:
        """
        Download all files from a Figshare article.

        :param repo_url: URL of the Figshare article.
        :type repo_url: str
        :param base_path: Base directory for downloading.
        :type base_path: Path
        :return: Path to the downloaded files directory.
        :rtype: Path
        :raises requests.HTTPError: If API requests fail.
        """
        article_id = repo_url.rstrip("/").split("/")[-1]
        meta = requests.get(f"{self.API_BASE}{article_id}").json()
        title = meta.get("title", f"figshare_{article_id}")

        safe_title = re.sub(r"[^a-zA-Z0-9._-]+", "_", title).strip("_")
        repo_path = base_path / safe_title

        # Check if already cloned
        if repo_path.exists() and any(repo_path.iterdir()):
            return repo_path

        repo_path.mkdir(parents=True, exist_ok=True)

        files = requests.get(f"{self.API_BASE}{article_id}/files").json()
        for f in files:
            path = repo_path / f["name"]
            with requests.get(f["download_url"], stream=True) as r:
                r.raise_for_status()
                with open(path, "wb") as out:
                    for chunk in r.iter_content(8192):
                        out.write(chunk)

        return repo_path


class OSFCloner(RepoCloner):
    """
    Cloner for Open Science Framework (OSF) projects.

    Uses the ``osf`` CLI tool to clone project files.
    Flattens the osfstorage directory structure after cloning.
    """

    def clone(self, repo_url: str, base_path: Path) -> Path:
        """
        Clone an OSF project.

        :param repo_url: URL of the OSF project (e.g., https://osf.io/abc123).
        :type repo_url: str
        :param base_path: Base directory for cloning.
        :type base_path: Path
        :return: Path to the cloned project.
        :rtype: Path
        :raises RuntimeError: If the OSF project is inaccessible.
        """
        project_id = repo_url.rstrip("/").split("/")[-1]
        repo_path = base_path / project_id

        # Skip cloning if already done
        if repo_path.exists() and any(repo_path.iterdir()):
            return repo_path

        repo_path.mkdir(parents=True, exist_ok=True)

        # Clone project (always creates osfstorage/)
        try:
            subprocess.run(
                ["osf", "-p", project_id, "clone", str(repo_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            # Project inaccessible, private, deleted, or network failure
            raise RuntimeError(f"OSF project {project_id} inaccessible") from e

        osf_storage = repo_path / "osfstorage"

        # Empty but accessible project
        if not osf_storage.exists():
            return repo_path

        # Move osfstorage/* → repo_path/*
        for item in osf_storage.iterdir():
            shutil.move(str(item), repo_path)

        # Remove osfstorage wrapper
        shutil.rmtree(osf_storage, ignore_errors=True)

        return repo_path


class DOICloner(RepoCloner):
    """
    Cloner that resolves DOI URLs and delegates to the appropriate cloner.

    Follows DOI redirects to determine the final repository URL,
    then uses :func:`get_repo_cloner` to select the correct cloner.
    """

    def clone(self, repo_url: str, base_path: Path) -> Path:
        """
        Resolve a DOI and clone the target repository.

        :param repo_url: DOI URL (e.g., https://doi.org/10.1234/example).
        :type repo_url: str
        :param base_path: Base directory for cloning.
        :type base_path: Path
        :return: Path to the cloned repository.
        :rtype: Path
        :raises requests.HTTPError: If DOI resolution fails.
        :raises RepoNotSupportedError: If the resolved URL is not supported.
        """
        resp = requests.head(repo_url, allow_redirects=True)
        resp.raise_for_status()

        final_url = resp.url
        cloner = get_repo_cloner(final_url)

        return cloner.clone(final_url, base_path)


CLONER_MAP = {
    "github.com": DefaultGitCloner,
    "gitlab.com": DefaultGitCloner,
    "gitee.com": DefaultGitCloner,
    "zenodo.org": ZenodoCloner,
    "figshare.com": FigshareCloner,
    "osf.io": OSFCloner,
}


def get_repo_cloner(repo_url: str) -> RepoCloner:
    """
    Get the appropriate cloner for a repository URL.

    Analyzes the URL domain to determine which :class:`RepoCloner`
    subclass should handle the repository.

    :param repo_url: URL of the repository.
    :type repo_url: str
    :return: An instance of the appropriate RepoCloner subclass.
    :rtype: RepoCloner
    :raises RepoNotSupportedError: If no cloner is available for the URL domain.

    :Example:

    >>> cloner = get_repo_cloner("https://github.com/user/repo")
    >>> isinstance(cloner, DefaultGitCloner)
    True

    Supported platforms:
        - GitHub, GitLab, Gitee (DefaultGitCloner)
        - Zenodo (ZenodoCloner)
        - Figshare (FigshareCloner)
        - OSF (OSFCloner)
        - DOI URLs (DOICloner, which resolves and delegates)
    """
    parsed_url = urlparse(repo_url)
    domain = parsed_url.netloc
    domain = re.sub(r"^www\.", "", domain).lower()

    if "doi.org" in domain or "dx.doi.org" in domain:
        return DOICloner()

    if domain in CLONER_MAP:
        return CLONER_MAP[domain]()

    if any(s in domain for s in ["git.", "gitlab"]):
        return DefaultGitCloner()

    raise RepoNotSupportedError(
        f"No specific cloner found for URL domain or format: {domain}"
    )
