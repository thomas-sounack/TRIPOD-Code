import requests
from pathlib import Path
from abc import ABC, abstractmethod
from urllib.parse import urlparse
import os
import re
from huggingface_hub import hf_hub_download
import git
import time

class RepoCloner(ABC):
    """Abstract base class for all repository cloners."""
    @abstractmethod
    def clone(self, repo_url: str, repo_path: Path):
        """Clones or downloads the content from the given URL to the specified path."""
        pass


class DefaultGitCloner(RepoCloner):
    """Cloner for standard Git repositories (GitHub, GitLab, Gitee, etc.)."""
    def clone(self, repo_url: str, repo_path: Path):
        if not repo_path.exists():
            # print(f"Cloning git repo: {repo_url}")
            # Use depth=1 for shallow clone to save time and space
            git.Repo.clone_from(repo_url, repo_path, depth=1)
        else:
            print(f"Repo already exists at {repo_path}. Skipping clone.")


class ZenodoCloner(RepoCloner):
    """Cloner for Zenodo repositories, downloading files via the API."""
    def clone(self, repo_url: str, repo_path: Path):
        if repo_path.exists() and any(repo_path.iterdir()):
             print(f"Zenodo repo already exists and is not empty at {repo_path}. Skipping download.")
             return

        repo_path.mkdir(parents=True, exist_ok=True)
        # print(f"Downloading Zenodo repository: {repo_url}")

        record_id = repo_url.rstrip('/').split('/')[-1]
        api_url = f"https://zenodo.org/api/records/{record_id}"

        try:
            resp = requests.get(api_url)
            resp.raise_for_status()
            data = resp.json()
            files = data.get('files', [])
            if not files:
                print("No files found in Zenodo record.")
                return

            for f in files:
                download_url = f['links']['self']
                file_name = f['key']
                local_path = repo_path / file_name
                local_path.parent.mkdir(parents=True, exist_ok=True)
                # print(f"Downloading {file_name}...")

                with requests.get(download_url, stream=True) as r:
                    r.raise_for_status()
                    with open(local_path, 'wb') as f_out:
                        for chunk in r.iter_content(chunk_size=8192):
                            f_out.write(chunk)

            # print(f"Zenodo repository downloaded to {repo_path}")
        except Exception as e:
            print(f"Failed Zenodo download from {api_url}: {e}")
            raise


class FigshareCloner(RepoCloner):
    """Cloner for Figshare repositories."""
    API_BASE = "https://api.figshare.com/v2/articles/"

    def clone(self, repo_url: str, repo_path: Path):
        if repo_path.exists() and any(repo_path.iterdir()):
             print(f"Figshare repo already exists and is not empty at {repo_path}. Skipping download.")
             return
             
        repo_path.mkdir(parents=True, exist_ok=True)
        # print(f"Downloading Figshare repository: {repo_url}")

        article_id = repo_url.rstrip('/').split('/')[-1]
        api_url = f"{self.API_BASE}{article_id}/files"

        try:
            resp = requests.get(api_url)
            resp.raise_for_status()
            files = resp.json()
            
            if not files:
                print("No files found in Figshare article.")
                return

            for f_item in files:
                file_name = f_item['name']
                download_url = f_item['download_url']
                local_path = repo_path / file_name
                # print(f"Downloading {file_name}...")
                
                with requests.get(download_url, stream=True) as r:
                    r.raise_for_status()
                    with open(local_path, 'wb') as f_out:
                        for chunk in r.iter_content(chunk_size=8192):
                            f_out.write(chunk)
                            
            # print(f"Figshare repository downloaded to {repo_path}")
        except Exception as e:
            print(f"Failed Figshare download from {api_url}: {e}")
            raise


class OSFCloner(RepoCloner):
    """Cloner for OSF.io projects."""
    API_BASE = "https://api.osf.io/v2/"

    def clone(self, repo_url: str, repo_path: Path):
        if repo_path.exists() and any(repo_path.iterdir()):
             print(f"OSF repo already exists and is not empty at {repo_path}. Skipping download.")
             return
             
        repo_path.mkdir(parents=True, exist_ok=True)
        # print(f"Downloading OSF.io project: {repo_url}")

        project_id = repo_url.rstrip('/').split('/')[-1]
        files_api = f"{self.API_BASE}nodes/{project_id}/files/osfstorage/"

        try:
            resp = requests.get(files_api)
            resp.raise_for_status()
            data = resp.json()
            for item in data.get('data', []):
                file_name = item['attributes']['name']
                download_url = item['links']['download']
                local_path = repo_path / file_name
                # print(f"Downloading {file_name}...")
                with requests.get(download_url, stream=True) as r:
                    r.raise_for_status()
                    with open(local_path, 'wb') as f_out:
                        for chunk in r.iter_content(chunk_size=8192):
                            f_out.write(chunk)
            # print(f"OSF.io project downloaded to {repo_path}")
        except Exception as e:
            print(f"Failed OSF.io download from {files_api}: {e}")
            raise


class GenericHTTPCloner(RepoCloner):
    """Fallback for simple HTTP/HTTPS files. Downloads a single file."""
    def clone(self, repo_url: str, repo_path: Path):
        repo_path.mkdir(parents=True, exist_ok=True)
        
        file_name = os.path.basename(urlparse(repo_url).path) or "downloaded_file"
        
        if file_name == "" or file_name == "/":
            file_name = "index.html"
            
        local_file = repo_path / file_name
        
        if local_file.exists():
             print(f"File already exists at {local_file}. Skipping download.")
             return
             
        # print(f"Downloading from HTTP(S): {repo_url} -> {local_file}")

        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with requests.get(repo_url, stream=True, timeout=30) as r:
                        r.raise_for_status()
                        with open(local_file, 'wb') as f_out:
                            for chunk in r.iter_content(chunk_size=8192):
                                f_out.write(chunk)
                        # print(f"Successfully saved {local_file}")
                        return
                except requests.exceptions.RequestException as req_e:
                    if attempt < max_retries - 1:
                        # print(f"Attempt {attempt + 1} failed: {req_e}. Retrying in {2**attempt}s...")
                        time.sleep(2**attempt)
                    else:
                        raise req_e

        except Exception as e:
            print(f"Failed HTTP download for {repo_url}: {e}")
            raise


class QToolCloner(RepoCloner):
    """
    NEW: Cloner for specific Q-tool websites that serve 
    source code via a dedicated '/src.php' endpoint.
    It constructs the correct URL and delegates the download to GenericHTTPCloner.
    """
    def clone(self, repo_url: str, repo_path: Path):
        parsed = urlparse(repo_url)
        new_url = f"{parsed.scheme}://{parsed.netloc}/src.php"
        # print(f"Detected Q-Tool link ({parsed.netloc}). Redirecting download to: {new_url}")

        GenericHTTPCloner().clone(new_url, repo_path)


class DOICloner(RepoCloner):
    """Resolves a DOI link to its final URL and delegates to the correct cloner."""
    def clone(self, repo_url: str, repo_path: Path):
        repo_path.mkdir(parents=True, exist_ok=True)
        # print(f"Resolving DOI: {repo_url}")

        try:
            resp = requests.head(repo_url, allow_redirects=True, timeout=10)
            resp.raise_for_status()
            final_url = resp.url
            # print(f"DOI resolved to: {final_url}")
            cloner = get_repo_cloner(final_url)
            cloner.clone(final_url, repo_path)

        except Exception as e:
            print(f"Failed to resolve DOI or delegate: {e}")
            raise


class HuggingFaceCloner(RepoCloner):
    """Cloner for Hugging Face repositories (primarily used for data/models)."""
    def clone(self, repo_url: str, repo_path: Path):
        repo_path.mkdir(parents=True, exist_ok=True)
        
        parsed = urlparse(repo_url)
        repo_id = parsed.path.strip("/")
        # print(f"Downloading Hugging Face repo: {repo_id}")

        try:
            file_path = hf_hub_download(repo_id=repo_id, filename="README.md", cache_dir=repo_path)
            # print(f"Downloaded README.md to {file_path}. Use hf_hub_download to get other files.")
        except Exception as e:
            print(f"Failed Hugging Face download: {e}. Check if repo_id '{repo_id}' is valid.")

CLONER_MAP = {
    'github.com': DefaultGitCloner,
    'gitlab.com': DefaultGitCloner,
    'gitee.com': DefaultGitCloner,
    'zenodo.org': ZenodoCloner,
    'figshare.com': FigshareCloner,
    'osf.io': OSFCloner,
    'huggingface.co': HuggingFaceCloner,
    'qrisk.org': QToolCloner,
    'www.qrisk.org': QToolCloner,
    'qdiabetes.org': QToolCloner,
    'www.qdiabetes.org': QToolCloner,
}


def get_repo_cloner(repo_url: str) -> RepoCloner:
    """Determines the appropriate RepoCloner subclass for a given URL."""
    parsed_url = urlparse(repo_url)
    domain = parsed_url.netloc
    domain = re.sub(r'^www\.', '', domain).lower()
    
    if 'doi.org' in domain or 'dx.doi.org' in domain:
        return DOICloner()

    if domain in CLONER_MAP:
        return CLONER_MAP[domain]()
    
    if any(s in domain for s in ['git.', '.github.io', '.gitlab.', '.gite.']):
        return DefaultGitCloner()
    
    if parsed_url.scheme in ('http', 'https'):
        return GenericHTTPCloner()
    raise ValueError(f"No specific cloner found for URL domain or format: {domain}")