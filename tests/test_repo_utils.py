"""
Tests for src/repo_utils module.

Tests repository extraction, cloning, and text processing utilities.
Uses mocked external calls for network operations.
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from repo_utils.main_extractor import (
    clone_and_extract_tree,
    RepoStatus,
    RepoExtractionResult,
    RepoTree,
    RepoInfo,
    get_tree,
    read_all_files,
    block_list,
    allow_list,
    TOKEN_CUTOFF_PER_FILE,
)
from repo_utils.repo_providers import (
    get_repo_cloner,
    RepoNotSupportedError,
    DefaultGitCloner,
    ZenodoCloner,
    FigshareCloner,
    OSFCloner,
    DOICloner,
    extract_file_tree,
)
from repo_utils.tokenizer import extract_text_from_file, tokenize_text


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_repo(tmp_path):
    """Create a temporary repository structure for testing."""
    # Create directories
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "data").mkdir()

    # Create files
    (tmp_path / "README.md").write_text("# Test Repository\nThis is a test.")
    (tmp_path / "requirements.txt").write_text("numpy==1.21.0\npandas>=1.3.0")
    (tmp_path / "src" / "main.py").write_text('def hello():\n    print("Hello")\n')
    (tmp_path / "src" / "utils.py").write_text(
        "# Utility functions\ndef add(a, b):\n    return a + b\n"
    )
    (tmp_path / "tests" / "test_main.py").write_text(
        "def test_hello():\n    assert True\n"
    )
    (tmp_path / "data" / "sample.csv").write_text("a,b,c\n1,2,3\n4,5,6\n")

    return tmp_path


@pytest.fixture
def temp_repo_with_various_files(tmp_path):
    """Create a repo with various file types for testing block/allow lists."""
    # Code files (allowlist)
    (tmp_path / "script.py").write_text("print('hello')")
    (tmp_path / "app.js").write_text("console.log('hello')")
    (tmp_path / "analysis.r").write_text("print('hello')")

    # Binary files (blocklist)
    (tmp_path / "model.pt").write_bytes(b"\x00\x01\x02")
    (tmp_path / "data.pkl").write_bytes(b"\x00\x01\x02")
    (tmp_path / "archive.zip").write_bytes(b"PK\x03\x04")

    # Regular text files (with cutoff)
    (tmp_path / "notes.txt").write_text("Some notes here")
    (tmp_path / "config.yaml").write_text("key: value")

    # Hidden files (should be ignored)
    (tmp_path / ".gitignore").write_text("*.pyc\n__pycache__/")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "secret.txt").write_text("secret")

    return tmp_path


# ============================================================================
# Tests for tokenizer.py
# ============================================================================


class TestTokenizer:
    def test_tokenize_text_returns_tokens_and_encoder(self):
        """Test that tokenize_text returns tokens and encoder."""
        text = "Hello, world! This is a test."
        tokens, enc = tokenize_text(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert enc is not None

    def test_tokenize_text_encoding_name(self):
        """Test tokenizing with different encoding."""
        text = "Test text"
        tokens, enc = tokenize_text(text, encoding_name="cl100k_base")
        assert len(tokens) > 0

    def test_tokenize_text_roundtrip(self):
        """Test that tokenized text can be decoded back."""
        original_text = "The quick brown fox jumps over the lazy dog."
        tokens, enc = tokenize_text(original_text)
        decoded = enc.decode(tokens)
        assert decoded == original_text

    def test_extract_text_from_python_file(self, tmp_path):
        """Test extracting text from a Python file."""
        py_file = tmp_path / "test.py"
        content = "def main():\n    print('Hello')\n"
        py_file.write_text(content)

        result = extract_text_from_file(py_file)
        assert result == content

    def test_extract_text_from_csv_file(self, tmp_path):
        """Test extracting text from a CSV file."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25\n")

        result = extract_text_from_file(csv_file)
        assert "Alice" in result
        assert "Bob" in result

    def test_extract_text_from_json_file(self, tmp_path):
        """Test extracting text from a JSON file."""
        json_file = tmp_path / "config.json"
        data = {"key": "value", "number": 42}
        json_file.write_text(json.dumps(data))

        result = extract_text_from_file(json_file)
        assert "key" in result
        assert "value" in result

    def test_extract_text_empty_on_binary_content(self, tmp_path):
        """Test that binary content returns empty string."""
        binary_file = tmp_path / "binary.dat"
        binary_file.write_bytes(b"\x00\x01\x02\x00\x03\x04")

        result = extract_text_from_file(binary_file)
        assert result == ""

    def test_extract_text_handles_unicode(self, tmp_path):
        """Test handling of Unicode content."""
        unicode_file = tmp_path / "unicode.txt"
        unicode_file.write_text("Hello 世界 🌍", encoding="utf-8")

        result = extract_text_from_file(unicode_file)
        assert "世界" in result
        assert "🌍" in result


# ============================================================================
# Tests for repo_providers.py
# ============================================================================


class TestGetRepoCloner:
    def test_github_returns_git_cloner(self):
        """Test that GitHub URLs return DefaultGitCloner."""
        cloner = get_repo_cloner("https://github.com/user/repo")
        assert isinstance(cloner, DefaultGitCloner)

    def test_gitlab_returns_git_cloner(self):
        """Test that GitLab URLs return DefaultGitCloner."""
        cloner = get_repo_cloner("https://gitlab.com/user/repo")
        assert isinstance(cloner, DefaultGitCloner)

    def test_gitee_returns_git_cloner(self):
        """Test that Gitee URLs return DefaultGitCloner."""
        cloner = get_repo_cloner("https://gitee.com/user/repo")
        assert isinstance(cloner, DefaultGitCloner)

    def test_zenodo_returns_zenodo_cloner(self):
        """Test that Zenodo URLs return ZenodoCloner."""
        cloner = get_repo_cloner("https://zenodo.org/record/12345")
        assert isinstance(cloner, ZenodoCloner)

    def test_figshare_returns_figshare_cloner(self):
        """Test that Figshare URLs return FigshareCloner."""
        cloner = get_repo_cloner("https://figshare.com/articles/12345")
        assert isinstance(cloner, FigshareCloner)

    def test_osf_returns_osf_cloner(self):
        """Test that OSF URLs return OSFCloner."""
        cloner = get_repo_cloner("https://osf.io/abc123")
        assert isinstance(cloner, OSFCloner)

    def test_doi_returns_doi_cloner(self):
        """Test that DOI URLs return DOICloner."""
        cloner = get_repo_cloner("https://doi.org/10.1234/test")
        assert isinstance(cloner, DOICloner)

    def test_www_prefix_stripped(self):
        """Test that www. prefix is properly handled."""
        cloner = get_repo_cloner("https://www.github.com/user/repo")
        assert isinstance(cloner, DefaultGitCloner)

    def test_unsupported_domain_raises_error(self):
        """Test that unsupported domains raise RepoNotSupportedError."""
        with pytest.raises(RepoNotSupportedError):
            get_repo_cloner("https://unknownsite.com/repo")

    def test_custom_gitlab_instance(self):
        """Test that custom GitLab instances are recognized."""
        cloner = get_repo_cloner("https://gitlab.example.com/user/repo")
        assert isinstance(cloner, DefaultGitCloner)


class TestExtractFileTree:
    def test_extract_file_tree(self, temp_repo):
        """Test extracting file tree from a repository."""
        files = extract_file_tree(temp_repo)
        assert "README.md" in files
        assert "requirements.txt" in files
        assert "src/main.py" in files


class TestDefaultGitCloner:
    @patch("repo_utils.repo_providers.git.Repo")
    def test_clone_creates_directory(self, mock_git_repo, tmp_path):
        """Test that cloning creates the expected directory."""
        cloner = DefaultGitCloner()
        repo_url = "https://github.com/user/test-repo"

        result = cloner.clone(repo_url, tmp_path)

        assert result == tmp_path / "test-repo"
        mock_git_repo.clone_from.assert_called_once()

    @patch("repo_utils.repo_providers.git.Repo")
    def test_clone_skips_existing_repo(self, mock_git_repo, tmp_path):
        """Test that cloning skips if repo already exists."""
        # Create existing repo directory
        existing_repo = tmp_path / "existing-repo"
        existing_repo.mkdir()
        (existing_repo / "file.txt").write_text("content")

        cloner = DefaultGitCloner()
        result = cloner.clone("https://github.com/user/existing-repo", tmp_path)

        assert result == existing_repo
        mock_git_repo.clone_from.assert_not_called()


class TestZenodoCloner:
    @patch("subprocess.run")
    def test_clone_calls_zenodo_get(self, mock_run, tmp_path):
        """Test that Zenodo cloner calls zenodo_get."""
        mock_run.return_value = MagicMock(returncode=0)

        cloner = ZenodoCloner()
        repo_url = "https://zenodo.org/record/12345"

        result = cloner.clone(repo_url, tmp_path)

        assert result == tmp_path / "zenodo_12345"
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "zenodo_get" in call_args


class TestFigshareCloner:
    @patch("requests.get")
    def test_clone_fetches_article_files(self, mock_get, tmp_path):
        """Test that Figshare cloner fetches article files."""
        # Mock metadata response
        meta_response = MagicMock()
        meta_response.json.return_value = {"title": "Test Article"}

        # Mock files response
        files_response = MagicMock()
        files_response.json.return_value = []

        mock_get.side_effect = [meta_response, files_response]

        cloner = FigshareCloner()
        result = cloner.clone("https://figshare.com/articles/12345", tmp_path)

        assert "Test_Article" in str(result) or "12345" in str(result)


class TestDOICloner:
    @patch("requests.head")
    def test_doi_resolves_to_github(self, mock_head, tmp_path):
        """Test that DOI resolver redirects to GitHub."""
        mock_response = MagicMock()
        mock_response.url = "https://github.com/user/repo"
        mock_head.return_value = mock_response

        cloner = DOICloner()

        # Patch the nested cloner call
        with patch.object(DefaultGitCloner, "clone") as mock_clone:
            mock_clone.return_value = tmp_path / "repo"
            cloner.clone("https://doi.org/10.1234/test", tmp_path)

        mock_head.assert_called_once_with(
            "https://doi.org/10.1234/test", allow_redirects=True
        )


# ============================================================================
# Tests for main_extractor.py
# ============================================================================


class TestRepoTree:
    def test_create_file_node(self):
        """Test creating a file node in RepoTree."""
        node = RepoTree(name="file.py")
        assert node.name == "file.py"
        assert node.children is None

    def test_create_directory_node(self):
        """Test creating a directory node with children."""
        children = [
            RepoTree(name="file1.py"),
            RepoTree(name="file2.py"),
        ]
        node = RepoTree(name="src", children=children)
        assert node.name == "src"
        assert len(node.children) == 2


class TestRepoExtractionResult:
    def test_create_ok_result(self):
        """Test creating an OK extraction result."""
        result = RepoExtractionResult(
            repo_url="https://github.com/user/repo",
            status=RepoStatus.OK,
            repo_path="/path/to/repo",
            output="Repository content here",
        )
        assert result.status == RepoStatus.OK
        assert result.error is None

    def test_create_error_result(self):
        """Test creating an error extraction result."""
        result = RepoExtractionResult(
            repo_url="https://github.com/user/repo",
            status=RepoStatus.INACCESSIBLE,
            error="Connection timeout",
        )
        assert result.status == RepoStatus.INACCESSIBLE
        assert result.error == "Connection timeout"


class TestGetTree:
    def test_get_tree_simple_structure(self, temp_repo):
        """Test getting tree for a simple repository structure."""
        tree = get_tree(temp_repo)

        # Convert to names for easier testing
        names = [node.name for node in tree]
        assert "README.md" in names
        assert "requirements.txt" in names
        assert "src" in names

    def test_get_tree_ignores_hidden_files(self, temp_repo_with_various_files):
        """Test that hidden files/directories are ignored."""
        tree = get_tree(temp_repo_with_various_files)
        names = [node.name for node in tree]

        assert ".gitignore" not in names
        assert ".hidden" not in names

    def test_get_tree_includes_subdirectories(self, temp_repo):
        """Test that subdirectories have children."""
        tree = get_tree(temp_repo)
        src_node = next((n for n in tree if n.name == "src"), None)

        assert src_node is not None
        assert src_node.children is not None
        child_names = [c.name for c in src_node.children]
        assert "main.py" in child_names


class TestReadAllFiles:
    def test_read_all_files_basic(self, temp_repo):
        """Test reading all files from a repository."""
        result = read_all_files(temp_repo, verbose=False, context_window=100000)

        assert "FILE: README.md" in result
        assert "Test Repository" in result
        assert "FILE: src/main.py" in result

    def test_read_all_files_skips_blocklist(self, temp_repo_with_various_files):
        """Test that blocklisted files are skipped."""
        result = read_all_files(
            temp_repo_with_various_files, verbose=False, context_window=100000
        )

        assert "model.pt" not in result
        assert "data.pkl" not in result
        assert "archive.zip" not in result

    def test_read_all_files_includes_allowlist(self, temp_repo_with_various_files):
        """Test that allowlisted files are included."""
        result = read_all_files(
            temp_repo_with_various_files, verbose=False, context_window=100000
        )

        assert "print('hello')" in result  # Python content
        assert "console.log" in result  # JS content

    def test_read_all_files_respects_context_window(self, tmp_path):
        """Test that content is truncated at context window."""
        # Create a large file
        large_content = "x" * 100000
        (tmp_path / "large.txt").write_text(large_content)

        result = read_all_files(tmp_path, verbose=False, context_window=1000)

        # Should be truncated
        assert "TRUNCATED" in result

    def test_read_all_files_readme_priority(self, tmp_path):
        """Test that README files are included fully."""
        readme_content = "# Important README\n" + "Documentation " * 1000
        (tmp_path / "README.md").write_text(readme_content)

        result = read_all_files(tmp_path, verbose=False, context_window=500000)

        assert "Important README" in result


class TestCloneAndExtractTree:
    @patch("repo_utils.main_extractor.get_repo_cloner")
    def test_clone_and_extract_success(self, mock_get_cloner, temp_repo, tmp_path):
        """Test successful clone and extraction."""
        # Setup mock cloner
        mock_cloner = MagicMock()
        mock_cloner.clone.return_value = temp_repo
        mock_get_cloner.return_value = mock_cloner

        result = clone_and_extract_tree(
            repo_url="https://github.com/user/repo",
            context_window=100000,
            clone_dir=str(tmp_path),
            verbose=False,
        )

        assert result.status == RepoStatus.OK
        assert result.output is not None
        assert "Repository tree" in result.output
        assert "README.md" in result.output

    @patch("repo_utils.main_extractor.get_repo_cloner")
    def test_clone_and_extract_empty_repo(self, mock_get_cloner, tmp_path):
        """Test extraction of empty repository."""
        # Create empty directory
        empty_repo = tmp_path / "empty"
        empty_repo.mkdir()

        mock_cloner = MagicMock()
        mock_cloner.clone.return_value = empty_repo
        mock_get_cloner.return_value = mock_cloner

        result = clone_and_extract_tree(
            repo_url="https://github.com/user/empty-repo",
            context_window=100000,
            clone_dir=str(tmp_path),
        )

        assert result.status == RepoStatus.EMPTY

    @patch("repo_utils.main_extractor.get_repo_cloner")
    def test_clone_and_extract_not_supported(self, mock_get_cloner, tmp_path):
        """Test handling of unsupported repository type."""
        mock_get_cloner.side_effect = RepoNotSupportedError("Unsupported platform")

        result = clone_and_extract_tree(
            repo_url="https://unsupported.com/repo",
            context_window=100000,
            clone_dir=str(tmp_path),
        )

        assert result.status == RepoStatus.NOT_SUPPORTED
        assert "Unsupported" in result.error

    @patch("repo_utils.main_extractor.get_repo_cloner")
    def test_clone_and_extract_inaccessible(self, mock_get_cloner, tmp_path):
        """Test handling of inaccessible repository."""
        mock_cloner = MagicMock()
        mock_cloner.clone.side_effect = Exception("Network error")
        mock_get_cloner.return_value = mock_cloner

        result = clone_and_extract_tree(
            repo_url="https://github.com/user/private-repo",
            context_window=100000,
            clone_dir=str(tmp_path),
        )

        assert result.status == RepoStatus.INACCESSIBLE
        assert "Network error" in result.error


class TestBlockAndAllowLists:
    def test_block_list_contains_expected_extensions(self):
        """Test that block list contains common binary extensions."""
        assert ".pt" in block_list
        assert ".pkl" in block_list
        assert ".zip" in block_list
        assert ".h5" in block_list

    def test_allow_list_contains_code_extensions(self):
        """Test that allow list contains code file extensions."""
        assert ".py" in allow_list
        assert ".js" in allow_list
        assert ".r" in allow_list
        assert ".java" in allow_list
        assert ".ipynb" in allow_list


class TestRepoStatus:
    def test_status_enum_values(self):
        """Test RepoStatus enum values."""
        assert RepoStatus.OK.value == "ok"
        assert RepoStatus.EMPTY.value == "empty"
        assert RepoStatus.INACCESSIBLE.value == "inaccessible"
        assert RepoStatus.NOT_SUPPORTED.value == "not_supported"


class TestTokenCutoff:
    def test_token_cutoff_per_file_is_reasonable(self):
        """Test that token cutoff is set to a reasonable value."""
        assert TOKEN_CUTOFF_PER_FILE == 3000

    def test_large_file_gets_truncated(self, tmp_path):
        """Test that large non-allowlisted files get truncated."""
        # Create a large text file (not in allowlist)
        large_content = "word " * 10000  # Should exceed token cutoff
        (tmp_path / "large_document.txt").write_text(large_content)

        result = read_all_files(tmp_path, verbose=False, context_window=1000000)

        # File should be included but truncated
        assert "FILE: large_document.txt" in result
        assert "TRUNCATED" in result


# ============================================================================
# Integration tests
# ============================================================================


class TestIntegration:
    def test_full_extraction_workflow(self, temp_repo):
        """Test the full workflow from tree to text extraction."""
        # Get tree
        tree = get_tree(temp_repo)
        assert len(tree) > 0

        # Read all files
        content = read_all_files(temp_repo, verbose=False, context_window=100000)
        assert len(content) > 0

        # Create RepoInfo
        repo_info = RepoInfo(url="https://github.com/test/repo", tree=tree)
        json_output = repo_info.model_dump_json(indent=2)
        assert "tree" in json_output

    def test_tokenization_consistency(self):
        """Test that tokenization is consistent across calls."""
        text = "This is a test sentence for tokenization."
        tokens1, _ = tokenize_text(text)
        tokens2, _ = tokenize_text(text)
        assert tokens1 == tokens2

    def test_file_extraction_empty_file(self, tmp_path):
        """Test handling of empty files."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        result = extract_text_from_file(empty_file)
        assert result == ""

    def test_nested_directory_structure(self, tmp_path):
        """Test handling of deeply nested directories."""
        # Create nested structure
        deep_path = tmp_path / "a" / "b" / "c" / "d"
        deep_path.mkdir(parents=True)
        (deep_path / "deep_file.py").write_text("# Deep file")

        tree = get_tree(tmp_path)
        content = read_all_files(tmp_path, verbose=False, context_window=100000)

        assert "deep_file.py" in content
