"""
Tests for src/llm_utils module.

Uses mocked OpenAI API calls to test without actual API requests.
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
from pydantic import BaseModel, Field
from typing import Optional, List

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_utils.llm_wrapper import LLM_wrapper
from llm_utils.structs import PaperAssessment, RepoAssessment, CodeStatementLocation
from llm_utils.paper_assessment_prompt import PAPER_ASSESSMENT_PROMPT
from llm_utils.repo_assessment_prompt import REPO_ASSESSMENT_PROMPT


# ============================================================================
# Fixtures
# ============================================================================


class SimpleOutput(BaseModel):
    """Simple output format for testing."""

    is_valid: bool = Field(..., description="Whether valid")
    reason: str = Field(..., description="Reason for the assessment")
    score: Optional[int] = Field(None, description="Optional score")


class NullableOutput(BaseModel):
    """Output format that allows None values for testing generate_nan_output."""

    is_valid: Optional[bool] = Field(None, description="Whether valid")
    reason: Optional[str] = Field(None, description="Reason for the assessment")
    score: Optional[int] = Field(None, description="Optional score")


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    with patch("llm_utils.llm_wrapper.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        yield mock_client


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a sample CSV file for testing."""
    df = pd.DataFrame(
        {
            "text": ["Sample text 1", "Sample text 2", "Sample text 3"],
            "id": [1, 2, 3],
        }
    )
    file_path = tmp_path / "test_input.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


@pytest.fixture
def sample_parquet_file(tmp_path):
    """Create a sample Parquet file for testing."""
    df = pd.DataFrame(
        {
            "content": ["Content A", "Content B"],
            "meta": ["meta1", "meta2"],
        }
    )
    file_path = tmp_path / "test_input.parquet"
    df.to_parquet(file_path)
    return str(file_path)


# ============================================================================
# Tests for PaperAssessment struct
# ============================================================================


class TestPaperAssessment:
    def test_create_valid_paper_assessment(self):
        """Test creating a valid PaperAssessment instance."""
        assessment = PaperAssessment(
            is_match=True,
            reason="Contains a multivariable prediction model",
            country_first_author_institution="United States",
            repo_url="https://github.com/example/repo",
            code_statement_locations=["methods", "data_availability_section"],
            code_statement_sentence="The code can be found here:",
        )
        assert assessment.is_match is True
        assert assessment.repo_url == "https://github.com/example/repo"
        assert "methods" in assessment.code_statement_locations

    def test_paper_assessment_with_none_repo(self):
        """Test PaperAssessment when no repo is provided."""
        assessment = PaperAssessment(
            is_match=False,
            reason="Does not meet criteria",
            country_first_author_institution="Germany",
            repo_url=None,
            code_statement_locations=None,
            code_statement_sentence=None,
        )
        assert assessment.is_match is False
        assert assessment.repo_url is None

    def test_paper_assessment_appendix_repo(self):
        """Test PaperAssessment with Appendix as repo URL."""
        assessment = PaperAssessment(
            is_match=True,
            reason="Model in supplementary",
            country_first_author_institution="Japan",
            repo_url="Appendix",
            code_statement_locations=["supplementary_material"],
            code_statement_sentence="Code available in supplementary materials",
        )
        assert assessment.repo_url == "Appendix"


# ============================================================================
# Tests for RepoAssessment struct
# ============================================================================


class TestRepoAssessment:
    def test_create_valid_repo_assessment(self):
        """Test creating a valid RepoAssessment instance."""
        assessment = RepoAssessment(
            is_empty=False,
            contains_readme=True,
            readme_purpose_and_outputs=True,
            contains_requirements=True,
            requirements_dependency_versions=True,
            contains_license=True,
            sufficient_code_documentation=True,
            is_modular_and_structured=True,
            implements_tests=False,
            fixes_seed_if_stochastic=True,
            lists_hardware_requirements=False,
            contains_link_to_paper=True,
            contains_citation=False,
            includes_data_or_sample=True,
            comments_and_explanations="Well-structured repository",
            coding_languages=["python", "r"],
        )
        assert assessment.is_empty is False
        assert assessment.contains_readme is True
        assert "python" in assessment.coding_languages

    def test_repo_assessment_empty_repo(self):
        """Test RepoAssessment for an empty repository."""
        assessment = RepoAssessment(
            is_empty=True,
            contains_readme=False,
            readme_purpose_and_outputs=None,
            contains_requirements=False,
            requirements_dependency_versions=None,
            contains_license=False,
            sufficient_code_documentation=False,
            is_modular_and_structured=False,
            implements_tests=False,
            fixes_seed_if_stochastic=None,
            lists_hardware_requirements=False,
            contains_link_to_paper=False,
            contains_citation=False,
            includes_data_or_sample=False,
            comments_and_explanations="Repository is empty",
            coding_languages=None,
        )
        assert assessment.is_empty is True
        assert assessment.readme_purpose_and_outputs is None


# ============================================================================
# Tests for LLM_wrapper initialization
# ============================================================================


class TestLLMWrapperInit:
    def test_init_creates_client(self, mock_openai_client):
        """Test that initialization creates an OpenAI client."""
        wrapper = LLM_wrapper(
            model_name="gpt-4o",
            system_prompt="You are a helpful assistant.",
            output_format_class=SimpleOutput,
        )
        assert wrapper.model_name == "gpt-4o"
        assert wrapper.system_prompt == "You are a helpful assistant."
        assert wrapper.output_format_class == SimpleOutput

    def test_init_with_paper_assessment(self, mock_openai_client):
        """Test initialization with PaperAssessment output format."""
        wrapper = LLM_wrapper(
            model_name="gpt-4o-mini",
            system_prompt=PAPER_ASSESSMENT_PROMPT,
            output_format_class=PaperAssessment,
        )
        assert wrapper.output_format_class == PaperAssessment


# ============================================================================
# Tests for generate_nan_output
# ============================================================================


class TestGenerateNanOutput:
    def test_generate_nan_output_nullable(self, mock_openai_client):
        """Test generating NaN output for NullableOutput (all Optional fields)."""
        wrapper = LLM_wrapper(
            model_name="gpt-4o",
            system_prompt="Test",
            output_format_class=NullableOutput,
        )
        nan_output = wrapper.generate_nan_output()
        assert nan_output.is_valid is None
        assert nan_output.reason is None
        assert nan_output.score is None

    def test_generate_nan_output_raises_for_required_fields(self, mock_openai_client):
        """Test that generate_nan_output raises ValidationError for non-Optional fields."""
        wrapper = LLM_wrapper(
            model_name="gpt-4o",
            system_prompt="Test",
            output_format_class=SimpleOutput,
        )
        # SimpleOutput has required (non-Optional) fields, so None values will fail
        with pytest.raises(Exception):  # Pydantic ValidationError
            wrapper.generate_nan_output()


# ============================================================================
# Tests for assess_one_row with mocked API
# ============================================================================


class TestAssessOneRow:
    def test_assess_one_row_success(self, mock_openai_client):
        """Test successful API call returns parsed output."""
        # Create mock response
        mock_response = MagicMock()
        mock_response.output_parsed = SimpleOutput(
            is_valid=True, reason="Valid input", score=85
        )
        mock_openai_client.responses.parse.return_value = mock_response

        wrapper = LLM_wrapper(
            model_name="gpt-4o",
            system_prompt="Analyze this",
            output_format_class=SimpleOutput,
        )
        result = wrapper.assess_one_row("Test input")

        assert result.is_valid is True
        assert result.reason == "Valid input"
        assert result.score == 85

    def test_assess_one_row_api_call_structure(self, mock_openai_client):
        """Test that API is called with correct parameters."""
        mock_response = MagicMock()
        mock_response.output_parsed = SimpleOutput(
            is_valid=True, reason="test", score=50
        )
        mock_openai_client.responses.parse.return_value = mock_response

        wrapper = LLM_wrapper(
            model_name="gpt-4o-mini",
            system_prompt="System prompt here",
            output_format_class=SimpleOutput,
        )
        wrapper.assess_one_row("User prompt here")

        mock_openai_client.responses.parse.assert_called_once()
        call_kwargs = mock_openai_client.responses.parse.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["text_format"] == SimpleOutput
        assert call_kwargs["truncation"] == "auto"
        assert len(call_kwargs["input"]) == 2
        assert call_kwargs["input"][0]["role"] == "system"
        assert call_kwargs["input"][0]["content"] == "System prompt here"
        assert call_kwargs["input"][1]["role"] == "user"
        assert call_kwargs["input"][1]["content"] == "User prompt here"

    def test_assess_one_row_retry_on_failure(self, mock_openai_client):
        """Test that API call retries on failure."""
        mock_response = MagicMock()
        mock_response.output_parsed = SimpleOutput(
            is_valid=True, reason="Success after retry", score=100
        )

        # First call fails, second succeeds
        mock_openai_client.responses.parse.side_effect = [
            Exception("API Error"),
            mock_response,
        ]

        wrapper = LLM_wrapper(
            model_name="gpt-4o",
            system_prompt="Test",
            output_format_class=SimpleOutput,
        )

        with patch("time.sleep"):  # Skip sleep during test
            result = wrapper.assess_one_row("Test")

        assert result.is_valid is True
        assert result.reason == "Success after retry"
        assert mock_openai_client.responses.parse.call_count == 2

    def test_assess_one_row_returns_none_after_max_retries(self, mock_openai_client):
        """Test that None is returned after max retries exhausted.

        Note: The actual implementation returns None (implicit) when all retries
        fail because the check `if attempt == max_retries` is never true in
        range(max_retries).
        """
        mock_openai_client.responses.parse.side_effect = Exception("Persistent failure")

        wrapper = LLM_wrapper(
            model_name="gpt-4o",
            system_prompt="Test",
            output_format_class=SimpleOutput,
        )

        with patch("time.sleep"):  # Skip sleep during test
            result = wrapper.assess_one_row("Test", max_retries=3)

        # After retries exhausted, function returns None
        assert result is None


# ============================================================================
# Tests for load_input_file
# ============================================================================


class TestLoadInputFile:
    def test_load_csv_file(self, sample_csv_file):
        """Test loading a CSV file."""
        df = LLM_wrapper.load_input_file(
            input_file_path=sample_csv_file,
            text_column="text",
            debug_mode=False,
        )
        assert len(df) == 3
        assert "text" in df.columns
        assert df["text"].iloc[0] == "Sample text 1"

    def test_load_parquet_file(self, sample_parquet_file):
        """Test loading a Parquet file."""
        df = LLM_wrapper.load_input_file(
            input_file_path=sample_parquet_file,
            text_column="content",
            debug_mode=False,
        )
        assert len(df) == 2
        assert "content" in df.columns

    def test_load_debug_mode(self, tmp_path):
        """Test that debug mode limits to 10 rows."""
        df = pd.DataFrame(
            {
                "text": [f"Row {i}" for i in range(50)],
            }
        )
        file_path = tmp_path / "large.csv"
        df.to_csv(file_path, index=False)

        result = LLM_wrapper.load_input_file(
            input_file_path=str(file_path),
            text_column="text",
            debug_mode=True,
        )
        assert len(result) == 10

    def test_load_missing_column_raises_error(self, sample_csv_file):
        """Test that missing required column raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            LLM_wrapper.load_input_file(
                input_file_path=sample_csv_file,
                text_column="nonexistent_column",
                debug_mode=False,
            )
        assert "Missing required column" in str(exc_info.value)


# ============================================================================
# Tests for assess_dataframe with mocked API
# ============================================================================


class TestAssessDataframe:
    def test_assess_dataframe_processes_all_rows(self, mock_openai_client, tmp_path):
        """Test that assess_dataframe processes all rows."""
        # Create test input
        input_df = pd.DataFrame(
            {
                "text": ["Text 1", "Text 2", "Text 3"],
            }
        )
        input_file = tmp_path / "input.csv"
        input_df.to_csv(input_file, index=False)

        # Mock API responses
        mock_response = MagicMock()
        mock_response.output_parsed = SimpleOutput(
            is_valid=True, reason="Valid", score=90
        )
        mock_openai_client.responses.parse.return_value = mock_response

        wrapper = LLM_wrapper(
            model_name="gpt-4o",
            system_prompt="Test",
            output_format_class=SimpleOutput,
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = wrapper.assess_dataframe(
            input_file_path=str(input_file),
            text_column="text",
            output_dir=str(output_dir),
            save_ckpt_every=10,
        )

        assert len(result) == 3
        assert "is_valid" in result.columns
        assert all(result["is_valid"] == True)
        assert mock_openai_client.responses.parse.call_count == 3

    def test_assess_dataframe_creates_checkpoint(self, mock_openai_client, tmp_path):
        """Test that checkpoint file is created."""
        input_df = pd.DataFrame({"text": ["Text 1"]})
        input_file = tmp_path / "input.csv"
        input_df.to_csv(input_file, index=False)

        mock_response = MagicMock()
        mock_response.output_parsed = SimpleOutput(
            is_valid=True, reason="Valid", score=90
        )
        mock_openai_client.responses.parse.return_value = mock_response

        wrapper = LLM_wrapper(
            model_name="test-model",
            system_prompt="Test",
            output_format_class=SimpleOutput,
        )

        output_dir = tmp_path / "output"
        wrapper.assess_dataframe(
            input_file_path=str(input_file),
            text_column="text",
            output_dir=str(output_dir),
        )

        checkpoint_dir = output_dir / "checkpoints"
        assert checkpoint_dir.exists()
        checkpoints = list(checkpoint_dir.glob("*.checkpoint.csv"))
        assert len(checkpoints) == 1

    def test_assess_dataframe_with_row_filter(self, mock_openai_client, tmp_path):
        """Test that row_filter correctly filters rows."""
        input_df = pd.DataFrame(
            {
                "text": ["Text 1", "Text 2", "Text 3"],
                "include": [True, False, True],
            }
        )
        input_file = tmp_path / "input.csv"
        input_df.to_csv(input_file, index=False)

        mock_response = MagicMock()
        mock_response.output_parsed = SimpleOutput(
            is_valid=True, reason="Valid", score=90
        )
        mock_openai_client.responses.parse.return_value = mock_response

        wrapper = LLM_wrapper(
            model_name="gpt-4o",
            system_prompt="Test",
            output_format_class=SimpleOutput,
        )

        output_dir = tmp_path / "output"
        wrapper.assess_dataframe(
            input_file_path=str(input_file),
            text_column="text",
            output_dir=str(output_dir),
            row_filter=lambda row: row["include"] == True,
        )

        # Only 2 rows should be processed (where include=True)
        assert mock_openai_client.responses.parse.call_count == 2

    def test_assess_dataframe_resumes_from_checkpoint(
        self, mock_openai_client, tmp_path
    ):
        """Test that processing resumes from checkpoint."""
        # Create checkpoint with some processed rows
        checkpoint_df = pd.DataFrame(
            {
                "text": ["Text 1", "Text 2", "Text 3"],
                "generation": [True, True, pd.NA],
                "is_valid": [True, True, pd.NA],
                "reason": ["Valid", "Valid", pd.NA],
                "score": [90, 85, pd.NA],
            }
        )

        output_dir = tmp_path / "output"
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_file = checkpoint_dir / "20240101_1200_test-model.checkpoint.csv"
        checkpoint_df.to_csv(checkpoint_file, index=False)

        # Also create input file
        input_file = tmp_path / "input.csv"
        pd.DataFrame({"text": ["Text 1", "Text 2", "Text 3"]}).to_csv(
            input_file, index=False
        )

        mock_response = MagicMock()
        mock_response.output_parsed = SimpleOutput(
            is_valid=True, reason="Valid", score=80
        )
        mock_openai_client.responses.parse.return_value = mock_response

        wrapper = LLM_wrapper(
            model_name="test-model",
            system_prompt="Test",
            output_format_class=SimpleOutput,
        )

        wrapper.assess_dataframe(
            input_file_path=str(input_file),
            text_column="text",
            output_dir=str(output_dir),
        )

        # Only 1 row should be processed (the one with generation=NA)
        assert mock_openai_client.responses.parse.call_count == 1


# ============================================================================
# Tests for prompts
# ============================================================================


class TestPrompts:
    def test_paper_assessment_prompt_contains_key_criteria(self):
        """Test that paper assessment prompt contains key criteria."""
        assert "multivariable prediction model" in PAPER_ASSESSMENT_PROMPT
        assert "is_match" in PAPER_ASSESSMENT_PROMPT
        assert "repo_url" in PAPER_ASSESSMENT_PROMPT
        assert "country" in PAPER_ASSESSMENT_PROMPT.lower()

    def test_repo_assessment_prompt_contains_key_criteria(self):
        """Test that repo assessment prompt contains key criteria."""
        assert "is_empty" in REPO_ASSESSMENT_PROMPT
        assert "contains_readme" in REPO_ASSESSMENT_PROMPT
        assert "contains_requirements" in REPO_ASSESSMENT_PROMPT
        assert "contains_license" in REPO_ASSESSMENT_PROMPT
        assert "reproducibility" in REPO_ASSESSMENT_PROMPT.lower()


# ============================================================================
# Integration-style tests with full mock chain
# ============================================================================


class TestIntegrationWithMocks:
    def test_full_paper_assessment_workflow(self, mock_openai_client, tmp_path):
        """Test complete paper assessment workflow with mocked API."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.output_parsed = PaperAssessment(
            is_match=True,
            reason="Contains logistic regression model",
            country_first_author_institution="United Kingdom",
            repo_url="https://github.com/example/repo",
            code_statement_locations=["methods"],
            code_statement_sentence="Code available at:",
        )
        mock_openai_client.responses.parse.return_value = mock_response

        # Create input file
        input_df = pd.DataFrame(
            {
                "full_text": ["Paper abstract and methods section..."],
            }
        )
        input_file = tmp_path / "papers.csv"
        input_df.to_csv(input_file, index=False)

        # Run assessment
        wrapper = LLM_wrapper(
            model_name="gpt-4o",
            system_prompt=PAPER_ASSESSMENT_PROMPT,
            output_format_class=PaperAssessment,
        )

        output_dir = tmp_path / "output"
        result = wrapper.assess_dataframe(
            input_file_path=str(input_file),
            text_column="full_text",
            output_dir=str(output_dir),
        )

        assert result["is_match"].iloc[0] == True
        assert result["country_first_author_institution"].iloc[0] == "United Kingdom"
        assert "github.com" in result["repo_url"].iloc[0]
