# Contributing to TRIPOD-Code

Thank you for your interest in contributing to TRIPOD-Code! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Pull Requests](#pull-requests)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Commit Messages](#commit-messages)

## Code of Conduct

This project adheres to our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## How to Contribute

### Reporting Bugs

Before submitting a bug report:

1. Check the [existing issues](https://github.com/thomas-sounack/TRIPOD-Code/issues) to avoid duplicates
2. Ensure you're using the latest version of the code

When submitting a bug report, include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs. actual behavior
- Your environment (OS, Python version, conda environment)
- Any relevant error messages or logs

### Suggesting Enhancements

Enhancement suggestions are welcome! When submitting:

- Use a clear, descriptive title
- Provide a detailed description of the proposed enhancement
- Explain why this enhancement would be useful
- Include examples if applicable

### Pull Requests

1. Fork the repository
2. Create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes
4. Add or update tests as needed
5. Ensure all tests pass
6. Commit your changes (see [Commit Messages](#commit-messages))
7. Push to your fork and submit a pull request

## Development Setup

1. **Clone your fork:**
   ```bash
   git clone https://github.com/YOUR-USERNAME/TRIPOD-Code.git
   cd TRIPOD-Code
   ```

2. **Create and activate the conda environment:**
   ```bash
   conda env create -f environment.yaml
   conda activate TRIPOD-Code
   ```

3. **Set up your OpenAI API key (if needed):**
   ```bash
   cp .env.example .env
   # Edit .env and add your API key
   ```

4. **Run the tests to verify your setup:**
   ```bash
   pytest tests/ -v
   ```

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code
- Use descriptive variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Comment complex logic

### Documentation

- Update docstrings when modifying functions
- Update the README if adding new features
- Add inline comments for complex logic

## Testing

All contributions should include appropriate tests.

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_llm_utils.py -v

# Run with coverage
pytest tests/ --cov=src
```

### Writing Tests

- Place tests in the `tests/` directory
- Follow the existing test structure
- Mock external API calls (see `test_llm_utils.py` for examples)
- Test both success and failure cases

## Commit Messages

Write clear, concise commit messages:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters
- Reference issues when relevant (e.g., "Fix #123")

### Examples

```
Add support for GitLab repository cloning

Implement DefaultGitCloner for GitLab URLs to extend
repository support beyond GitHub.

Fixes #42
```

```
Fix token counting in large files

Handle edge case where files exceed context window
by implementing proper truncation.
```

---

Thank you for contributing to TRIPOD-Code!
