from pydantic import BaseModel
from typing import Optional


class PaperAssessment(BaseModel):
    is_match: bool
    reason: str
    url: str


class RepoAssessment(BaseModel):
    # Relevance
    is_empty: bool

    # README
    contains_readme: bool
    readme_purpose_and_outputs: Optional[bool]

    # Requirements
    contains_requirements: bool
    requirements_dependency_versions: Optional[bool]

    # License
    contains_license: bool

    # Documentation
    sufficient_code_documentation: bool

    # Modularity
    is_modular_and_structured: bool

    # Testing
    implements_tests: bool

    # Reproducibility
    fixes_seed_if_stochastic: Optional[bool]
    lists_hardware_requirements: bool

    # Citation and Linking
    contains_link_to_paper: bool
    contains_citation: bool

    # Data
    includes_data_or_sample: bool
