[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18897514.svg)](https://doi.org/10.5281/zenodo.18897514)

# TRIPOD-Code

This repository contains the code used in [Code Sharing In Prediction Model Research: A Scoping Review](#TODO).

## Table of Contents
1. [Introduction](#introduction)
2. [Setup](#setup)
3. [How to use this repository](#how-to-use-this-repository)
5. [Reference](#reference)

## Introduction

In this work, we conducted a large scale analysis of research articles and their associated code repositories to quantify current code-sharing practices. This repository contains the code used to obtain our results.

To characterize articles and repositories, we used OpenAI's structured output generation tool. If you are not familiar with it, you can find an example of how it works in this notebook: **TODO**, where we apply it on this repository.

## Setup

To use our repository, you first need to follow these steps:

1. **Clone Repository and access it**
   ```bash
   git clone https://github.com/thomas-sounack/TRIPOD-Code.git
   cd TRIPOD-Code
   ```

2. **Install Conda (if not already installed)**  
   If you don't have Conda installed, you can download it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).

3. **Create the Conda Environment and activate it**  
   Use the provided `environment.yaml` file to create the environment. Run the following command:
   ```bash
   conda env create -f environment.yaml
   conda activate TRIPOD-Code
   ```

4. **Add your OpenAI API key**  
    Our datasets at every step of the process are made available. However, if you would like to reproduce our results, or want to run new data through OpenAI's model, you will need to add an API key.
    For this, rename the file `.env.example` to `.env`, and paste your API key after `OPENAI_API_KEY=`.

    Note: you can visit [OpenAI's website](https://developers.openai.com/api/docs/quickstart/) for more information on how to obtain an API key, and pricing. It cost about $150 to run this pipeline end-to-end when this study was realized.

## How to use this repository

To reproduce our results, you can run the following notebooks:

0. [Merge TRIPOD references](https://github.com/thomas-sounack/TRIPOD-Code/blob/21791f4dfa95f18cf1ea3e7fcb6c196cc15b4912/notebooks/00_merge_TRIPOD_references.ipynb): 
    - Inputs: This notebook uses the data present in `data/00_references`. The csv files present in the `TRIPOD/` and `TRIPOD_AI` folders were downloaded from the Pubmed website, each corresponding to one entry of the TRIPOD statements.
    - Purpose: The notebook aggregates all of the records citing one of the TRIPOD statements into a single file, and removes duplicate entries
    - Outputs: The notebook outputs a single file: `data/00_references/merged_references.parquet.br`.

1. [Gather full text](https://github.com/thomas-sounack/TRIPOD-Code/blob/21791f4dfa95f18cf1ea3e7fcb6c196cc15b4912/notebooks/01_gather_full_text.ipynb):
    - Inputs: The file generated in the last step: `data/00_references/merged_references.parquet.br`.
    - Purpose: The notebook gathers the full text of the records using the PMC Open Access API. It removes records that are not indexed in the PMC Open Access API.
    - Outputs: The notebook outputs a single file: `data/01_full_text/references_with_full_text.parquet.br`.

2. Article Assessment:
    - **2a)** [Create paper assesstment annotations](https://github.com/thomas-sounack/TRIPOD-Code/blob/21791f4dfa95f18cf1ea3e7fcb6c196cc15b4912/notebooks/02a_create_paper_assessment_annotations.ipynb):
        - Inputs: The file generated in the last step `data/01_full_text/references_with_full_text.parquet.br` and the file containing the annotations of 500 articles: `data/02_paper_assessment/raw_annotations_paper_assessment.csv`.
        - Purpose: This notebook adds the full text information to the annotated records, and renames columns with a `gt_` suffix to facilitate model evaluation.
        - Outputs: The notebook outputs a single file: `data/02_paper_assessment/gt_paper_assessment.csv`.

    - **2b)** [Evaluate paper assessment prompt](https://github.com/thomas-sounack/TRIPOD-Code/blob/21791f4dfa95f18cf1ea3e7fcb6c196cc15b4912/notebooks/02b_evaluate_paper_assessment_prompt.ipynb):
        - Inputs: The file generated in the last step: `data/02_paper_assessment/gt_paper_assessment.csv`, as well as the model prompt and structured output schema defined in `src/llm_utils/paper_assessment_prompt.py` and `src/llm_utils/structs.py`.
        - Purpose: This notebook evaluates the LLM's article assessment using the 500 annotated articles.
        - Outputs: The model's outputs (saved in `data/02_paper_assessment/evaluation_paper_assessment.csv`) and performance metrics.

    - **2c)** [Generate paper assessment predictions](https://github.com/thomas-sounack/TRIPOD-Code/blob/21791f4dfa95f18cf1ea3e7fcb6c196cc15b4912/notebooks/02c_generate_paper_assessment_predictions.ipynb): 
        - Inputs: The file containing the full text articles generated in step 1: `data/01_full_text/references_with_full_text.parquet.br`, as well as the model prompt and structured output schema defined in `src/llm_utils/paper_assessment_prompt.py` and `src/llm_utils/structs.py`.
        - Purpose: This notebook runs the LLM's article assessment on the entire dataset.
        - Outputs: The model's outputs (saved in `data/02_paper_assessment/paper_assessment_pred.parquet.br`).

3. Repository Assessment:

    - **3a)** [Create repository assessment annotations](https://github.com/thomas-sounack/TRIPOD-Code/blob/21791f4dfa95f18cf1ea3e7fcb6c196cc15b4912/notebooks/03a_create_repo_assessment_annotations.ipynb):
        - Inputs: The file containing the annotated repositories (`data/03_repo_assessment/raw_annotations_repo_assessment.csv`).
        - Purpose: This notebook downloads the repositories using the repository utility (`src/repo_utils/`), and adds the repository contents to the annotated data, renaming columns with a `gt_` suffix to facilitate model evaluation.
        - Outputs: The notebook outputs a single file `data/03_repo_assessment/gt_repo_assessment.csv`.

    - **3b)** [Evaluate repository assessment prompt](https://github.com/thomas-sounack/TRIPOD-Code/blob/21791f4dfa95f18cf1ea3e7fcb6c196cc15b4912/notebooks/03b_evaluate_repo_assessment_prompt.ipynb):
        - Inputs: The file generated in the last step: `data/03_repo_assessment/gt_repo_assessment.csv`, as well as the model prompt and structured output schema defined in `src/llm_utils/repo_assessment_prompt.py` and `src/llm_utils/structs.py`.
        - Purpose: This notebook evaluates the LLM's repository assessment using the annotated repositories.
        - Outputs: The model's outputs (saved in `data/03_repo_assessment/evaluation_repo_assessment.csv`) and performance metrics.

    - **3c)** [Download repositories](https://github.com/thomas-sounack/TRIPOD-Code/blob/21791f4dfa95f18cf1ea3e7fcb6c196cc15b4912/notebooks/03c_download_repos.ipynb): 
        - Inputs: The file containing the extracted repository links generated in step 2c (`data/02_paper_assessment/paper_assessment_pred.parquet.br`).
        - Purpose: This notebook downloads the repositories for the entire dataset using the repository utility (`src/repo_utils/`).
        - Outputs: A file containing all the rows of the dataset for which a repository link was found in step 2c, along with the repository contents: `data/03_repo_assessment/references_with_repo.parquet.br`.

    - **3d)** [Generate repository assessment predictions](https://github.com/thomas-sounack/TRIPOD-Code/blob/21791f4dfa95f18cf1ea3e7fcb6c196cc15b4912/notebooks/03d_generate_repo_assessment_predictions.ipynb):
        - Inputs: The file generated in the last step: `data/03_repo_assessment/references_with_repo.parquet.br`.
        - Purpose: This notebook runs the LLM's repository assessment on all repositories that were downloaded.
        - Outputs: The model's outputs (saved in `data/03_repo_assessment/references_repo_assessment_pred.parquet.br`).

4. [Final analysis](https://github.com/thomas-sounack/TRIPOD-Code/blob/21791f4dfa95f18cf1ea3e7fcb6c196cc15b4912/notebooks/04_final_analysis.ipynb):
    - Inputs: The LLM article assessment (`data/02_paper_assessment/paper_assessment_pred.parquet.br`) and repository assessment (`data/03_repo_assessment/references_repo_assessment_pred.parquet.br`).
    - Purpose: Generates figures and analyses using the model outputs.
    - Outputs: Article and repository statistics, and figures saved in `data/figures`.
    

## Reference

If you use this repository, please cite our [paper](#TODO):

#TODO: insert citation
