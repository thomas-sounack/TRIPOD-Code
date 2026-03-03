# TRIPOD-Code

This repository contains the code used in [Code Sharing In Prediction Model Research: A Scoping Review](#TODO).

## Table of Contents
1. [Introduction](#introduction)
2. [Setup](#setup)
3. [How to use this repository](#how-to-use-this-repository)
5. [Reference](#reference)

## Introduction

In this work, we conducted a large scale analysis of research articles and their associated code repositories to quantify current code-sharing practices. This repository contains the code used to obtain our results.

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
   Use the provided `environment.yml` file to create the environment. Run the following command:
   ```bash
   conda env create -f environment.yml
   conda activate TRIPOD-Code
   ```

4. **Add your OpenAI API key**  
    Our datasets at every step of the process are made available. However, if you would like to reproduce our results, or want to run new data through OpenAI's model, you will need to add an API key.
    For this, rename the file `.env.txt` to `.env`, and paste your API key after `OPENAI_API_KEY=`.

## How to use this repository

Order of notebooks, how to use them, inputs and outputs of each

## Reference

If you use this repository, please cite our [paper](#TODO):

#TODO: insert citation