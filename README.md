# LLM Grooming Detection System

## Overview

This repository provides tools and models for detecting grooming behavior in chat conversations using large language models (LLMs) and traditional ML approaches. It includes data preprocessing, model training, evaluation, and inference utilities.

---

## Features

- **Data Filtering & Preprocessing**
  - `filters/filter_pan12.py`: Converts PAN12 XML to filtered JSON, labels predatory conversations, and enforces minimum message/participant criteria.
  - `filters/filter_pj.py`: Parses and filters Perverted Justice (PJ) chat logs, extracting conversations and contextual metadata.

- **Dataset Statistics**
  - `count_filtered_pan12_conversations.py`: Counts conversations in filtered datasets for both training and test splits.

- **Model Training**
  - `LLM.py` and `training.py`: Interactive CLI for training LLMs on chat data. Supports:
    - Model selection (from HuggingFace or local)
    - Custom epochs, learning rates, and output naming
    - Training from state files for reproducibility

- **Model Evaluation**
  - `LLM.py` and `evaluation.py`: CLI for evaluating models on test datasets. Features:
    - Batch evaluation of multiple models
    - Parallelized evaluation for speed
    - Summarization of results and filtering by model/settings
    - Custom evaluation via config files

- **Results Analysis**
  - `print_results_csv.py`: Pretty-prints evaluation results CSVs, sorted by F1 score.

---

## Main Interactions

- **Train a Model**
  - Run `LLM.py`, select "Train models", choose dataset size, model, epochs, and learning rate.
  - Optionally, train from a state file for batch jobs.

- **Evaluate Models**
  - Run `LLM.py`, select "Evaluate models", choose evaluation type (new, summarizer, continue, or custom config).
  - Supports parallel evaluation and result filtering.

- **Add Untrained Models**
  - Download and prepare base models for fine-tuning via the CLI.

- **Delete Models**
  - Remove unwanted fine-tuned models interactively.

- **Filter and Prepare Data**
  - Use scripts in `filters/` to process raw datasets into filtered JSON for training/evaluation.

- **Count Conversations**
  - Run `count_filtered_pan12_conversations.py` to get stats on filtered datasets.

---

## Getting Started

1. **Install dependencies** (see requirements in each script or use pip for `transformers`, `tensorflow`, etc.).
2. **Prepare datasets** using the scripts in `filters/`.
3. **Train or evaluate models** using `LLM.py`.

---

## Notes

- The system is modular: you can use only the data, only the models, or the full pipeline.

---
