# Multilingual Semantic-Relational Benchmark

This repository contains the tools and resources for generating a multilingual, multi-level benchmark for evaluating Large Language Models (LLMs) on their understanding of semantic relations. The benchmark is generated using BabelNet and is designed to be used with the `lm-eval-harness` library.

## Overview

The primary goal of this project is to create a comprehensive benchmark for evaluating the semantic-relational reasoning capabilities of LLMs across a wide range of languages. The benchmark focuses on three key semantic relations:

*   **Hypernymy:** The "is-a" relationship (e.g., "dog" is a "mammal").
*   **Meronymy:** The "part-of" relationship (e.g., "wheel" is a part of a "car").
*   **Semantic Analogies:** The relationship between two pairs of words (e.g., "king" is to "queen" as "man" is to "woman").

The benchmark is designed to be multilingual, with support for high, medium, and low-resource languages. It also includes different difficulty levels based on the language resource availability.

## Key Features

*   **Multilingual:** The benchmark covers a wide range of languages, categorized into high, medium, and low-resource tiers.
*   **Multiple Semantic Relations:** The benchmark evaluates LLMs on their understanding of hypernymy, meronymy, and semantic analogies.
*   **Difficulty Levels:** The benchmark provides different difficulty levels based on the language resource availability, allowing for a more fine-grained analysis of LLM performance.
*   **Integration with `lm-eval-harness`:** The benchmark is designed to be used with the popular `lm-eval-harness` library, making it easy to evaluate a wide range of LLMs.
*   **Extensible:** The data generation pipeline is modular and can be extended to include other semantic relations or languages.

## Quick Start

To get started with the project, you can run the following command to generate the data and prepare it for evaluation:

```bash
python DataGeneration/1_word_assembler.py
python DataGeneration/2_fetch_words_with_hyper_mero.py
python DataGeneration/3_multilingual_babelnet_relations.py
python DataGeneration/4_generate_questions.py
python DataGeneration/5_generate_analogies.py
python DataGeneration/6_generate_gloss_questions.py
```

## Prerequisites

*   Python 3.8+
*   BabelNet account and API key
*   Java 8 or higher

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/your-username/Babelnet_Client.git
    cd Babelnet_Client
    ```

2.  Install the required Python libraries. It is recommended to use a virtual environment:

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file is not provided. You will need to install the necessary libraries manually. These include `babelnet`, `pyyaml`, etc.)*

3.  Set up the BabelNet configuration. You will need to have a running BabelNet instance. Update the `DataGeneration/babelnet_conf.yml` file with your BabelNet RPC URL:

    ```yaml
    RPC_URL: "tcp://127.0.0.1:7790"
    ```

## Running the Pipeline

The data generation pipeline consists of several scripts that need to be run in order.

1.  **`1_word_assembler.py`**: This script assembles a list of seed words.
2.  **`2_fetch_words_with_hyper_mero.py`**: This script fetches words with hypernym and meronym relations from BabelNet.
3.  **`3_multilingual_babelnet_relations.py`**: This script generates multilingual relations from BabelNet.
4.  **`4_generate_questions.py`**: This script generates hypernymy and meronymy questions from the collected data.
5.  **`5_generate_analogies.py`**: This script generates semantic analogy questions.
6.  **`6_generate_gloss_questions.py`**: This script generates gloss questions.

## Repository Structure

```
Babelnet_Client/
├── DataGeneration/           # Scripts for generating the benchmark data
├── EvaluationFiles/          # Files for evaluating LLMs with lm-eval-harness
│   ├── QA_Json/              # Generated question-answer JSON files
│   └── Tasks/                # lm-eval-harness task configuration files
├── GeneratedFiles/           # Intermediate and final generated files
├── results/                  # Evaluation results and analysis scripts
├── .gitignore
├── All_API_Test.py
├── language_categorization.py
├── LICENSE
├── README.md
└── report_content.txt
```

## Language Coverage

The benchmark supports a wide range of languages, which are categorized into three tiers based on their resource availability:

*   **High-Resource Languages:** English, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean, Arabic, Turkish, Dutch, Polish, Swedish, Norwegian, Danish, Finnish, Czech, Romanian, Hungarian, Ukrainian, Hebrew, Bulgarian, Greek.
*   **Medium-Resource Languages:** Croatian, Serbian, Slovak, Slovenian, Lithuanian, Latvian, Estonian, Thai, Vietnamese, Malay, Persian, Indonesian, Tamil, Hindi, Bengali.
*   **Low-Resource Languages:** Swahili, Icelandic, Maltese, Irish, Welsh, Bosnian, Georgian, Amharic, Uzbek, Tagalog.

The language configuration can be found in `DataGeneration/language_config.py`.

## Semantic Relations & Difficulty Levels

The benchmark evaluates LLMs on three semantic relations:

*   **Hypernymy:** The "is-a" relationship.
*   **Meronymy:** The "part-of" relationship.
*   **Semantic Analogies:** The relationship between two pairs of words.

The difficulty levels are defined based on the language resource availability:

*   **Monolingual (English):** All questions and answers are in English.
*   **High-Resource:** Questions and answers are in high-resource languages.
*   **Medium-Resource:** Questions and answers are in medium-resource languages.
*   **Low-Resource:** Questions and answers are in low-resource languages.
*   **All:** A mix of all languages.

## Evaluation with LM-Eval-Harness

The benchmark is designed to be used with `lm-eval-harness`. The task configuration files are located in the `EvaluationFiles/Tasks` directory. Each YAML file defines a specific evaluation task. For example, `analogies_all.yaml` defines the task for evaluating semantic analogies across all languages.

To run the evaluation, you will need to place the generated JSON files in the appropriate directory and then run `lm-eval-harness` with the desired task.

## Configuration

The main configuration file for the data generation pipeline is `DataGeneration/babelnet_conf.yml`. This file contains the RPC URL for your BabelNet instance.

The language configuration is defined in `DataGeneration/language_config.py`. You can modify this file to add or remove languages.

## Results Structure

The `results` directory contains the evaluation results from `lm-eval-harness`. The results are organized by model and task. The `CompiledResults` subdirectory contains scripts for analyzing and compiling the results. The `DeepAnalysis` subdirectory contains scripts for more in-depth analysis of the results.

## License

This project is licensed for academic and research purposes only. Any commercial use of this project, its code, or the data generated by it is strictly forbidden. See the `LICENSE` file for more details.

## Acknowledgments

This project was developed by Ashutosh Mahajan and Pallabi Pathak at the University of Würzburg.

## Citation

If you use this benchmark in your research, please cite the following:

```
@misc{mahajan2025multilingual,
  author = {Mahajan, Ashutosh and Pathak, Pallabi},
  title = {A Multilingual Semantic-Relational Benchmark for Large Language Models},
  year = {2025},
  publisher = {University of Würzburg}
}
```
