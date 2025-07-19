# 🧠 Lexical Meaning Benchmark

**Multilingual Semantic Benchmarks for Hypernymy, Meronymy, and Analogies**

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red)
![LM Eval Compatible](https://img.shields.io/badge/lm--eval-compatible-green)

---

## 📖 Overview

This repository provides a comprehensive pipeline for generating and evaluating **multilingual lexical semantic question-answering datasets** using [BabelNet](https://babelnet.org/). The benchmark covers semantic relations across **50 languages** with **5 difficulty levels**, designed for systematic evaluation of language models' semantic understanding.

### Key Features

- **🌍 Multilingual Coverage**: 50 languages across high-, medium-, and low-resource tiers
- **🔗 Semantic Relations**: Hypernymy, meronymy, and semantic analogies
- **📊 Difficulty Scaling**: Five levels from random to very close semantic matches
- **🔄 Cross-lingual Support**: Monolingual and cross-lingual question generation
- **⚡ LM Eval Ready**: Compatible with [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **BabelNet RPC Server** (Docker setup required)
  - Follow the [BabelNet Docker Setup Guide](https://babelnet.org/guide)
- **Dependencies**: Install required packages (see `requirements.txt`)

### Installation

```bash
git clone https://github.com/ashtok/Lexical_Meaning_Benchmark.git
cd <directory>
pip install -r requirements.txt
```

### Running the Pipeline

Execute the following scripts in order:

```bash
# 1. Assemble seed synsets
python DataGeneration/1_word_assembler.py

# 2. Filter synsets with semantic relations
python DataGeneration/2_fetch_words_with_hyper_mero.py

# 3. Build multilingual semantic network
python DataGeneration/3_multilingual_babelnet_relations.py

# 4. Generate hypernymy & meronymy datasets
python DataGeneration/generate_hypernym_meronym_qa.py

# 5. Generate semantic analogy datasets
python DataGeneration/generate_semantic_analogies_qa.py
```

**Configuration**: Modify `language_config.py` and script constants (e.g., `NUM_SYNSETS`, `NUM_QUESTIONS_PER_TYPE`) to adjust generation parameters.

---

## 📁 Repository Structure

```
├── DataGeneration/              # Core pipeline scripts
│   ├── 1_word_assembler.py     # Seed synset assembly
│   ├── 2_fetch_words_with_hyper_mero.py  # Relation filtering
│   ├── 3_multilingual_babelnet_relations.py  # Multilingual network
│   ├── generate_hypernym_meronym_qa.py  # Hypernymy/meronymy QA
│   ├── generate_semantic_analogies_qa.py  # Analogy generation
│   ├── generate_questions.py   # Question generation utilities
│   ├── generate_analogies.py   # Analogy utilities
│   ├── fetch_relatives_helper.py  # BabelNet relation helpers
│   ├── babelnet_conf.yml       # BabelNet configuration
│   └── language_config.py      # Language tier definitions
│
├── EvaluationFiles/            # LM-eval-harness task files
│   ├── analogies_*.yaml        # Analogy evaluation tasks
│   ├── hypernymy_*.yaml        # Hypernymy evaluation tasks
│   ├── meronymy_*.yaml         # Meronymy evaluation tasks
│   ├── *_questions_*.json      # Question datasets
│   └── msi_*_custom_task.yaml  # Custom task definitions
│
├── GeneratedFiles/             # Generated datasets and intermediates
│   ├── seed_words_10.txt       # Seed word lists
│   ├── assembled_words.txt     # Assembled synsets
│   ├── babelnet_with_relations.txt  # Semantic relations
│   └── JsonFiles/              # Structured datasets
│       ├── Hypernymy/          # Hypernymy datasets
│       ├── Meronymy/           # Meronymy datasets
│       └── Analogies/          # Analogy datasets
│
└── results/                    # Evaluation results
    ├── Hypernymy/              # Hypernymy results by tier
    ├── Meronymy/               # Meronymy results by tier
    └── Analogies/              # Analogy results by tier
        ├── High/               # High-resource languages
        ├── Medium/             # Medium-resource languages
        ├── Low/                # Low-resource languages
        ├── Mixed/              # Mixed difficulty
        └── Monolingual_EN/     # English monolingual
```

---

## 🌐 Language Coverage

| **Tier** | **Count** | **Languages** |
|-----------|-----------|---------------|
| **High** | 25 | English, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean, Arabic, Turkish, Dutch, Polish, Swedish, Norwegian, Danish, Finnish, Czech, Romanian, Hungarian, Ukrainian, Hebrew, Bulgarian, Greek |
| **Medium** | 15 | Croatian, Serbian, Slovak, Slovenian, Lithuanian, Latvian, Estonian, Thai, Vietnamese, Malay, Persian, Indonesian, Tamil, Hindi, Bengali |
| **Low** | 10 | Swahili, Icelandic, Maltese, Irish, Welsh, Bosnian, Georgian, Amharic, Uzbek, Tagalog |

---

## 📊 Semantic Relations & Difficulty Levels

### Relation Types

- **Hypernymy**: Cat → Animal (is-a relationship)
- **Meronymy**: Wheel → Car (part-of relationship)  
- **Semantic Analogies**: Cat:Kitten :: Dog:Puppy (A:B :: C:D patterns)

### Difficulty Levels

1. **Random Unrelated** - Basic random distractors
2. **Mixed (Random + Semantic)** - Combination of random and semantic distractors
3. **Semantically Related** - Cohyponyms and related concepts
4. **Close Matches** - Hyponyms and close semantic relations
5. **Very Close Matches** - Meronyms and highly related terms

---

## 🧪 Evaluation with LM-Eval-Harness

This benchmark is fully compatible with the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness). Pre-configured task files are provided in `EvaluationFiles/`.

### Running Evaluations (after setting up LM Evaluation Harness)

```bash
# Evaluate hypernymy (all languages)
lm_eval --model <model_name> --tasks hypernymy_all --output_path results/hypernymy/

# Evaluate analogies (high-resource only)
lm_eval --model <model_name> --tasks analogies_high --output_path results/analogies/

# Evaluate meronymy (monolingual English)
lm_eval --model <model_name> --tasks meronymy_mono --output_path results/meronymy/
```

### Available Task Categories

- **All Languages**: `hypernymy_all`, `meronymy_all`, `analogies_all`
- **Resource Tiers**: `*_high`, `*_medium`, `*_low`, `*_mixed`
- **Monolingual**: `*_mono` (English only)

---

## 🔧 Configuration

### BabelNet Setup

1. **Install Docker** and pull the BabelNet RPC image
2. **Configure** `babelnet_conf.yml` with your API credentials
3. **Start** the BabelNet RPC server locally

### Pipeline Parameters

Key configuration options in scripts:

- `NUM_SYNSETS`: Number of synsets to process
- `NUM_QUESTIONS_PER_TYPE`: Questions generated per relation type
- `DIFFICULTY_LEVELS`: Enabled difficulty levels
- `TARGET_LANGUAGES`: Languages to include in generation

Modify `language_config.py` for language tier customization.

---

## 📈 Results Structure

Evaluation results are organized by:

- **Task Type**: Hypernymy, Meronymy, Analogies
- **Resource Tier**: High, Medium, Low, Mixed
- **Language Scope**: Multilingual vs. Monolingual
- **Model Performance**: Accuracy scores across difficulty levels

Example result path: `results/Analogies/High/model_performance.json`

---

## 📜 License

**All Rights Reserved**

This repository is proprietary and not open source. No part of this project may be copied, modified, distributed, or used without explicit written permission from the authors.

---

## 🙏 Acknowledgments

- **[BabelNet](https://babelnet.org/)** for providing the multilingual semantic knowledge base
- **[LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)** by EleutherAI for the evaluation framework
- The **multilingual NLP research community** for advancing semantic understanding

---

## 📚 Citation

If you use this benchmark in academic research, please cite:

```bibtex
@misc{lexical-meaning-benchmark,
  title={Lexical Meaning Benchmark: Multilingual Semantic Benchmarks for Hypernymy, Meronymy, and Analogies},
  author={Pallabi Pathak and Ashutosh Mahajan},
  institution={University of Würzburg},
  year={2025},
  note={Project in Computer Science - All rights reserved}
}
```

---

*Last updated: [19/07/2025]*
