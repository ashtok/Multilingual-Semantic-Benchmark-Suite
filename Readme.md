# Multilingual Lexical-Semantic Benchmark

This repository contains the tools, data generation pipeline, and evaluation resources for building and analyzing a large-scale multilingual benchmark designed to evaluate fine-grained lexical semantic reasoning in Large Language Models (LLMs).

The benchmark is grounded in BabelNet, a multilingual semantic network that integrates lexicographic and encyclopedic knowledge. It is designed to be compatible with the lm-eval-harness framework, allowing systematic evaluation across multiple LLMs.

## 🎯 Motivation

Large Language Models demonstrate impressive capabilities in reasoning, translation, and knowledge recall, but their ability to capture lexical semantic relationships across languages remains underexplored. Traditional benchmarks often focus on high-resource languages (primarily English) and lack fine-grained controls over semantic relations and task difficulty.

This project introduces a comprehensive multilingual benchmark that fills this gap by systematically probing models' understanding of:

- **Lexical semantic relations** (Hypernymy, Meronymy, Analogies, Gloss comprehension)
- **Cross-lingual generalization** (high, medium, and low-resource languages)
- **Difficulty scaling** (easy, medium, hard distractors for controlled challenge)
- **Prompting effects** (zero-shot vs. few-shot performance)

## 🧩 Benchmark Design

The benchmark evaluates four semantic reasoning tasks:

### Hypernymy ("is-a")
- **Example:** dog → mammal
- Probes hierarchical categorization

### Meronymy ("part-of")
- **Example:** wheel → car
- Assesses part-whole understanding

### Semantic Analogies
- **Example:** king : queen :: man : woman
- Measures relational proportional reasoning

### Gloss-based Inference
- Questions are derived from BabelNet glosses (definitions)
- Evaluates how well models interpret definitions in semantic reasoning

### Difficulty Scaling

Each task includes multiple difficulty levels, where distractors (incorrect options) are chosen based on semantic similarity:

- **Easy:** Distant distractors (unrelated words)
- **Medium:** Semantically closer distractors
- **Hard:** Fine-grained semantic distinctions

This design ensures evaluation captures not only general lexical knowledge but also sensitivity to nuanced distinctions.

## 🌍 Language Coverage

The benchmark spans 50+ languages, categorized by resource availability:

**High-resource:** English, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean, Arabic, Turkish, Dutch, Polish, Swedish, Norwegian, Danish, Finnish, Czech, Romanian, Hungarian, Ukrainian, Hebrew, Bulgarian, Greek.

**Medium-resource:** Hindi, Bengali, Persian, Indonesian, Thai, Vietnamese, Malay, Tamil, Croatian, Serbian, Slovak, Slovenian, Lithuanian, Latvian, Estonian.

**Low-resource:** Swahili, Maltese, Georgian, Amharic, Uzbek, Tagalog, Bosnian, Irish, Welsh, Icelandic.

Configuration files are maintained in `DataGeneration/language_config.py`.

## ✨ Key Features

- **Multilingual & Cross-lingual:** Evaluation across high-, medium-, and low-resource languages
- **Multiple Task Types:** Hypernymy, Meronymy, Analogies, Gloss-based reasoning
- **Difficulty Levels:** Fine-grained control of distractor difficulty for nuanced analysis
- **Zero-Shot vs. Few-Shot:** Direct comparison of prompting strategies
- **Integration with lm-eval-harness:** Standardized evaluation pipeline
- **Extensible:** Modular design to support new languages or semantic relations

## ⚡ Quick Start

### Data Generation Pipeline

Run the following scripts in sequence to generate benchmark data:

```bash
python DataGeneration/1_word_assembler.py
python DataGeneration/2_fetch_words_with_hyper_mero.py
python DataGeneration/3_multilingual_babelnet_relations.py
python DataGeneration/4_generate_questions.py
python DataGeneration/5_generate_analogies.py
python DataGeneration/6_generate_gloss_questions.py
```

### Evaluation

Generated data is stored in `EvaluationFiles/QA_Json/`.
Evaluation configurations (YAML) are located in `EvaluationFiles/Tasks/`.

Run with lm-eval-harness:

```bash
lm_eval --model <model_name> --tasks <task_yaml> --output_path results/
```

## 📂 Repository Structure

```
Babelnet_Client/
├── DataGeneration/           # Scripts for data generation
├── EvaluationFiles/          # Task configs + QA JSONs for lm-eval-harness
│   ├── QA_Json/
│   └── Tasks/
├── GeneratedFiles/           # Intermediate + final benchmark data
├── results/                  # Evaluation results & analysis
│   ├── CompiledResults/      # Aggregated outputs
│   └── DeepAnalysis/         # Fine-grained model comparisons
├── language_categorization.py
├── report_content.txt
├── README.md
└── LICENSE
```

## 🧪 Evaluation Insights

The benchmark was used to evaluate five state-of-the-art LLMs:

- LLaMA-3.1-8B-Instruct
- Qwen3-8B
- Mistral-7B-Instruct-v0.3
- Gemma-7B-IT
- Gemma-3-1B-IT

### Key Findings

#### Model Scaling Matters
- Larger models consistently outperformed smaller models
- The smallest model (Gemma-3-1B-IT) underperformed across all tasks, confirming the strong impact of scale on semantic reasoning

#### Few-Shot > Zero-Shot
- Few-shot learning provided substantial gains across models
- Larger models achieved the best absolute performance in few-shot mode
- Smaller models showed greater relative improvement, suggesting they benefit more from contextual guidance

#### Task-Specific Performance
- **Hypernymy & Meronymy:** Showed the largest gains in few-shot settings, highlighting their reliance on contextual disambiguation
- **Analogies:** Particularly challenging, with performance gaps widening at higher difficulty levels
- **Gloss Tasks:** Displayed moderate but stable improvements, suggesting definitional reasoning is a distinct capability

#### Difficulty Sensitivity
- Accuracy decreased systematically as distractor difficulty increased
- Larger models were better at handling fine-grained semantic distinctions, while smaller models struggled disproportionately

## ⚙️ Prerequisites

- Python 3.8
- BabelNet account & API key

## 📜 License

This project is licensed for academic and research purposes only.
Commercial use is strictly prohibited.
See the LICENSE file for details.

## 🙏 Acknowledgments

Developed by:
- **Ashutosh Mahajan**
- **Pallabi Pathak**

at the University of Würzburg.

## 📖 Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{mahajan2025multilingual,
  author = {Mahajan, Ashutosh and Pathak, Pallabi},
  title = {A Multilingual Lexical-Semantic Benchmark for Large Language Models},
  year = {2025},
  publisher = {University of Würzburg}
}
```
