# Multilingual Semantic Relations Generator

A comprehensive system for generating multilingual semantic relation questions using BabelNet, designed to create diverse question-answering datasets across multiple languages and resource levels.

## Features

- **Multilingual Support**: Generate questions across high, medium, and low-resource languages
- **Multiple Semantic Relations**: Support for hypernyms, hyponyms, meronyms, cohyponyms, and holonyms
- **Adaptive Difficulty**: 5 difficulty levels with intelligent distractor generation
- **Semantic Analogies**: Generate complex analogy questions across language pairs
- **BabelNet Integration**: Leverage BabelNet's multilingual knowledge graph

## Getting Started

### Prerequisites

- Python 3.8
- BabelNet API access and credentials or Offline RPC mode (preferred)

### Quick Start

```bash
# 1. Generate seed words and assemble semantic network
python 1_word_assembler.py

# 2. Filter words with required semantic relations
python 2_fetch_words_with_hyper_mero.py

# 3. Create multilingual semantic dataset
python 3_multilingual_babelnet_relations.py

# 4. Generate hypernymy and meronymy questions
python generate_hypernym_meronym_qa.py

# 5. Generate semantic analogy questions
python generate_semantic_analogies_qa.py
```

## Core Components

### Word Assembler
Recursively traverses BabelNet's semantic network to discover related concepts with depth-first traversal and configurable limits.

### Relation Filter
Filters synsets that have all required semantic relations, ensuring quality control for downstream question generation.

### Multilingual Dataset Creator
Builds comprehensive multilingual semantic datasets with 497 synsets, multi-language translations, and rich metadata.

### Question Generators
- **Hypernymy/Meronymy Questions**: 5,000 questions per relation type with cross-lingual generation
- **Semantic Analogies**: 5000 complex analogy patterns (A:B :: C:D) with cross-lingual generation

## Language Support

### High-Resource Languages
English, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean, Arabic, Turkish, Dutch, Polish, Swedish, Norwegian, Danish, Finnish, Czech, Romanian, Hungarian, Ukrainian, Hebrew, Bulgarian, Greek 

### Medium-Resource Languages
Croatian, Serbian, Slovak, Slovenian, Lithuanian, Latvian, Estonian, Thai, Vietnamese, Malay, Persian, Indonesian, Tamil, Hindi, Bengali

### Low-Resource Languages
Swahili, Icelandic, Maltese, Irish, Welsh, Bosnian, Georgian, Amharic, Uzbek, Tagalog

## Difficulty Levels

1. **Random Unrelated**: Basic random distractors
2. **Mixed Random + Semantic**: Combination approach
3. **Semantically Related**: Cohyponyms and related concepts
4. **Close Semantic Matches**: Hyponyms and direct relations
5. **Very Close Matches**: Meronyms and highly related concepts

## Configuration

Modify `language_config.py` to adjust language resource levels and target language sets. Key parameters in generation scripts include `NUM_QUESTIONS_PER_TYPE`, `NUM_SYNSETS`, `max_depth`, and `max_items`.

## License

This project is **not open source** and is licensed under an **All Rights Reserved** model. No part of this repository may be copied, modified, or distributed without explicit written permission.


## Acknowledgments

- [BabelNet](https://babelnet.org/) for multilingual semantic knowledge
- The multilingual NLP research community

**Note**: This project requires BabelNet API access. Please ensure you have proper credentials and respect BabelNet's usage policies.
