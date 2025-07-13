import json
import random
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
from enum import Enum
from language_config import LANGUAGE_CONFIG
from tqdm import tqdm
import logging

# Configure logging - reduced to WARNING level
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class DifficultyLevel(Enum):
    RANDOM = 1
    MIXED = 2
    SEMANTIC = 3
    CLOSE_SEMANTIC = 4
    VERY_CLOSE = 5


class MultilingualMode(Enum):
    EN_TO_HIGH = "en_to_high"
    EN_TO_MEDIUM = "en_to_medium"
    EN_TO_LOW = "en_to_low"
    EN_TO_ALL = "en_to_all"
    MONOLINGUAL_EN = "monolingual_en"
    ALL = "all"


@dataclass
class QuestionMetadata:
    resource_pair: str
    prompt_lang: str
    from_lang: str
    to_lang: str
    difficulty: int
    distractor_type: str
    generation_time: str
    synset_id: str
    multilingual_mode: str
    relation_type: str


@dataclass
class Question:
    id: str
    prompt: str
    options: List[str]
    answer_index: int
    metadata: QuestionMetadata


class AnalogyGenerator:
    def __init__(self, data: List[Dict], min_distractors: int = 3, n_choices: int = 4):
        self.data = data
        self.min_distractors = min_distractors
        self.n_choices = n_choices
        self.lemma_lookup, self.semantic_relations = self._build_lemma_lookup()

    def _build_lemma_lookup(self) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, Set[str]]]]:
        """Build lookup tables for lemmas and semantic relations."""
        lemma_lookup = defaultdict(set)
        semantic_relations = defaultdict(lambda: defaultdict(set))

        for entry in self.data:
            # Add main translations
            for lang_code, trans in entry.get("translations", {}).items():
                lemma_lookup[lang_code].add(trans["lemma"])

            # Build semantic relation mappings
            for rel_type in ["hypernyms", "hyponyms", "meronyms", "holonyms", "cohyponyms"]:
                for rel_entry in entry.get(rel_type, []):
                    for lang_code, trans in rel_entry.get("translations", {}).items():
                        lemma_lookup[lang_code].add(trans["lemma"])
                        semantic_relations[lang_code][rel_type].add(trans["lemma"])

        return lemma_lookup, semantic_relations

    @staticmethod
    def _get_lang_info(lang_code: str) -> Tuple[Optional[str], Optional[str]]:
        """Get language name and resource level for a language code."""
        for level, langs in LANGUAGE_CONFIG.items():
            for lang, v in langs.items():
                if v["code"] == lang_code:
                    return v["name"], level
        return None, None

    @staticmethod
    def _get_languages_by_resource(resource_level: str) -> List[str]:
        """Get all language codes for a given resource level."""
        return [v["code"] for v in LANGUAGE_CONFIG[resource_level].values()]

    def _get_language_pairs(self, mode: MultilingualMode) -> Tuple[List[str], List[str]]:
        """Get source and target languages based on multilingual mode."""
        if mode == MultilingualMode.EN_TO_HIGH:
            target_languages = [lang for lang in self._get_languages_by_resource("high_resource") if lang != "en"]
            from_languages = ["en"]
        elif mode == MultilingualMode.EN_TO_MEDIUM:
            target_languages = self._get_languages_by_resource("medium_resource")
            from_languages = ["en"]
        elif mode == MultilingualMode.EN_TO_LOW:
            target_languages = self._get_languages_by_resource("low_resource")
            from_languages = ["en"]
        elif mode == MultilingualMode.EN_TO_ALL:
            target_languages = (
                    self._get_languages_by_resource("high_resource") +
                    self._get_languages_by_resource("medium_resource") +
                    self._get_languages_by_resource("low_resource")
            )
            target_languages = [lang for lang in target_languages if lang != "en"]
            from_languages = ["en"]
        elif mode == MultilingualMode.MONOLINGUAL_EN:
            target_languages = ["en"]
            from_languages = ["en"]
        else:  # ALL
            all_languages = (
                    self._get_languages_by_resource("high_resource") +
                    self._get_languages_by_resource("medium_resource") +
                    self._get_languages_by_resource("low_resource")
            )
            target_languages = all_languages
            from_languages = all_languages

        return from_languages, target_languages

    def _generate_distractors(self, correct_lemma: str, all_candidates: Set[str],
                              target_lang: str, relation_type: str,
                              difficulty: DifficultyLevel) -> Tuple[List[str], str]:
        """Generate distractors based on difficulty level using strategy pattern."""
        strategies = {
            DifficultyLevel.RANDOM: self._random_distractors,
            DifficultyLevel.MIXED: self._mixed_distractors,
            DifficultyLevel.SEMANTIC: self._semantic_distractors,
            DifficultyLevel.CLOSE_SEMANTIC: self._close_semantic_distractors,
            DifficultyLevel.VERY_CLOSE: self._very_close_distractors
        }

        return strategies[difficulty](correct_lemma, all_candidates, target_lang, relation_type)

    def _random_distractors(self, correct_lemma: str, all_candidates: Set[str],
                            target_lang: str, relation_type: str) -> Tuple[List[str], str]:
        """Generate random distractors."""
        distractors = set(random.sample(list(all_candidates),
                                        min(self.n_choices - 1, len(all_candidates))))
        return self._finalize_distractors(distractors, correct_lemma, all_candidates), "random_unrelated"

    def _mixed_distractors(self, correct_lemma: str, all_candidates: Set[str],
                           target_lang: str, relation_type: str) -> Tuple[List[str], str]:
        """Generate mixed random and semantic distractors."""
        random_words = set(random.sample(list(all_candidates),
                                         min(self.n_choices - 2, len(all_candidates))))
        semantic_words = set()

        if target_lang in self.semantic_relations:
            cohyponyms = self.semantic_relations[target_lang].get("cohyponyms", set())
            if cohyponyms:
                semantic_words = set(random.sample(list(cohyponyms), min(1, len(cohyponyms))))

        distractors = random_words.union(semantic_words)
        return self._finalize_distractors(distractors, correct_lemma, all_candidates), "mixed_random_semantic"

    def _semantic_distractors(self, correct_lemma: str, all_candidates: Set[str],
                              target_lang: str, relation_type: str) -> Tuple[List[str], str]:
        """Generate semantically related distractors."""
        distractors = set()

        if target_lang in self.semantic_relations:
            # Use relation-specific distractors
            semantic_pool = self.semantic_relations[target_lang].get(relation_type, set())
            semantic_pool.update(self.semantic_relations[target_lang].get("cohyponyms", set()))

            if len(semantic_pool) >= self.n_choices - 1:
                distractors = set(random.sample(list(semantic_pool), self.n_choices - 1))
            else:
                distractors = semantic_pool.union(
                    set(random.sample(list(all_candidates),
                                      min(self.n_choices - 1 - len(semantic_pool), len(all_candidates))))
                )
        else:
            distractors = set(random.sample(list(all_candidates), self.n_choices - 1))

        return self._finalize_distractors(distractors, correct_lemma, all_candidates), "semantically_related"

    def _close_semantic_distractors(self, correct_lemma: str, all_candidates: Set[str],
                                    target_lang: str, relation_type: str) -> Tuple[List[str], str]:
        """Generate close semantic match distractors."""
        close_matches = set()

        if target_lang in self.semantic_relations:
            # Use the same relation type for close matches
            close_matches.update(self.semantic_relations[target_lang].get(relation_type, set()))
            close_matches.update(self.semantic_relations[target_lang].get("cohyponyms", set()))

        if len(close_matches) >= self.n_choices - 1:
            distractors = set(random.sample(list(close_matches), self.n_choices - 1))
        else:
            semantic_pool = self.semantic_relations[target_lang].get("hypernyms", set())
            needed = self.n_choices - 1 - len(close_matches)
            additional = set(random.sample(list(semantic_pool), min(needed, len(semantic_pool))))
            distractors = close_matches | additional

        return self._finalize_distractors(distractors, correct_lemma, all_candidates), "close_semantic_matches"

    def _very_close_distractors(self, correct_lemma: str, all_candidates: Set[str],
                                target_lang: str, relation_type: str) -> Tuple[List[str], str]:
        """Generate very close match distractors."""
        very_close_matches = set()

        if target_lang in self.semantic_relations:
            # Use the exact same relation type for very close matches
            very_close_matches.update(self.semantic_relations[target_lang].get(relation_type, set()))
            very_close_matches.update(self.semantic_relations[target_lang].get("meronyms", set()))

        if len(very_close_matches) >= self.n_choices - 1:
            distractors = set(random.sample(list(very_close_matches), self.n_choices - 1))
        else:
            remaining_needed = self.n_choices - 1 - len(very_close_matches)
            other_semantic = (
                    self.semantic_relations[target_lang].get("hyponyms", set()) |
                    self.semantic_relations[target_lang].get("hypernyms", set())
            )
            additional = set(random.sample(list(other_semantic), min(remaining_needed, len(other_semantic))))
            distractors = very_close_matches | additional

        return self._finalize_distractors(distractors, correct_lemma, all_candidates), "very_close_matches"

    def _finalize_distractors(self, distractors: Set[str], correct_lemma: str,
                              all_candidates: Set[str]) -> List[str]:
        """Finalize distractor set by removing correct answer and filling gaps."""
        distractors.discard(correct_lemma)

        while len(distractors) < (self.n_choices - 1):
            remaining = all_candidates - distractors - {correct_lemma}
            if remaining:
                distractors.add(random.choice(list(remaining)))
            else:
                break

        return list(distractors)

    def _create_prompt_text(self, from_code: str, to_code: str, A_lemma: str,
                            B_lemma: str, C_lemma: str) -> Tuple[str, str]:
        """Create prompt text for the analogy question."""
        from_lang_name, _ = self._get_lang_info(from_code)
        to_lang_name, _ = self._get_lang_info(to_code)

        prompt = (
            f"Complete the analogy:\n\n"
            f"{A_lemma} ({from_lang_name}) is to {B_lemma} ({from_lang_name})\n"
            f"as\n"
            f"{C_lemma} ({to_lang_name}) is to ____?\n\n"
            f"Choose the correct option in {to_lang_name}:"
        )

        return prompt, "en"

    def _collect_valid_entries(self, from_languages: List[str],
                               target_languages: List[str]) -> Dict[str, List[Tuple]]:
        """Collect valid entries organized by language pairs."""
        valid_entries = defaultdict(list)

        for entry in self.data:
            # Check if entry has any semantic relations
            candidate_relations = [rel for rel in ["hypernyms", "hyponyms", "meronyms", "holonyms", "cohyponyms"]
                                   if entry.get(rel)]
            if not candidate_relations:
                continue

            for from_code in from_languages:
                if from_code not in entry.get("translations", {}):
                    continue

                for to_code in target_languages:
                    if from_code == to_code and from_code != "en":
                        continue

                    # Check if we have valid relation entries for this language pair
                    valid_relations = []
                    for rel_type in candidate_relations:
                        for rel_entry in entry.get(rel_type, []):
                            if from_code in rel_entry.get("translations", {}):
                                valid_relations.append((rel_type, rel_entry))

                    if valid_relations:
                        lang_pair = f"{from_code}_to_{to_code}"
                        valid_entries[lang_pair].append((entry, valid_relations))

        return valid_entries

    def _find_analogy_pair(self, relation_type: str, to_code: str,
                           used_pairs: Set[Tuple[str, str]]) -> Optional[Tuple[Dict, Dict]]:
        """Find a second analogy pair for the given relation type."""
        candidates = []

        for entry in self.data:
            if not entry.get(relation_type):
                continue

            if to_code not in entry.get("translations", {}):
                continue

            for rel_entry in entry.get(relation_type, []):
                if to_code not in rel_entry.get("translations", {}):
                    continue

                C_lemma = entry["translations"][to_code]["lemma"]
                D_lemma = rel_entry["translations"][to_code]["lemma"]

                # Check if this pair was already used
                if (C_lemma, D_lemma) not in used_pairs and (D_lemma, C_lemma) not in used_pairs:
                    candidates.append((entry, rel_entry))

        if candidates:
            return random.choice(candidates)
        return None

    def _generate_balanced_questions(self, valid_entries: Dict,
                                     target_questions_per_pair: int,
                                     multilingual_mode: str) -> List[Question]:
        """Generate balanced questions across language pairs and difficulty levels."""
        questions = []
        qid = 0
        generation_time = datetime.utcnow().isoformat() + "Z"

        # Calculate questions per difficulty level
        num_difficulties = len(DifficultyLevel)
        questions_per_difficulty = target_questions_per_pair // num_difficulties
        remaining_questions = target_questions_per_pair % num_difficulties

        for lang_pair, entries in valid_entries.items():
            from_code, to_code = lang_pair.split("_to_")
            random.shuffle(entries)

            # Track used analogy pairs to prevent repetitions
            used_pairs = set()
            lang_pair_questions = []

            for difficulty_enum in DifficultyLevel:
                target_for_difficulty = questions_per_difficulty
                if difficulty_enum.value <= remaining_questions:
                    target_for_difficulty += 1

                questions_generated = 0
                entry_idx = 0

                while questions_generated < target_for_difficulty and entry_idx < len(entries):
                    entry, valid_relations = entries[entry_idx]

                    # Try to generate a question
                    question = self._create_single_question(
                        entry, valid_relations, from_code, to_code,
                        difficulty_enum, qid, generation_time,
                        multilingual_mode, used_pairs
                    )

                    if question:
                        lang_pair_questions.append(question)
                        qid += 1
                        questions_generated += 1

                    entry_idx += 1

            # Add all questions for this language pair
            questions.extend(lang_pair_questions)

        return questions

    def _create_single_question(self, entry: Dict, valid_relations: List[Tuple],
                                from_code: str, to_code: str,
                                difficulty: DifficultyLevel, qid: int,
                                generation_time: str, multilingual_mode: str,
                                used_pairs: Set[Tuple[str, str]]) -> Optional[Question]:
        """Create a single analogy question from entry data."""
        try:
            # Pick a semantic relation to use
            relation_type, relation_entry = random.choice(valid_relations)

            # Get A and B (first pair)
            A_lemma = entry["translations"][from_code]["lemma"]
            B_lemma = relation_entry["translations"][from_code]["lemma"]

            # Find second analogy pair (C, D)
            second_pair = self._find_analogy_pair(relation_type, to_code, used_pairs)
            if not second_pair:
                return None

            second_entry, second_relation_entry = second_pair
            C_lemma = second_entry["translations"][to_code]["lemma"]
            D_correct_lemma = second_relation_entry["translations"][to_code]["lemma"]

            # Add this pair to used pairs
            used_pairs.add((C_lemma, D_correct_lemma))

            # Generate distractors
            all_candidates = self.lemma_lookup[to_code] - {D_correct_lemma}
            if len(all_candidates) < self.min_distractors:
                return None

            distractors, distractor_type = self._generate_distractors(
                D_correct_lemma, all_candidates, to_code, relation_type, difficulty
            )

            options = distractors + [D_correct_lemma]
            random.shuffle(options)
            answer_index = options.index(D_correct_lemma)

            # Create prompt
            prompt_text, prompt_lang_code = self._create_prompt_text(
                from_code, to_code, A_lemma, B_lemma, C_lemma
            )

            from_resource = self._get_lang_info(from_code)[1]
            to_resource = self._get_lang_info(to_code)[1]
            resource_pair = f"{from_resource}_to_{to_resource}"

            metadata = QuestionMetadata(
                resource_pair=resource_pair,
                prompt_lang=prompt_lang_code,
                from_lang=from_code,
                to_lang=to_code,
                difficulty=difficulty.value,
                distractor_type=distractor_type,
                generation_time=generation_time,
                synset_id=entry.get("synset_id", ""),
                multilingual_mode=multilingual_mode,
                relation_type=relation_type
            )

            return Question(
                id=f"analogy_{qid}_{from_code}_to_{to_code}_diff{difficulty.value}",
                prompt=prompt_text,
                options=options,
                answer_index=answer_index,
                metadata=metadata
            )

        except Exception as e:
            logger.warning(f"Failed to create analogy question for {from_code} to {to_code}: {e}")
            return None

    def generate_analogies(self, output_filename: str,
                           multilingual_mode: MultilingualMode = MultilingualMode.ALL,
                           target_questions_per_pair: int = 100) -> None:
        """Generate analogy questions for all language pairs."""
        print(f"Generating analogy questions ({multilingual_mode.value})...")

        from_languages, target_languages = self._get_language_pairs(multilingual_mode)

        # Collect valid entries
        valid_entries = self._collect_valid_entries(from_languages, target_languages)

        if not valid_entries:
            print(f"No valid entries found for analogies with mode {multilingual_mode.value}")
            return

        # Generate questions
        questions = self._generate_balanced_questions(
            valid_entries, target_questions_per_pair, multilingual_mode.value
        )

        # Convert to dict format for JSON serialization
        questions_dict = [
            {
                "id": q.id,
                "prompt": q.prompt,
                "options": q.options,
                "answer_index": q.answer_index,
                "metadata": {
                    "resource_pair": q.metadata.resource_pair,
                    "prompt_lang": q.metadata.prompt_lang,
                    "from_lang": q.metadata.from_lang,
                    "to_lang": q.metadata.to_lang,
                    "difficulty": q.metadata.difficulty,
                    "distractor_type": q.metadata.distractor_type,
                    "generation_time": q.metadata.generation_time,
                    "synset_id": q.metadata.synset_id,
                    "multilingual_mode": q.metadata.multilingual_mode,
                    "relation_type": q.metadata.relation_type
                }
            }
            for q in questions
        ]

        # Save to file
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(questions_dict, f, indent=2, ensure_ascii=False)

        # Print minimal statistics
        print(f"Generated {len(questions_dict)} questions for {len(valid_entries)} language pairs")
        print(f"Saved to: {output_filename}")


def main():
    """Main function to run analogy question generation."""
    # Load data
    print("Loading data...")
    with open("../GeneratedFiles/JsonFiles/multilingual_babelnet_relations.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Initialize generator
    generator = AnalogyGenerator(data)

    # Define modes and their target question counts
    modes = [
        (MultilingualMode.EN_TO_HIGH, 100),
        (MultilingualMode.EN_TO_MEDIUM, 100),
        (MultilingualMode.EN_TO_LOW, 100),
        (MultilingualMode.MONOLINGUAL_EN, 400),
        (MultilingualMode.ALL, 10)
    ]

    # Generate questions for each mode
    for mode, target_questions in modes:
        output_file = f"../GeneratedFiles/JsonFiles/Analogies/semantic_analogy_questions_{mode.value}.json"

        generator.generate_analogies(
            output_filename=output_file,
            multilingual_mode=mode,
            target_questions_per_pair=target_questions
        )

    print("Analogy question generation complete!")


if __name__ == "__main__":
    main()



        # Loading data...
        # Generating analogy questions (en_to_high)...
        # Generated 2400 questions for 24 language pairs
        # Saved to: ../GeneratedFiles/JsonFiles/Analogies/semantic_analogy_questions_en_to_high.json
        # Generating analogy questions (en_to_medium)...
        # Generated 1500 questions for 15 language pairs
        # Saved to: ../GeneratedFiles/JsonFiles/Analogies/semantic_analogy_questions_en_to_medium.json
        # Generating analogy questions (en_to_low)...
        # Generated 1000 questions for 10 language pairs
        # Saved to: ../GeneratedFiles/JsonFiles/Analogies/semantic_analogy_questions_en_to_low.json
        # Generating analogy questions (monolingual_en)...
        # Generated 400 questions for 1 language pairs
        # Saved to: ../GeneratedFiles/JsonFiles/Analogies/semantic_analogy_questions_monolingual_en.json
        # Generating analogy questions (all)...
        # Generated 24510 questions for 2451 language pairs
        # Saved to: ../GeneratedFiles/JsonFiles/Analogies/semantic_analogy_questions_all.json
        # Analogy question generation complete!