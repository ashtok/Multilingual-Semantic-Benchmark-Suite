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
    RANDOM_CROSS_DOMAIN = 1  # Random words from completely different synsets
    DISTANT_HYPERNYMS = 2  # Words from hypernyms of hypernyms (grandfather concepts)
    DIRECT_RELATIONS = 3  # Direct hypernyms or hyponyms of the target
    COHYPONYMS = 4  # Sister concepts (often hardest conceptually)
    CLOSE_RELATIONS = 5  # Meronyms or synsets with shared hypernyms


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


@dataclass
class Question:
    id: str
    prompt: str
    options: List[str]
    answer_index: int
    metadata: QuestionMetadata


class QuestionGenerator:
    def __init__(self, data: List[Dict], min_distractors: int = 3, n_choices: int = 4):
        self.data = data
        self.min_distractors = min_distractors
        self.n_choices = n_choices
        self.synset_lookup = self._build_synset_lookup()
        self.lemma_lookup, self.semantic_relations = self._build_lemma_lookup()
        self.synset_to_entry = {entry.get("synset_id"): entry for entry in data if entry.get("synset_id")}

    def _build_synset_lookup(self) -> Dict[str, Dict]:
        """Build lookup table for synsets."""
        return {entry.get("synset_id"): entry for entry in self.data if entry.get("synset_id")}

    def _build_lemma_lookup(self) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, Set[str]]]]:
        """Build lookup tables for lemmas and semantic relations."""
        lemma_lookup = defaultdict(set)
        semantic_relations = defaultdict(lambda: defaultdict(set))

        for entry in self.data:
            # Add main translations
            for lang_code, trans in entry.get("translations", {}).items():
                lemma_lookup[lang_code].add(trans["lemma"])

            # Build semantic relation mappings with synset awareness
            for rel_type in ["hypernyms", "hyponyms", "meronyms", "cohyponyms"]:
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

    def _get_distant_hypernyms(self, entry: Dict, target_lang: str) -> Set[str]:
        """Get hypernyms of hypernyms (grandfather concepts)."""
        distant_hypernyms = set()

        # Get direct hypernyms
        for hypernym_entry in entry.get("hypernyms", []):
            hypernym_synset_id = hypernym_entry.get("synset_id")
            if hypernym_synset_id and hypernym_synset_id in self.synset_to_entry:
                # Get hypernyms of this hypernym (grandfather concepts)
                grandfather_entry = self.synset_to_entry[hypernym_synset_id]
                for grandfather_hypernym in grandfather_entry.get("hypernyms", []):
                    if target_lang in grandfather_hypernym.get("translations", {}):
                        distant_hypernyms.add(grandfather_hypernym["translations"][target_lang]["lemma"])

        return distant_hypernyms

    def _get_shared_hypernym_synsets(self, entry: Dict, target_lang: str) -> Set[str]:
        """Get synsets that share hypernyms with the current synset."""
        shared_hypernym_words = set()

        # Get our hypernyms
        our_hypernyms = set()
        for hypernym_entry in entry.get("hypernyms", []):
            hypernym_synset_id = hypernym_entry.get("synset_id")
            if hypernym_synset_id:
                our_hypernyms.add(hypernym_synset_id)

        # Find other synsets that share these hypernyms
        for other_entry in self.data:
            if other_entry.get("synset_id") == entry.get("synset_id"):
                continue  # Skip self

            for other_hypernym in other_entry.get("hypernyms", []):
                other_hypernym_id = other_hypernym.get("synset_id")
                if other_hypernym_id in our_hypernyms:
                    # This synset shares a hypernym with us
                    if target_lang in other_entry.get("translations", {}):
                        shared_hypernym_words.add(other_entry["translations"][target_lang]["lemma"])

        return shared_hypernym_words

    def _build_semantic_exclusion_set(self, entry: Dict) -> Set[str]:
        """Build comprehensive set of all semantically related synset IDs to exclude."""
        excluded_synsets = set()
        current_synset_id = entry.get("synset_id")
        excluded_synsets.add(current_synset_id)

        # Add all direct semantic relations
        for rel_type in ["hypernyms", "hyponyms", "meronyms", "cohyponyms"]:
            for rel_entry in entry.get(rel_type, []):
                excluded_synsets.add(rel_entry.get("synset_id"))

        # Add multi-level hypernyms (up to 3 levels)
        self._add_recursive_hypernyms(entry, excluded_synsets, max_depth=3)

        # Add multi-level hyponyms (down to 2 levels)
        self._add_recursive_hyponyms(entry, excluded_synsets, max_depth=2)

        # Add synsets that share any hypernyms with our synset
        our_hypernyms = set()
        for hypernym_entry in entry.get("hypernyms", []):
            our_hypernyms.add(hypernym_entry.get("synset_id"))

        # Find synsets sharing these hypernyms
        for other_entry in self.data:
            other_synset_id = other_entry.get("synset_id")
            if other_synset_id == current_synset_id:
                continue

            for other_hypernym in other_entry.get("hypernyms", []):
                if other_hypernym.get("synset_id") in our_hypernyms:
                    excluded_synsets.add(other_synset_id)
                    break

        return excluded_synsets

    def _add_recursive_hypernyms(self, entry: Dict, excluded_set: Set[str], max_depth: int, current_depth: int = 0):
        """Recursively add hypernyms up to max_depth levels."""
        if current_depth >= max_depth:
            return

        for hypernym_entry in entry.get("hypernyms", []):
            hypernym_synset_id = hypernym_entry.get("synset_id")
            if hypernym_synset_id and hypernym_synset_id not in excluded_set:
                excluded_set.add(hypernym_synset_id)
                # Recurse to next level
                if hypernym_synset_id in self.synset_to_entry:
                    self._add_recursive_hypernyms(
                        self.synset_to_entry[hypernym_synset_id],
                        excluded_set,
                        max_depth,
                        current_depth + 1
                    )

    def _add_recursive_hyponyms(self, entry: Dict, excluded_set: Set[str], max_depth: int, current_depth: int = 0):
        """Recursively add hyponyms down to max_depth levels."""
        if current_depth >= max_depth:
            return

        for hyponym_entry in entry.get("hyponyms", []):
            hyponym_synset_id = hyponym_entry.get("synset_id")
            if hyponym_synset_id and hyponym_synset_id not in excluded_set:
                excluded_set.add(hyponym_synset_id)
                # Recurse to next level
                if hyponym_synset_id in self.synset_to_entry:
                    self._add_recursive_hyponyms(
                        self.synset_to_entry[hyponym_synset_id],
                        excluded_set,
                        max_depth,
                        current_depth + 1
                    )

    def _get_cross_domain_words(self, entry: Dict, target_lang: str, sample_size: int = 50) -> Set[str]:
        """Get words from completely different semantic domains with comprehensive exclusion."""
        # Build comprehensive exclusion set
        excluded_synsets = self._build_semantic_exclusion_set(entry)

        # Find truly cross-domain candidates
        candidate_words = []
        for other_entry in self.data:
            other_synset_id = other_entry.get("synset_id")

            # Skip if this synset is in our exclusion set
            if other_synset_id in excluded_synsets:
                continue

            # Additional check: ensure no shared vocabulary
            if self._shares_vocabulary_with_target(entry, other_entry, target_lang):
                continue

            # This synset appears to be truly cross-domain
            if target_lang in other_entry.get("translations", {}):
                candidate_words.append(other_entry["translations"][target_lang]["lemma"])

        # Sample to avoid memory issues and ensure diversity
        if len(candidate_words) > sample_size:
            cross_domain_words = set(random.sample(candidate_words, sample_size))
        else:
            cross_domain_words = set(candidate_words)

        return cross_domain_words

    def _shares_vocabulary_with_target(self, target_entry: Dict, candidate_entry: Dict, lang: str) -> bool:
        """Check if candidate synset shares vocabulary/lemmas with target synset's semantic field."""
        if lang not in target_entry.get("translations", {}) or lang not in candidate_entry.get("translations", {}):
            return False

        target_lemma = target_entry["translations"][lang]["lemma"].lower()
        candidate_lemma = candidate_entry["translations"][lang]["lemma"].lower()

        # Check for substring relationships (e.g., "dog" in "bulldog")
        if target_lemma in candidate_lemma or candidate_lemma in target_lemma:
            return True

        # Check if they share significant word stems (basic version)
        if len(target_lemma) > 3 and len(candidate_lemma) > 3:
            if target_lemma[:3] == candidate_lemma[:3]:  # Same 3-letter prefix
                return True

        return False

    def _generate_distractors(self, correct_lemma: str, all_candidates: Set[str],
                              target_lang: str, entry: Dict, difficulty: DifficultyLevel) -> Tuple[List[str], str]:
        """Generate distractors based on semantic hierarchy difficulty levels."""

        if difficulty == DifficultyLevel.RANDOM_CROSS_DOMAIN:
            return self._random_cross_domain_distractors(correct_lemma, all_candidates, target_lang, entry)
        elif difficulty == DifficultyLevel.DISTANT_HYPERNYMS:
            return self._distant_hypernym_distractors(correct_lemma, all_candidates, target_lang, entry)
        elif difficulty == DifficultyLevel.DIRECT_RELATIONS:
            return self._direct_relation_distractors(correct_lemma, all_candidates, target_lang, entry)
        elif difficulty == DifficultyLevel.COHYPONYMS:
            return self._cohyponym_distractors(correct_lemma, all_candidates, target_lang, entry)
        elif difficulty == DifficultyLevel.CLOSE_RELATIONS:
            return self._close_relation_distractors(correct_lemma, all_candidates, target_lang, entry)
        else:
            # Fallback to random
            return self._random_cross_domain_distractors(correct_lemma, all_candidates, target_lang, entry)

    def _random_cross_domain_distractors(self, correct_lemma: str, all_candidates: Set[str],
                                         target_lang: str, entry: Dict) -> Tuple[List[str], str]:
        """Level 1: Random words from completely different semantic domains."""
        cross_domain_words = self._get_cross_domain_words(entry, target_lang)

        if len(cross_domain_words) >= self.n_choices - 1:
            distractors = set(random.sample(list(cross_domain_words), self.n_choices - 1))
        else:
            # Fallback to any random words if not enough cross-domain words
            remaining_needed = self.n_choices - 1 - len(cross_domain_words)
            additional = set(random.sample(list(all_candidates),
                                           min(remaining_needed, len(all_candidates))))
            distractors = cross_domain_words | additional

        return self._finalize_distractors(distractors, correct_lemma, all_candidates), "cross_domain_random"

    def _distant_hypernym_distractors(self, correct_lemma: str, all_candidates: Set[str],
                                      target_lang: str, entry: Dict) -> Tuple[List[str], str]:
        """Level 2: Words from hypernyms of hypernyms (grandfather concepts)."""
        distant_hypernyms = self._get_distant_hypernyms(entry, target_lang)

        if len(distant_hypernyms) >= self.n_choices - 1:
            distractors = set(random.sample(list(distant_hypernyms), self.n_choices - 1))
        else:
            # Add some cross-domain words to fill gaps
            cross_domain = self._get_cross_domain_words(entry, target_lang, 20)
            available = distant_hypernyms | cross_domain
            if len(available) >= self.n_choices - 1:
                distractors = set(random.sample(list(available), self.n_choices - 1))
            else:
                remaining = self.n_choices - 1 - len(available)
                additional = set(random.sample(list(all_candidates), min(remaining, len(all_candidates))))
                distractors = available | additional

        return self._finalize_distractors(distractors, correct_lemma, all_candidates), "distant_hypernyms"

    def _direct_relation_distractors(self, correct_lemma: str, all_candidates: Set[str],
                                     target_lang: str, entry: Dict) -> Tuple[List[str], str]:
        """Level 3: Direct hypernyms or hyponyms of the target."""
        direct_relations = set()

        # Add direct hypernyms
        for hypernym_entry in entry.get("hypernyms", []):
            if target_lang in hypernym_entry.get("translations", {}):
                direct_relations.add(hypernym_entry["translations"][target_lang]["lemma"])

        # Add direct hyponyms
        for hyponym_entry in entry.get("hyponyms", []):
            if target_lang in hyponym_entry.get("translations", {}):
                direct_relations.add(hyponym_entry["translations"][target_lang]["lemma"])

        if len(direct_relations) >= self.n_choices - 1:
            distractors = set(random.sample(list(direct_relations), self.n_choices - 1))
        else:
            # Add distant hypernyms to fill gaps
            distant = self._get_distant_hypernyms(entry, target_lang)
            available = direct_relations | distant
            if len(available) >= self.n_choices - 1:
                distractors = set(random.sample(list(available), self.n_choices - 1))
            else:
                remaining = self.n_choices - 1 - len(available)
                additional = set(random.sample(list(all_candidates), min(remaining, len(all_candidates))))
                distractors = available | additional

        return self._finalize_distractors(distractors, correct_lemma, all_candidates), "direct_relations"

    def _cohyponym_distractors(self, correct_lemma: str, all_candidates: Set[str],
                               target_lang: str, entry: Dict) -> Tuple[List[str], str]:
        """Level 4: Sister concepts (cohyponyms) - often hardest conceptually."""
        cohyponyms = set()

        # Add direct cohyponyms
        for cohyponym_entry in entry.get("cohyponyms", []):
            if target_lang in cohyponym_entry.get("translations", {}):
                cohyponyms.add(cohyponym_entry["translations"][target_lang]["lemma"])

        if len(cohyponyms) >= self.n_choices - 1:
            distractors = set(random.sample(list(cohyponyms), self.n_choices - 1))
        else:
            # Add synsets sharing hypernyms
            shared_hypernym_words = self._get_shared_hypernym_synsets(entry, target_lang)
            available = cohyponyms | shared_hypernym_words
            if len(available) >= self.n_choices - 1:
                distractors = set(random.sample(list(available), self.n_choices - 1))
            else:
                # Add direct relations as fallback
                direct_relations = set()
                for hypernym_entry in entry.get("hypernyms", []):
                    if target_lang in hypernym_entry.get("translations", {}):
                        direct_relations.add(hypernym_entry["translations"][target_lang]["lemma"])

                available = available | direct_relations
                if len(available) >= self.n_choices - 1:
                    distractors = set(random.sample(list(available), self.n_choices - 1))
                else:
                    remaining = self.n_choices - 1 - len(available)
                    additional = set(random.sample(list(all_candidates), min(remaining, len(all_candidates))))
                    distractors = available | additional

        return self._finalize_distractors(distractors, correct_lemma, all_candidates), "cohyponyms"

    def _close_relation_distractors(self, correct_lemma: str, all_candidates: Set[str],
                                    target_lang: str, entry: Dict) -> Tuple[List[str], str]:
        """Level 5: Meronyms or synsets with shared hypernyms - very close relations."""
        close_relations = set()

        # Add meronyms (part-whole relationships)
        for meronym_entry in entry.get("meronyms", []):
            if target_lang in meronym_entry.get("translations", {}):
                close_relations.add(meronym_entry["translations"][target_lang]["lemma"])

        # Add synsets sharing direct hypernyms
        shared_hypernym_words = self._get_shared_hypernym_synsets(entry, target_lang)
        close_relations = close_relations | shared_hypernym_words

        if len(close_relations) >= self.n_choices - 1:
            distractors = set(random.sample(list(close_relations), self.n_choices - 1))
        else:
            # Add cohyponyms as fallback
            cohyponyms = set()
            for cohyponym_entry in entry.get("cohyponyms", []):
                if target_lang in cohyponym_entry.get("translations", {}):
                    cohyponyms.add(cohyponym_entry["translations"][target_lang]["lemma"])

            available = close_relations | cohyponyms
            if len(available) >= self.n_choices - 1:
                distractors = set(random.sample(list(available), self.n_choices - 1))
            else:
                remaining = self.n_choices - 1 - len(available)
                additional = set(random.sample(list(all_candidates), min(remaining, len(all_candidates))))
                distractors = available | additional

        return self._finalize_distractors(distractors, correct_lemma, all_candidates), "close_relations"

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

    def _create_prompt_text(self, task_type: str, from_code: str, to_code: str,
                            prompt_word: str) -> Tuple[str, str]:
        """Create prompt text for the question."""
        from_lang_name, _ = self._get_lang_info(from_code)
        to_lang_name, _ = self._get_lang_info(to_code)

        relation_phrases = {
            "hypernymy": "a hypernym (broader category)",
            "meronymy": "a meronym (part, component, or member)"
        }

        relation_phrase = relation_phrases.get(task_type, "a semantic relation")

        prompt = (f"Which of the following is {relation_phrase} of the {from_lang_name} "
                  f"word \"{prompt_word}\"? (Options in {to_lang_name}.)")

        return prompt, "en"

    def _collect_valid_entries(self, relation_field: str, from_languages: List[str],
                               target_languages: List[str]) -> Dict[str, List[Tuple]]:
        """Collect valid entries organized by language pairs."""
        valid_entries = defaultdict(list)

        for entry in self.data:
            if not entry.get(relation_field):
                continue

            for from_code in from_languages:
                if from_code not in entry.get("translations", {}):
                    continue

                # Filter related entries with target language translations
                related_entries = [
                    rel_entry for rel_entry in entry.get(relation_field, [])
                    if any(to_code in rel_entry.get("translations", {}) for to_code in target_languages)
                ]

                if not related_entries:
                    continue

                for to_code in target_languages:
                    if from_code == to_code and from_code != "en":
                        continue

                    valid_related_entries = [
                        rel_entry for rel_entry in related_entries
                        if to_code in rel_entry.get("translations", {})
                    ]

                    if valid_related_entries:
                        lang_pair = f"{from_code}_to_{to_code}"
                        valid_entries[lang_pair].append((entry, valid_related_entries))

        return valid_entries

    def _generate_balanced_questions(self, valid_entries: Dict, task_type: str,
                                     relation_field: str, target_questions_per_pair: int,
                                     multilingual_mode: str) -> List[Question]:
        """Generate balanced questions across language pairs and difficulty levels with no word repetitions."""
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

            # Track used source words for this language pair to prevent repetitions
            used_source_words = set()
            lang_pair_questions = []

            for difficulty_enum in DifficultyLevel:
                target_for_difficulty = questions_per_difficulty
                if difficulty_enum.value <= remaining_questions:
                    target_for_difficulty += 1

                questions_generated = 0
                entry_idx = 0

                while questions_generated < target_for_difficulty and entry_idx < len(entries):
                    entry, valid_related_entries = entries[entry_idx]

                    # Get source word (prompt word)
                    source_word = entry["translations"][from_code]["lemma"]

                    # Skip if this source word was already used in this language pair
                    if source_word in used_source_words:
                        entry_idx += 1
                        continue

                    # Generate question
                    question = self._create_single_question(
                        entry, valid_related_entries, from_code, to_code,
                        task_type, relation_field, difficulty_enum,
                        qid, generation_time, multilingual_mode
                    )

                    if question:
                        lang_pair_questions.append(question)
                        used_source_words.add(source_word)
                        qid += 1
                        questions_generated += 1

                    entry_idx += 1

            # Add all questions for this language pair
            questions.extend(lang_pair_questions)

        return questions

    def _create_single_question(self, entry: Dict, valid_related_entries: List[Dict],
                                from_code: str, to_code: str, task_type: str,
                                relation_field: str, difficulty: DifficultyLevel,
                                qid: int, generation_time: str, multilingual_mode: str) -> Optional[Question]:
        """Create a single question from entry data."""
        try:
            prompt_word = entry["translations"][from_code]["lemma"]
            relation_entry = random.choice(valid_related_entries)
            correct_lemma = relation_entry["translations"][to_code]["lemma"]

            all_candidates = self.lemma_lookup[to_code] - {correct_lemma}
            if len(all_candidates) < self.min_distractors:
                return None

            distractors, distractor_type = self._generate_distractors(
                correct_lemma, all_candidates, to_code, entry, difficulty
            )

            options = distractors + [correct_lemma]
            random.shuffle(options)
            answer_index = options.index(correct_lemma)

            prompt_text, prompt_lang_code = self._create_prompt_text(
                task_type, from_code, to_code, prompt_word
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
                multilingual_mode=multilingual_mode
            )

            return Question(
                id=f"{task_type}_{qid}_{from_code}_to_{to_code}_diff{difficulty.value}",
                prompt=prompt_text,
                options=options,
                answer_index=answer_index,
                metadata=metadata
            )

        except Exception as e:
            logger.warning(f"Failed to create question for {from_code} to {to_code}: {e}")
            return None

    def generate_task(self, task_type: str, relation_field: str, output_filename: str,
                      multilingual_mode: MultilingualMode = MultilingualMode.ALL,
                      target_questions_per_pair: int = 100) -> None:
        """Generate questions for a specific task type."""
        print(f"Generating {task_type} questions ({multilingual_mode.value})...")

        from_languages, target_languages = self._get_language_pairs(multilingual_mode)

        # Collect valid entries
        valid_entries = self._collect_valid_entries(relation_field, from_languages, target_languages)

        if not valid_entries:
            print(f"No valid entries found for {task_type} with mode {multilingual_mode.value}")
            return

        # Generate questions
        questions = self._generate_balanced_questions(
            valid_entries, task_type, relation_field,
            target_questions_per_pair, multilingual_mode.value
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
                    "multilingual_mode": q.metadata.multilingual_mode
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
    """Main function to run question generation."""
    # Load data
    print("Loading data...")
    with open("../GeneratedFiles/JsonFiles/multilingual_babelnet_relations.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Initialize generator
    generator = QuestionGenerator(data)

    # Define tasks to generate
    tasks = [
        ("hypernymy", "hypernyms", "Hypernymy"),
        ("meronymy", "meronyms", "Meronymy")
    ]

    modes = [
        MultilingualMode.EN_TO_HIGH,
        MultilingualMode.EN_TO_MEDIUM,
        MultilingualMode.EN_TO_LOW,
        MultilingualMode.MONOLINGUAL_EN,
        MultilingualMode.EN_TO_ALL,
        MultilingualMode.ALL
    ]

    # Generate questions for each task and mode
    for task_type, relation_field, task_name in tasks:
        for mode in modes:
            if mode == MultilingualMode.MONOLINGUAL_EN:
                target_questions = 1100
            elif mode == MultilingualMode.ALL:
                target_questions = 2
            else:
                target_questions = 5

            output_file = f"../GeneratedFiles/JsonFiles/{task_name}/{task_type}_questions_{mode.value}.json"

            generator.generate_task(
                task_type=task_type,
                relation_field=relation_field,
                output_filename=output_file,
                multilingual_mode=mode,
                target_questions_per_pair=target_questions
            )

    print("Question generation complete!")


if __name__ == "__main__":
    main()


# Generating hypernymy questions (en_to_high)...
# Generated 120 questions for 24 language pairs
# Saved to: ../GeneratedFiles/JsonFiles/Hypernymy/hypernymy_questions_en_to_high.json
# Generating hypernymy questions (en_to_medium)...
# Generated 75 questions for 15 language pairs
# Saved to: ../GeneratedFiles/JsonFiles/Hypernymy/hypernymy_questions_en_to_medium.json
# Generating hypernymy questions (en_to_low)...
# Generated 50 questions for 10 language pairs
# Saved to: ../GeneratedFiles/JsonFiles/Hypernymy/hypernymy_questions_en_to_low.json
# Generating hypernymy questions (monolingual_en)...
# Generated 1087 questions for 1 language pairs
# Saved to: ../GeneratedFiles/JsonFiles/Hypernymy/hypernymy_questions_monolingual_en.json
# Generating hypernymy questions (en_to_all)...
# Generated 245 questions for 49 language pairs
# Saved to: ../GeneratedFiles/JsonFiles/Hypernymy/hypernymy_questions_en_to_all.json
# Generating hypernymy questions (all)...
# Generated 4902 questions for 2451 language pairs
# Saved to: ../GeneratedFiles/JsonFiles/Hypernymy/hypernymy_questions_all.json
# Generating meronymy questions (en_to_high)...
# Generated 120 questions for 24 language pairs
# Saved to: ../GeneratedFiles/JsonFiles/Meronymy/meronymy_questions_en_to_high.json
# Generating meronymy questions (en_to_medium)...
# Generated 75 questions for 15 language pairs
# Saved to: ../GeneratedFiles/JsonFiles/Meronymy/meronymy_questions_en_to_medium.json
# Generating meronymy questions (en_to_low)...
# Generated 50 questions for 10 language pairs
# Saved to: ../GeneratedFiles/JsonFiles/Meronymy/meronymy_questions_en_to_low.json
# Generating meronymy questions (monolingual_en)...
# Generated 1087 questions for 1 language pairs
# Saved to: ../GeneratedFiles/JsonFiles/Meronymy/meronymy_questions_monolingual_en.json
# Generating meronymy questions (en_to_all)...
# Generated 245 questions for 49 language pairs
# Saved to: ../GeneratedFiles/JsonFiles/Meronymy/meronymy_questions_en_to_all.json
# Generating meronymy questions (all)...
# Generated 4902 questions for 2451 language pairs
# Saved to: ../GeneratedFiles/JsonFiles/Meronymy/meronymy_questions_all.json
# Question generation complete!