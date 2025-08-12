import json
import random
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
from enum import Enum
from language_config import LANGUAGE_CONFIG
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class DifficultyLevel(Enum):
    RANDOM = 1
    SAME_DOMAIN = 2
    CLOSE = 3


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


class GlossQuestionGenerator:
    def __init__(self, data: List[Dict], min_distractors: int = 3, n_choices: int = 4):
        self.data = data
        self.min_distractors = min_distractors
        self.n_choices = n_choices
        self.lemma_lookup = self._build_lemma_lookup()

    def _build_lemma_lookup(self) -> Dict[str, Set[str]]:
        lemma_lookup = defaultdict(set)
        for entry in self.data:
            for lang_code, trans in entry.get("translations", {}).items():
                lemma_lookup[lang_code].add(trans["lemma"])
        return lemma_lookup

    @staticmethod
    def _get_lang_info(lang_code: str) -> Tuple[Optional[str], Optional[str]]:
        for level, langs in LANGUAGE_CONFIG.items():
            for lang, v in langs.items():
                if v["code"] == lang_code:
                    return v["name"], level
        return None, None

    @staticmethod
    def _get_languages_by_resource(resource_level: str) -> List[str]:
        return [v["code"] for v in LANGUAGE_CONFIG[resource_level].values()]

    def _get_language_pairs(self, mode: MultilingualMode) -> Tuple[List[str], List[str]]:
        if mode == MultilingualMode.EN_TO_HIGH:
            return ["en"], [lang for lang in self._get_languages_by_resource("high_resource") if lang != "en"]
        elif mode == MultilingualMode.EN_TO_MEDIUM:
            return ["en"], self._get_languages_by_resource("medium_resource")
        elif mode == MultilingualMode.EN_TO_LOW:
            return ["en"], self._get_languages_by_resource("low_resource")
        elif mode == MultilingualMode.EN_TO_ALL:
            all_targets = (
                self._get_languages_by_resource("high_resource") +
                self._get_languages_by_resource("medium_resource") +
                self._get_languages_by_resource("low_resource")
            )
            return ["en"], [lang for lang in all_targets if lang != "en"]
        elif mode == MultilingualMode.MONOLINGUAL_EN:
            return ["en"], ["en"]
        else:
            all_langs = (
                self._get_languages_by_resource("high_resource") +
                self._get_languages_by_resource("medium_resource") +
                self._get_languages_by_resource("low_resource")
            )
            return all_langs, all_langs

    def _collect_valid_entries(self, from_languages: List[str], target_languages: List[str]) -> Dict[str, List[Dict]]:
        valid_entries = defaultdict(list)
        for entry in self.data:
            if "glossary" not in entry or not entry["glossary"]:
                continue

            for from_code in from_languages:
                if from_code not in entry["glossary"]:
                    continue

                for to_code in target_languages:
                    if to_code not in entry["translations"]:
                        continue
                    if from_code == to_code and from_code != "en":
                        continue
                    valid_entries[f"{from_code}_to_{to_code}"].append(entry)

        return valid_entries

    def _generate_distractors(self, correct_lemma: str, all_candidates: Set[str], difficulty: DifficultyLevel) -> Tuple[List[str], str]:
        filtered = all_candidates - {correct_lemma}
        if difficulty == DifficultyLevel.RANDOM:
            distractors = random.sample(list(filtered), min(self.n_choices - 1, len(filtered)))
            return distractors, "random"
        else:
            # For now, same as random, can be improved with semantic proximity
            distractors = random.sample(list(filtered), min(self.n_choices - 1, len(filtered)))
            return distractors, "random"

    def _create_prompt_text(self, gloss_text: str, from_lang: str, to_lang: str) -> Tuple[str, str]:
        from_name, _ = self._get_lang_info(from_lang)
        to_name, _ = self._get_lang_info(to_lang)
        if from_lang == to_lang:
            return f"Definition: {gloss_text}\n\nWhich word matches this definition?", "en"
        else:
            return f"Definition ({from_name}): {gloss_text}\n\nChoose the correct word in {to_name}:", "en"

    def generate_gloss_questions(self, output_filename: str, multilingual_mode: MultilingualMode, target_questions_per_pair: int = 50) -> None:
        print(f"Generating gloss-based questions ({multilingual_mode.value})...")
        from_langs, to_langs = self._get_language_pairs(multilingual_mode)
        valid_entries = self._collect_valid_entries(from_langs, to_langs)

        if not valid_entries:
            print(f"No valid entries for {multilingual_mode.value}")
            return

        questions = []
        qid = 0
        generation_time = datetime.utcnow().isoformat() + "Z"

        for lang_pair, entries in valid_entries.items():
            from_code, to_code = lang_pair.split("_to_")
            random.shuffle(entries)
            used_glosses = set()
            questions_per_diff = target_questions_per_pair // len(DifficultyLevel)

            for difficulty in DifficultyLevel:
                generated_count = 0
                for entry in entries:
                    gloss_info = entry["glossary"].get(from_code)
                    if not gloss_info:
                        continue
                    gloss_text = gloss_info.get("text", "").strip()
                    if not gloss_text or gloss_text in used_glosses:
                        continue

                    correct_lemma = entry["translations"][to_code]["lemma"]
                    all_candidates = self.lemma_lookup[to_code]
                    if len(all_candidates) < self.min_distractors:
                        continue

                    distractors, distractor_type = self._generate_distractors(correct_lemma, all_candidates, difficulty)
                    if len(distractors) < self.n_choices - 1:
                        continue

                    options = distractors + [correct_lemma]
                    random.shuffle(options)
                    answer_index = options.index(correct_lemma)

                    prompt_text, prompt_lang_code = self._create_prompt_text(gloss_text, from_code, to_code)
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
                        multilingual_mode=multilingual_mode.value
                    )

                    questions.append(Question(
                        id=f"gloss_{qid}_{from_code}_to_{to_code}_diff{difficulty.value}",
                        prompt=prompt_text,
                        options=options,
                        answer_index=answer_index,
                        metadata=metadata
                    ))
                    used_glosses.add(gloss_text)
                    qid += 1
                    generated_count += 1
                    if generated_count >= questions_per_diff:
                        break

        questions_dict = [
            {
                "id": q.id,
                "prompt": q.prompt,
                "options": q.options,
                "answer_index": q.answer_index,
                "metadata": vars(q.metadata)
            }
            for q in questions
        ]

        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(questions_dict, f, indent=2, ensure_ascii=False)

        print(f"Generated {len(questions_dict)} questions across {len(valid_entries)} language pairs")
        print(f"Saved to: {output_filename}")


def main():
    with open("../GeneratedFiles/JsonFiles/multilingual_babelnet_relations.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    generator = GlossQuestionGenerator(data)

    modes = [
        MultilingualMode.EN_TO_HIGH,
        MultilingualMode.EN_TO_MEDIUM,
        MultilingualMode.EN_TO_LOW,
        MultilingualMode.MONOLINGUAL_EN,
        MultilingualMode.ALL
    ]

    for mode in modes:
        output_file = f"../GeneratedFiles/JsonFiles/Gloss/gloss_questions_{mode.value}.json"
        generator.generate_gloss_questions(output_file, mode, target_questions_per_pair=50)


if __name__ == "__main__":
    main()
