import json
import random
from collections import defaultdict
from datetime import datetime
from babelnet import Language
from language_config import LANGUAGE_CONFIG

NUM_QUESTIONS_PER_TYPE = 20  # adjust as you like

# Load your JSON data
with open("babelnet_relations_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)


# ---------------------------
# Helpers
# ---------------------------
DISTRACTOR_STRATEGY = {
    "hypernymy": ["cohyponyms", "hyponyms"],
    "meronymy": ["cohyponyms", "hyponyms", "hypernyms"],
    # No entries for hyponymy or cohyponymy tasks — they’re skipped
}

ONLY_EN_TO_EN = True  # Set to True to generate only English-to-English questions

def build_lemma_lookup(data):
    """
    Builds a dictionary mapping:
      lemma_lookup[lang_code] -> set(all lemmas)
    Also builds semantic relation mappings for better distractor selection
    """
    lemma_lookup = defaultdict(set)
    semantic_relations = defaultdict(lambda: defaultdict(set))

    for entry in data:
        synset_id = entry.get("synset_id", "")

        # Add main translations
        for lang_code, trans in entry.get("translations", {}).items():
            lemma_lookup[lang_code].add(trans["lemma"])

        # Build semantic relation mappings for distractor selection
        for rel_type in ["hypernyms", "hyponyms", "meronyms", "cohyponyms"]:
            for rel_entry in entry.get(rel_type, []):
                for lang_code, trans in rel_entry.get("translations", {}).items():
                    lemma_lookup[lang_code].add(trans["lemma"])
                    semantic_relations[lang_code][rel_type].add(trans["lemma"])

    return lemma_lookup, semantic_relations


def pick_resource_level(lang_code):
    for level, langs in LANGUAGE_CONFIG.items():
        for lang, v in langs.items():
            if v["code"] == lang_code:
                return level
    return None


def get_lang_name(lang_code):
    for level, langs in LANGUAGE_CONFIG.items():
        for lang, v in langs.items():
            if v["code"] == lang_code:
                return v["name"]
    return None


def pick_language_pair():
    levels = list(LANGUAGE_CONFIG.keys())
    from_level = random.choice(levels)
    to_level = random.choice(levels)

    from_lang = random.choice(list(LANGUAGE_CONFIG[from_level].values()))
    to_lang = random.choice(list(LANGUAGE_CONFIG[to_level].values()))

    return from_lang, to_lang, f"{from_level}_to_{to_level}"


def generate_options(correct_lemma, all_candidates, semantic_relations,
                     target_lang, entry, relation_field, n_choices=4, difficulty=3):
    distractors = set()

    # Get allowed semantic relations for this task type (relation_field)
    allowed_relations = DISTRACTOR_STRATEGY.get(relation_field, [])

    if difficulty == 1:
        # Easy: Random unrelated words
        distractors = set(random.sample(list(all_candidates), min(n_choices - 1, len(all_candidates))))
        distractor_type = "random_unrelated"

    elif difficulty == 2:
        # Medium-Easy: Mix of random and allowed semantic relations
        random_words = set(random.sample(list(all_candidates), min(n_choices - 2, len(all_candidates))))

        semantic_words = set()
        for rel in allowed_relations:
            semantic_set = semantic_relations.get(target_lang, {}).get(rel, set())
            if semantic_set:
                semantic_words.update(random.sample(list(semantic_set), min(1, len(semantic_set))))

        distractors = random_words.union(semantic_words)
        distractor_type = "mixed_random_semantic"

    elif difficulty == 3:
        # Medium: Semantically related words from allowed relations
        semantic_pool = set()
        for rel in allowed_relations:
            semantic_pool.update(semantic_relations.get(target_lang, {}).get(rel, set()))

        if len(semantic_pool) >= n_choices - 1:
            distractors = set(random.sample(list(semantic_pool), n_choices - 1))
        else:
            distractors = semantic_pool.union(
                set(random.sample(list(all_candidates), n_choices - 1 - len(semantic_pool)))
            )
        distractor_type = "semantically_related"

    elif difficulty == 4:
        # Hard: Close semantic matches
        close_matches = set()

        # Also include hypernyms and hyponyms if allowed
        for rel in allowed_relations:
            close_matches.update(semantic_relations.get(target_lang, {}).get(rel, set()))

        if len(close_matches) >= n_choices - 1:
            distractors = set(random.sample(list(close_matches), n_choices - 1))
        else:
            distractors = close_matches.union(
                set(random.sample(list(all_candidates), n_choices - 1 - len(close_matches)))
            )
        distractor_type = "close_semantic_matches"

    else:  # difficulty == 5
        # Very Hard: Very close or confusing matches
        very_close_matches = set()

        # Add meronyms if allowed and available
        if "meronyms" in allowed_relations:
            very_close_matches.update(semantic_relations.get(target_lang, {}).get("meronyms", set()))

        # Add cohyponyms if allowed and available
        if "cohyponyms" in allowed_relations:
            very_close_matches.update(semantic_relations.get(target_lang, {}).get("cohyponyms", set()))

        if len(very_close_matches) >= n_choices - 1:
            distractors = set(random.sample(list(very_close_matches), n_choices - 1))
        else:
            remaining_needed = n_choices - 1 - len(very_close_matches)
            other_semantic = set()
            for rel in allowed_relations:
                other_semantic.update(semantic_relations.get(target_lang, {}).get(rel, set()))
            distractors = very_close_matches.union(
                set(random.sample(list(other_semantic), min(remaining_needed, len(other_semantic))))
            )
        distractor_type = "very_close_matches"

    # Ensure correct lemma not included and enough distractors
    distractors.discard(correct_lemma)
    while len(distractors) < (n_choices - 1):
        remaining = all_candidates - distractors - {correct_lemma}
        if remaining:
            distractors.add(random.choice(list(remaining)))
        else:
            break

    return list(distractors), distractor_type


# ---------------------------
# Task Generation
# ---------------------------

lemma_lookup, semantic_relations = build_lemma_lookup(data)
generation_time = datetime.utcnow().isoformat() + "Z"


def generate_task(task_type, relation_field, output_filename):
    """
    Generate tasks of a given semantic relation:
        - task_type: "hypernymy", "meronymy"
        - relation_field: e.g. "hypernyms", "meronyms"
    """
    questions = []
    qid = 0

    # Track statistics
    difficulty_stats = defaultdict(int)
    resource_pair_stats = defaultdict(int)
    prompt_language_stats = defaultdict(int)

    while len(questions) < NUM_QUESTIONS_PER_TYPE:
        entry = random.choice(data)

        if not entry.get(relation_field):
            continue

        from_lang, to_lang, resource_pair = pick_language_pair()
        from_code = from_lang["code"]
        to_code = to_lang["code"]

        if ONLY_EN_TO_EN:
            if from_code != "en" or to_code != "en":
                continue  # skip non English-English pairs

        if from_code not in entry["translations"]:
            continue

        prompt_word = entry["translations"][from_code]["lemma"]

        # Pick one relation target
        relation_entry = random.choice(entry[relation_field])
        if to_code not in relation_entry["translations"]:
            continue

        correct_lemma = relation_entry["translations"][to_code]["lemma"]

        all_candidates = lemma_lookup[to_code] - {correct_lemma}
        if len(all_candidates) < 3:
            continue

        difficulty_level = random.randint(1, 5)
        distractors, distractor_type = generate_options(
            correct_lemma,
            all_candidates,
            semantic_relations,
            to_code,
            entry,
            relation_field,
            difficulty=difficulty_level
        )

        options = distractors + [correct_lemma]
        random.shuffle(options)
        answer_index = options.index(correct_lemma)

        # Decide on prompt language (English vs. source language)
        # 50% chance of English, 50% chance of source language
        use_english_prompt = random.choice([True, False])

        if use_english_prompt:
            # English prompt
            if task_type == "hypernymy":
                relation_phrase = "a hypernym (broader category)"
            elif task_type == "meronymy":
                relation_phrase = "a meronym (part, component, or member)"

            prompt_text = f"Which of the following is {relation_phrase} of the {get_lang_name(from_code)} word \"{prompt_word}\"? (Options in {get_lang_name(to_code)}.)"
            prompt_lang_code = "en"
        else:
            # Source language prompt (or fallback to Spanish if source lang prompts not available)
            if from_code == "es":
                if task_type == "hypernymy":
                    relation_phrase = "un hiperónimo (categoría más amplia)"
                elif task_type == "meronymy":
                    relation_phrase = "un merónimo (parte, componente, o miembro)"
                prompt_text = f"¿Cuál de las siguientes es {relation_phrase} de la palabra \"{prompt_word}\"? (Opciones en {get_lang_name(to_code)}.)"
                prompt_lang_code = "es"
            else:
                # Fallback to English for non-Spanish source languages
                if task_type == "hypernymy":
                    relation_phrase = "a hypernym (broader category)"
                elif task_type == "meronymy":
                    relation_phrase = "a meronym (part, component, or member)"
                prompt_text = f"Which of the following is {relation_phrase} of the {get_lang_name(from_code)} word \"{prompt_word}\"? (Options in {get_lang_name(to_code)}.)"
                prompt_lang_code = "en"

        question = {
            "id": f"{task_type}_{qid}_{from_code}_to_{to_code}",
            "prompt": prompt_text,
            "options": options,
            "answer_index": answer_index,
            "explanation": f"{correct_lemma} is the {task_type} of {prompt_word}; the others are {distractor_type}.",
            "metadata": {
                "prompt_language": prompt_lang_code,
                "word_language": from_code,
                "options_language": to_code,
                "root_lemma": prompt_word,
                "correct_answer": correct_lemma,
                "synset_id": entry.get("synset_id", ""),
                "resource_pairing": resource_pair,
                "difficulty_level": difficulty_level,
                "semantic_relation_type": task_type,
                "language_resource_levels": {
                    "word_language_level": pick_resource_level(from_code),
                    "options_language_level": pick_resource_level(to_code)
                },
                "cross_lingual_type": (
                    "intra-lingual" if from_code == to_code
                    else "cross-lingual"
                ),
                "prompt_type": (
                    "english_prompt" if prompt_lang_code == "en"
                    else "source_language_prompt"
                ),
                "resource_level_transfer": resource_pair,
                "num_options": len(options),
                "language_pair_code": f"{from_code}-{to_code}",
                "distractor_type": distractor_type,
                "generation_timestamp": generation_time,
                "concept_name": entry.get("gloss", ""),
                "source_entry_id": entry.get("synset_id", ""),
                "distractor_difficulty": {
                    "1": "random_unrelated",
                    "2": "mixed_random_semantic",
                    "3": "semantically_related",
                    "4": "close_semantic_matches",
                    "5": "very_close_matches"
                }[str(difficulty_level)]
            }
        }
        questions.append(question)
        qid += 1

        # Update statistics
        difficulty_stats[difficulty_level] += 1
        resource_pair_stats[resource_pair] += 1
        prompt_language_stats[prompt_lang_code] += 1

    # Save JSONL
    with open(output_filename, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"✔️ Generated {len(questions)} {task_type} tasks → saved to {output_filename}")
    print(f"   Difficulty distribution: {dict(difficulty_stats)}")
    print(f"   Resource pair distribution: {dict(resource_pair_stats)}")
    print(f"   Prompt language distribution: {dict(prompt_language_stats)}")


# Run only hypernymy and meronymy tasks
relations_to_generate = [
    ("hypernymy", "hypernyms"),
    ("meronymy", "meronyms")
]

for task_type, relation_field in relations_to_generate:
    generate_task(
        task_type=task_type,
        relation_field=relation_field,
        output_filename=f"{task_type}_tasks_detailed.jsonl"
    )

print("✅ All tasks generated successfully.")