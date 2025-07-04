import json
import random
from collections import defaultdict
from datetime import datetime
from babelnet import Language
from language_config import LANGUAGE_CONFIG

NUM_QUESTIONS_PER_TYPE = 5000   # adjust as you like

# Load your JSON data
with open("babelnet_relations_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)


# ---------------------------
# Helpers
# ---------------------------

def build_lemma_lookup(data):
    lemma_lookup = defaultdict(set)
    semantic_relations = defaultdict(lambda: defaultdict(set))

    for entry in data:
        synset_id = entry.get("synset_id", "")

        # Add main translations
        for lang_code, trans in entry.get("translations", {}).items():
            lemma_lookup[lang_code].add(trans["lemma"])

        # Build semantic relation mappings for distractors
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

    if difficulty == 1:
        distractors = set(random.sample(list(all_candidates), min(n_choices - 1, len(all_candidates))))
        distractor_type = "random_unrelated"

    elif difficulty == 2:
        random_words = set(random.sample(list(all_candidates), min(n_choices - 2, len(all_candidates))))
        semantic_words = set()
        if target_lang in semantic_relations and "cohyponyms" in semantic_relations[target_lang]:
            cohyponyms = semantic_relations[target_lang]["cohyponyms"]
            if cohyponyms:
                semantic_words = set(random.sample(list(cohyponyms), min(1, len(cohyponyms))))
        distractors = random_words.union(semantic_words)
        distractor_type = "mixed_random_semantic"

    elif difficulty == 3:
        if target_lang in semantic_relations:
            cohyponyms = semantic_relations[target_lang].get("cohyponyms", set())
            if len(cohyponyms) >= n_choices - 1:
                distractors = set(random.sample(list(cohyponyms), n_choices - 1))
            else:
                other_relations = semantic_relations[target_lang].get("hyponyms", set()).union(
                    semantic_relations[target_lang].get("hypernyms", set())
                )
                available = cohyponyms.union(other_relations)
                if len(available) >= n_choices - 1:
                    distractors = set(random.sample(list(available), n_choices - 1))
                else:
                    distractors = available.union(
                        set(random.sample(list(all_candidates), n_choices - 1 - len(available)))
                    )
        else:
            distractors = set(random.sample(list(all_candidates), n_choices - 1))
        distractor_type = "semantically_related"

    elif difficulty == 4:
        close_matches = set()
        if target_lang in semantic_relations and "hyponyms" in semantic_relations[target_lang]:
            close_matches.update(semantic_relations[target_lang]["hyponyms"])
        for hypernym_entry in entry.get("hypernyms", []):
            if target_lang in hypernym_entry.get("translations", {}):
                close_matches.add(hypernym_entry["translations"][target_lang]["lemma"])
        if len(close_matches) >= n_choices - 1:
            distractors = set(random.sample(list(close_matches), n_choices - 1))
        else:
            semantic_pool = semantic_relations[target_lang].get("cohyponyms", set())
            distractors = close_matches.union(
                set(random.sample(list(semantic_pool), min(n_choices - 1 - len(close_matches), len(semantic_pool))))
            )
        distractor_type = "close_semantic_matches"

    else:  # difficulty == 5
        very_close_matches = set()
        if target_lang in semantic_relations:
            very_close_matches.update(semantic_relations[target_lang].get("meronyms", set()))
            cohyponyms = semantic_relations[target_lang].get("cohyponyms", set())
            very_close_matches.update(cohyponyms)
        if len(very_close_matches) >= n_choices - 1:
            distractors = set(random.sample(list(very_close_matches), n_choices - 1))
        else:
            remaining_needed = n_choices - 1 - len(very_close_matches)
            other_semantic = semantic_relations[target_lang].get("hyponyms", set()).union(
                semantic_relations[target_lang].get("hypernyms", set())
            )
            distractors = very_close_matches.union(
                set(random.sample(list(other_semantic), min(remaining_needed, len(other_semantic))))
            )
        distractor_type = "very_close_matches"

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


def create_prompt_text(task_type, from_code, to_code, prompt_word):
    # Compose prompt text based on language and relation
    if task_type == "hypernymy":
        relation_phrase_en = "a hypernym (broader category)"
        relation_phrase_es = "un hiperónimo (categoría más amplia)"
    elif task_type == "meronymy":
        relation_phrase_en = "a meronym (part, component, or member)"
        relation_phrase_es = "un merónimo (parte, componente, o miembro)"
    else:
        relation_phrase_en = "a semantic relation"
        relation_phrase_es = "una relación semántica"

    if from_code == "es":
        return f"¿Cuál de las siguientes es {relation_phrase_es} de la palabra \"{prompt_word}\"? (Opciones en {get_lang_name(to_code)}.)", "es"
    else:
        return f"Which of the following is {relation_phrase_en} of the {get_lang_name(from_code)} word \"{prompt_word}\"? (Options in {get_lang_name(to_code)}.)", "en"


def generate_task(task_type, relation_field, output_filename):
    questions = []
    qid = 0

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

        if from_code not in entry["translations"]:
            continue

        # Filter related entries that have translation in target language
        related_entries = [
            rel_entry for rel_entry in entry.get(relation_field, [])
            if to_code in rel_entry.get("translations", {})
        ]
        if not related_entries:
            continue

        prompt_word = entry["translations"][from_code]["lemma"]

        # Pick one related entry as correct answer
        relation_entry = random.choice(related_entries)
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

        # Generate prompt text
        prompt_text, prompt_lang_code = create_prompt_text(task_type, from_code, to_code, prompt_word)

        # --- Build the original question ---
        question = {
            "id": f"{task_type}_{qid}_{from_code}_to_{to_code}",
            "prompt": prompt_text,
            "options": options,
            "answer_index": answer_index,
            "metadata": {
                "resource_pair": resource_pair,
                "prompt_lang": prompt_lang_code,
                "from_lang": from_code,
                "to_lang": to_code,
                "difficulty": difficulty_level,
                "distractor_type": distractor_type,
                "generation_time": generation_time,
            }
        }
        questions.append(question)

        # --- Build the English to English question for the same prompt word ---
        # Make sure English translations exist
        if "en" in entry["translations"]:
            en_prompt_word = entry["translations"]["en"]["lemma"]

            # Pick English related entries for correct answer
            en_related_entries = [
                rel_entry for rel_entry in entry.get(relation_field, [])
                if "en" in rel_entry.get("translations", {})
            ]
            if en_related_entries:
                en_relation_entry = random.choice(en_related_entries)
                en_correct_lemma = en_relation_entry["translations"]["en"]["lemma"]

                en_all_candidates = lemma_lookup["en"] - {en_correct_lemma}
                if len(en_all_candidates) >= 3:
                    en_distractors, en_distractor_type = generate_options(
                        en_correct_lemma,
                        en_all_candidates,
                        semantic_relations,
                        "en",
                        entry,
                        relation_field,
                        difficulty=difficulty_level
                    )
                    en_options = en_distractors + [en_correct_lemma]
                    random.shuffle(en_options)
                    en_answer_index = en_options.index(en_correct_lemma)

                    en_prompt_text, en_prompt_lang_code = create_prompt_text(task_type, "en", "en", en_prompt_word)

                    en_question = {
                        "id": f"{task_type}_{qid}_en_to_en",
                        "prompt": en_prompt_text,
                        "options": en_options,
                        "answer_index": en_answer_index,
                        "metadata": {
                            "resource_pair": "en_to_en",
                            "prompt_lang": en_prompt_lang_code,
                            "from_lang": "en",
                            "to_lang": "en",
                            "difficulty": difficulty_level,
                            "distractor_type": en_distractor_type,
                            "generation_time": generation_time,
                        }
                    }
                    questions.append(en_question)

        qid += 1

    # Save questions to output file
    with open(output_filename, "w", encoding="utf-8") as f_out:
        json.dump(questions, f_out, indent=2, ensure_ascii=False)

    print(f"Generated {len(questions)} questions for {task_type}, saved to {output_filename}")


# Generate hypernymy questions
generate_task("hypernymy", "hypernyms", "hypernymy_questions.json")

# Generate meronymy questions
generate_task("meronymy", "meronyms", "meronymy_questions.json")

# Similarly for other relations...

