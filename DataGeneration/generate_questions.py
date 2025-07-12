import json
import random
from collections import defaultdict
from datetime import datetime
from language_config import LANGUAGE_CONFIG
from tqdm import tqdm

# Load your JSON data
with open("../GeneratedFiles/JsonFiles/multilingual_babelnet_relations.json", "r", encoding="utf-8") as f:
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


def get_lang_name(lang_code):
    for level, langs in LANGUAGE_CONFIG.items():
        for lang, v in langs.items():
            if v["code"] == lang_code:
                return v["name"]
    return None


def get_resource_level(lang_code):
    for level, langs in LANGUAGE_CONFIG.items():
        for lang, v in langs.items():
            if v["code"] == lang_code:
                return level
    return None


def get_languages_by_resource(resource_level):
    """Get all language codes for a given resource level"""
    return [v["code"] for v in LANGUAGE_CONFIG[resource_level].values()]


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


def create_prompt_text(task_type, from_code, to_code, prompt_word):
    # Always use English prompts
    if task_type == "hypernymy":
        relation_phrase = "a hypernym (broader category)"
    elif task_type == "meronymy":
        relation_phrase = "a meronym (part, component, or member)"
    else:
        relation_phrase = "a semantic relation"

    return f"Which of the following is {relation_phrase} of the {get_lang_name(from_code)} word \"{prompt_word}\"? (Options in {get_lang_name(to_code)}.)", "en"


def collect_valid_entries(data, relation_field, from_languages, target_languages):
    """Collect all valid entries organized by language pairs for even distribution"""
    valid_entries = defaultdict(list)

    for entry in data:
        if not entry.get(relation_field):
            continue

        for from_code in from_languages:
            if from_code not in entry.get("translations", {}):
                continue

            # Filter related entries that have translations in target languages
            related_entries = [
                rel_entry for rel_entry in entry.get(relation_field, [])
                if any(to_code in rel_entry.get("translations", {}) for to_code in target_languages)
            ]

            if not related_entries:
                continue

            for to_code in target_languages:
                if from_code == to_code and from_code != "en":  # Allow monolingual English
                    continue

                # Filter related entries that have translation in this target language
                valid_related_entries = [
                    rel_entry for rel_entry in related_entries
                    if to_code in rel_entry.get("translations", {})
                ]

                if valid_related_entries:
                    lang_pair = f"{from_code}_to_{to_code}"
                    valid_entries[lang_pair].append((entry, valid_related_entries))

    return valid_entries


def generate_balanced_questions(valid_entries, lemma_lookup, semantic_relations, task_type,
                                relation_field, target_questions_per_pair, multilingual_mode):
    """Generate questions with even distribution across language pairs"""
    questions = []
    qid = 0
    generation_time = datetime.utcnow().isoformat() + "Z"

    for lang_pair, entries in valid_entries.items():
        from_code, to_code = lang_pair.split("_to_")
        questions_generated = 0

        # Shuffle entries for random sampling
        random.shuffle(entries)

        # Generate questions for this language pair
        entry_idx = 0
        while questions_generated < target_questions_per_pair and entry_idx < len(entries):
            entry, valid_related_entries = entries[entry_idx]

            prompt_word = entry["translations"][from_code]["lemma"]

            # Pick one related entry as correct answer
            relation_entry = random.choice(valid_related_entries)
            correct_lemma = relation_entry["translations"][to_code]["lemma"]

            all_candidates = lemma_lookup[to_code] - {correct_lemma}
            if len(all_candidates) < 3:
                entry_idx += 1
                continue

            # Generate questions with different difficulty levels
            for difficulty_level in range(1, 6):
                if questions_generated >= target_questions_per_pair:
                    break

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

                # Determine resource pair
                from_resource = get_resource_level(from_code)
                to_resource = get_resource_level(to_code)
                resource_pair = f"{from_resource}_to_{to_resource}"

                question = {
                    "id": f"{task_type}_{qid}_{from_code}_to_{to_code}_diff{difficulty_level}",
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
                        "synset_id": entry.get("synset_id", ""),
                        "multilingual_mode": multilingual_mode
                    }
                }
                questions.append(question)
                qid += 1
                questions_generated += 1

            entry_idx += 1

    return questions


# ---------------------------
# Task Generation
# ---------------------------

def generate_task(task_type, relation_field, output_filename, multilingual_mode="all",
                  target_questions_per_pair=100):
    """
    Generate questions for each word in the JSON data with even distribution.

    Args:
        task_type: Type of task ("hypernymy", "meronymy", etc.)
        relation_field: Field in JSON containing relations ("hypernyms", "meronyms", etc.)
        output_filename: Output file path
        multilingual_mode: Controls language pairs
        target_questions_per_pair: Target number of questions per language pair
    """

    lemma_lookup, semantic_relations = build_lemma_lookup(data)

    # Define language pairs based on multilingual mode
    if multilingual_mode == "en_to_high":
        target_languages = get_languages_by_resource("high_resource")
        target_languages = [lang for lang in target_languages if lang != "en"]
        from_languages = ["en"]
    elif multilingual_mode == "en_to_medium":
        target_languages = get_languages_by_resource("medium_resource")
        from_languages = ["en"]
    elif multilingual_mode == "en_to_low":
        target_languages = get_languages_by_resource("low_resource")
        from_languages = ["en"]
    elif multilingual_mode == "en_to_all":
        target_languages = (get_languages_by_resource("high_resource") +
                            get_languages_by_resource("medium_resource") +
                            get_languages_by_resource("low_resource"))
        target_languages = [lang for lang in target_languages if lang != "en"]
        from_languages = ["en"]
    elif multilingual_mode == "monolingual_en":
        target_languages = ["en"]
        from_languages = ["en"]
    else:  # "all" - original behavior
        all_languages = (get_languages_by_resource("high_resource") +
                         get_languages_by_resource("medium_resource") +
                         get_languages_by_resource("low_resource"))
        target_languages = all_languages
        from_languages = all_languages

    # Collect valid entries organized by language pairs
    print(f"Collecting valid entries for {task_type}...")
    valid_entries = collect_valid_entries(data, relation_field, from_languages, target_languages)

    if not valid_entries:
        print(f"No valid entries found for {task_type} with mode {multilingual_mode}")
        return

    print(f"Found {len(valid_entries)} language pairs with valid data")

    # Generate balanced questions
    print(f"Generating balanced questions...")
    questions = generate_balanced_questions(
        valid_entries, lemma_lookup, semantic_relations, task_type,
        relation_field, target_questions_per_pair, multilingual_mode
    )

    # Save questions to output file
    with open(output_filename, "w", encoding="utf-8") as f_out:
        json.dump(questions, f_out, indent=2, ensure_ascii=False)

    # Print summary statistics
    language_pairs = defaultdict(int)
    resource_pairs = defaultdict(int)

    for q in questions:
        lang_pair = f"{q['metadata']['from_lang']}_to_{q['metadata']['to_lang']}"
        language_pairs[lang_pair] += 1
        resource_pairs[q["metadata"]["resource_pair"]] += 1

    print(f"\n=== {task_type.upper()} Questions Generated (mode: {multilingual_mode}) ===")
    print(f"Total questions: {len(questions)}")
    print(f"Language pairs: {len(language_pairs)}")

    print(f"\nQuestions per language pair:")
    for pair, count in sorted(language_pairs.items()):
        print(f"  {pair}: {count}")

    print(f"\nResource pair distribution:")
    for pair, count in sorted(resource_pairs.items()):
        print(f"  {pair}: {count}")

    print(f"\nSaved to: {output_filename}")
    print("=" * 60)


# Example usage with balanced generation:

# Generate hypernymy questions with English to high-resource languages
generate_task("hypernymy", "hypernyms",
              "../GeneratedFiles/JsonFiles/Hypernymy/hypernymy_questions_en_to_high.json",
              multilingual_mode="en_to_high",
              target_questions_per_pair=100)

# Generate meronymy questions with English to medium-resource languages
generate_task("meronymy", "meronyms",
              "../GeneratedFiles/JsonFiles/Meronymy/meronymy_questions_en_to_medium.json",
              multilingual_mode="en_to_medium",
              target_questions_per_pair=100)

# Generate hypernymy questions with English to low-resource languages
generate_task("hypernymy", "hypernyms",
              "../GeneratedFiles/JsonFiles/Hypernymy/hypernymy_questions_en_to_low.json",
              multilingual_mode="en_to_low",
              target_questions_per_pair=100)

# Generate monolingual English questions
generate_task("hypernymy", "hypernyms",
              "../GeneratedFiles/JsonFiles/Hypernymy/hypernymy_questions_monolingual_en.json",
              multilingual_mode="monolingual_en",
              target_questions_per_pair=100)

# Generate all language pairs (original behavior)
generate_task("hypernymy", "hypernyms",
              "../GeneratedFiles/JsonFiles/Hypernymy/hypernymy_questions_all.json",
              multilingual_mode="all",
              target_questions_per_pair=50)  # Lower target for "all" mode due to many pairs