import json
import random
from collections import defaultdict
from datetime import datetime
from babelnet import Language
from language_config import LANGUAGE_CONFIG

NUM_ANALOGY_QUESTIONS = 5000   # Adjust as needed

# ---------------------------
# Load Data
# ---------------------------

with open("babelnet_relations_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ---------------------------
# Reuse your existing helpers
# ---------------------------

def build_lemma_lookup(data):
    lemma_lookup = defaultdict(set)
    semantic_relations = defaultdict(lambda: defaultdict(set))

    for entry in data:
        synset_id = entry.get("synset_id", "")

        for lang_code, trans in entry.get("translations", {}).items():
            lemma_lookup[lang_code].add(trans["lemma"])

        for rel_type in ["hypernyms", "hyponyms", "meronyms", "holonyms", "cohyponyms"]:
            for rel_entry in entry.get(rel_type, []):
                for lang_code, trans in rel_entry.get("translations", {}).items():
                    lemma_lookup[lang_code].add(trans["lemma"])
                    semantic_relations[lang_code][rel_type].add(trans["lemma"])

    return lemma_lookup, semantic_relations


def pick_language_pair():
    levels = list(LANGUAGE_CONFIG.keys())
    from_level = random.choice(levels)
    to_level = random.choice(levels)

    from_lang = random.choice(list(LANGUAGE_CONFIG[from_level].values()))
    to_lang = random.choice(list(LANGUAGE_CONFIG[to_level].values()))

    return from_lang, to_lang, f"{from_level}_to_{to_level}"


def get_lang_name(lang_code):
    for level, langs in LANGUAGE_CONFIG.items():
        for lang, v in langs.items():
            if v["code"] == lang_code:
                return v["name"]
    return None


# ---------------------------
# Analogy Generation
# ---------------------------

lemma_lookup, semantic_relations = build_lemma_lookup(data)
generation_time = datetime.utcnow().isoformat() + "Z"

def generate_distractors(correct_lemma, all_candidates, semantic_relations,
                         target_lang, relation_type, n_choices=4, difficulty=3):

    distractors = set()

    if difficulty == 1:
        distractors = set(random.sample(list(all_candidates), min(n_choices - 1, len(all_candidates))))
        distractor_type = "random_unrelated"

    elif difficulty == 2:
        random_words = set(random.sample(list(all_candidates), min(n_choices - 2, len(all_candidates))))
        sem_words = semantic_relations[target_lang].get("cohyponyms", set()) if target_lang in semantic_relations else set()
        if sem_words:
            sem_sample = set(random.sample(list(sem_words), min(1, len(sem_words))))
        else:
            sem_sample = set()
        distractors = random_words.union(sem_sample)
        distractor_type = "mixed_random_semantic"

    elif difficulty >= 3:
        sem_pool = set()
        if target_lang in semantic_relations:
            sem_pool.update(semantic_relations[target_lang].get(relation_type, set()))
            sem_pool.update(semantic_relations[target_lang].get("cohyponyms", set()))
        if len(sem_pool) >= n_choices - 1:
            distractors = set(random.sample(list(sem_pool), n_choices - 1))
        else:
            distractors = sem_pool.union(
                set(random.sample(list(all_candidates), min(n_choices - 1 - len(sem_pool), len(all_candidates))))
            )
        distractor_type = "semantically_related"

    distractors.discard(correct_lemma)
    while len(distractors) < (n_choices - 1):
        remaining = all_candidates - distractors - {correct_lemma}
        if remaining:
            distractors.add(random.choice(list(remaining)))
        else:
            break

    return list(distractors), distractor_type


def generate_analogies(output_filename):

    analogies = []
    qid = 0

    while len(analogies) < NUM_ANALOGY_QUESTIONS:
        entry = random.choice(data)

        # Pick a semantic relation to use
        candidate_relations = [rel for rel in ["hypernyms", "hyponyms", "meronyms", "holonyms", "cohyponyms"]
                               if entry.get(rel)]
        if not candidate_relations:
            continue

        relation_type = random.choice(candidate_relations)
        rel_entries = entry[relation_type]

        # Pick (A, B) = (entry, related entry)
        relation_entry = random.choice(rel_entries)

        # Pick source language
        from_lang, to_lang, resource_pair = pick_language_pair()
        from_code = from_lang["code"]
        to_code = to_lang["code"]

        if from_code not in entry["translations"]:
            continue
        if from_code not in relation_entry.get("translations", {}):
            continue

        A_lemma = entry["translations"][from_code]["lemma"]
        B_lemma = relation_entry["translations"][from_code]["lemma"]

        # Find a second analogy pair (C, D) in the same relation
        candidates_for_second_pair = [
            e for e in data if e != entry and e.get(relation_type)
        ]
        if not candidates_for_second_pair:
            continue

        second_entry = random.choice(candidates_for_second_pair)
        second_rel_entries = second_entry[relation_type]
        second_relation_entry = random.choice(second_rel_entries)

        if to_code not in second_entry.get("translations", {}):
            continue
        if to_code not in second_relation_entry.get("translations", {}):
            continue

        C_lemma = second_entry["translations"][to_code]["lemma"]
        D_correct_lemma = second_relation_entry["translations"][to_code]["lemma"]

        # Generate distractors
        all_candidates = lemma_lookup[to_code] - {D_correct_lemma}
        if len(all_candidates) < 3:
            continue

        difficulty_level = random.randint(1, 5)
        distractors, distractor_type = generate_distractors(
            D_correct_lemma,
            all_candidates,
            semantic_relations,
            to_code,
            relation_type,
            difficulty=difficulty_level
        )

        options = distractors + [D_correct_lemma]
        random.shuffle(options)
        answer_index = options.index(D_correct_lemma)

        # Build prompt text
        prompt_text = (
            f"Complete the analogy:\n\n"
            f"{A_lemma} ({get_lang_name(from_code)}) is to {B_lemma} ({get_lang_name(from_code)})\n"
            f"as\n"
            f"{C_lemma} ({get_lang_name(to_code)}) is to ____?\n\n"
            f"Choose the correct option in {get_lang_name(to_code)}:"
        )

        question = {
            "id": f"analogy_{qid}_{from_code}_to_{to_code}",
            "prompt": prompt_text,
            "options": options,
            "answer_index": answer_index,
            "metadata": {
                "resource_pair": resource_pair,
                "relation_type": relation_type,
                "from_lang": from_code,
                "to_lang": to_code,
                "difficulty": difficulty_level,
                "distractor_type": distractor_type,
                "generation_time": generation_time
            }
        }
        analogies.append(question)
        qid += 1

    # Save
    with open(output_filename, "w", encoding="utf-8") as f_out:
        json.dump(analogies, f_out, indent=2, ensure_ascii=False)

    print(f"Generated {len(analogies)} semantic analogy questions. Saved to {output_filename}")


if __name__ == "__main__":
    generate_analogies("semantic_analogy_questions.json")
