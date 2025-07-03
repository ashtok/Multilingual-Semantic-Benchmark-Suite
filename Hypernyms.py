import babelnet as bn
from babelnet import BabelSynsetID, Language
from babelnet.data.relation import BabelPointer
from collections import Counter
import random
import json


def fetch_hypernyms(synset, max_hypernyms=5):
    """Fetch hypernyms of a given synset, limited by max_hypernyms."""
    hypernyms = []
    try:
        edges = synset.outgoing_edges(BabelPointer.ANY_HYPERNYM)
        for edge in edges[:max_hypernyms]:
            target_synset = bn.get_synset(edge.id_target)
            lemma = target_synset.main_sense(Language.EN).full_lemma if target_synset.main_sense(Language.EN) else "N/A"
            hypernyms.append({
                "pointer": edge.pointer.name,
                "id": edge.id_target.id,
                "lemma": lemma,
                "synset": target_synset
            })
    except Exception as e:
        print(f"[!] Error fetching hypernyms for {synset.id}: {e}")
    return hypernyms


def fetch_hyponyms(synset, max_hyponyms=10):
    """Fetch hyponyms (more specific terms) of a given synset."""
    hyponyms = []
    try:
        edges = synset.outgoing_edges(BabelPointer.ANY_HYPONYM)
        for edge in edges[:max_hyponyms]:
            target_synset = bn.get_synset(edge.id_target)
            lemma = target_synset.main_sense(Language.EN).full_lemma if target_synset.main_sense(Language.EN) else "N/A"
            if lemma != "N/A":
                hyponyms.append({
                    "id": edge.id_target.id,
                    "lemma": lemma,
                    "synset": target_synset
                })
    except Exception as e:
        print(f"[!] Error fetching hyponyms for {synset.id}: {e}")
    return hyponyms


def get_cohyponyms(synset, max_cohyponyms=10):
    """Get co-hyponyms (siblings in the hierarchy) by finding hypernyms and their hyponyms."""
    cohyponyms = []
    try:
        # Get hypernyms first
        hypernym_edges = synset.outgoing_edges(BabelPointer.ANY_HYPERNYM)
        for hypernym_edge in hypernym_edges[:3]:  # Limit to avoid too many API calls
            hypernym_synset = bn.get_synset(hypernym_edge.id_target)

            # Get hyponyms of the hypernym (these are co-hyponyms)
            hyponym_edges = hypernym_synset.outgoing_edges(BabelPointer.ANY_HYPONYM)
            for hyponym_edge in hyponym_edges[:max_cohyponyms]:
                if hyponym_edge.id_target.id != synset.id.id:  # Exclude original synset
                    target_synset = bn.get_synset(hyponym_edge.id_target)
                    lemma = target_synset.main_sense(Language.EN).full_lemma if target_synset.main_sense(
                        Language.EN) else "N/A"
                    if lemma != "N/A":
                        cohyponyms.append({
                            "id": hyponym_edge.id_target.id,
                            "lemma": lemma,
                            "synset": target_synset
                        })
    except Exception as e:
        print(f"[!] Error fetching co-hyponyms for {synset.id}: {e}")

    return cohyponyms


def get_spanish_translation(synset):
    """Get Spanish translation of a synset if available."""
    try:
        spanish_sense = synset.main_sense(Language.ES)
        if spanish_sense:
            return spanish_sense.full_lemma
    except Exception as e:
        print(f"[DEBUG] No Spanish translation available: {e}")
    return None


def generate_hypernym_question(root_synset_id, question_id=1, include_spanish=True):
    """
    Generate a hypernym validation question using BabelNet Python library.

    Args:
        root_synset_id: BabelNet synset ID string
        question_id: Question number
        include_spanish: Whether to try including Spanish translations

    Returns:
        Dictionary with question data or None if generation fails.
    """
    try:
        root_synset = bn.get_synset(BabelSynsetID(root_synset_id))

        # Get English lemma (same as your working code)
        root_lemma_en = None
        if root_synset.main_sense(Language.EN):
            root_lemma_en = root_synset.main_sense(Language.EN).full_lemma

        if not root_lemma_en:
            print(f"[!] Could not get English lemma for {root_synset_id}")
            return None

        print(f"[DEBUG] Root concept: {root_lemma_en} ({root_synset_id})")

        # Get Spanish translation if requested
        root_lemma_es = None
        if include_spanish:
            root_lemma_es = get_spanish_translation(root_synset)
            if root_lemma_es:
                print(f"[DEBUG] Spanish translation: {root_lemma_es}")

        # Get hypernyms using your working method
        hypernyms = fetch_hypernyms(root_synset, max_hypernyms=5)
        if not hypernyms:
            print(f"[!] No hypernyms found for {root_synset_id}")
            return None

        print(f"[DEBUG] Found {len(hypernyms)} hypernyms: {[h['lemma'] for h in hypernyms if h['lemma'] != 'N/A']}")

        # Get distractors
        cohyponyms = get_cohyponyms(root_synset, max_cohyponyms=8)
        hyponyms = fetch_hyponyms(root_synset, max_hyponyms=5)

        print(f"[DEBUG] Found {len(cohyponyms)} co-hyponyms, {len(hyponyms)} hyponyms")

        # Filter out hypernyms with N/A lemmas
        valid_hypernyms = [h for h in hypernyms if h['lemma'] != 'N/A']
        if not valid_hypernyms:
            print(f"[!] No valid hypernyms found for {root_synset_id}")
            return None

        # Select correct hypernym
        correct_hypernym = random.choice(valid_hypernyms)
        print(f"[DEBUG] Selected hypernym: {correct_hypernym['lemma']}")

        # Get Spanish translation for correct hypernym if needed
        correct_answer_es = None
        if include_spanish:
            correct_answer_es = get_spanish_translation(correct_hypernym["synset"])

        # Create distractors pool
        distractor_pool = []

        # Add co-hyponyms as distractors
        for cohyp in cohyponyms:
            if cohyp["lemma"] != correct_hypernym["lemma"]:
                spanish_trans = None
                if include_spanish:
                    spanish_trans = get_spanish_translation(cohyp["synset"])

                distractor_pool.append({
                    "lemma_en": cohyp["lemma"],
                    "lemma_es": spanish_trans,
                    "type": "co-hyponym",
                    "synset": cohyp["synset"]
                })

        # Add hyponyms as distractors
        for hyp in hyponyms:
            if hyp["lemma"] != correct_hypernym["lemma"]:
                spanish_trans = None
                if include_spanish:
                    spanish_trans = get_spanish_translation(hyp["synset"])

                distractor_pool.append({
                    "lemma_en": hyp["lemma"],
                    "lemma_es": spanish_trans,
                    "type": "hyponym",
                    "synset": hyp["synset"]
                })

        # Add other hypernyms as distractors
        for hyper in valid_hypernyms:
            if hyper["lemma"] != correct_hypernym["lemma"]:
                spanish_trans = None
                if include_spanish:
                    spanish_trans = get_spanish_translation(hyper["synset"])

                distractor_pool.append({
                    "lemma_en": hyper["lemma"],
                    "lemma_es": spanish_trans,
                    "type": "alternative_hypernym",
                    "synset": hyper["synset"]
                })

        print(f"[DEBUG] Generated {len(distractor_pool)} potential distractors")

        if len(distractor_pool) < 2:
            print(f"[!] Not enough distractors for {root_synset_id}")
            return None

        # Select 3 distractors
        num_distractors = min(3, len(distractor_pool))
        selected_distractors = random.sample(distractor_pool, num_distractors)

        # Create options - English only version
        options = [
            {
                "label": "A",
                "text_en": correct_hypernym["lemma"],
                "text_es": correct_answer_es if correct_answer_es else correct_hypernym["lemma"],
                "is_correct": True,
                "type": "hypernym"
            }
        ]

        for i, distractor in enumerate(selected_distractors):
            options.append({
                "label": chr(66 + i),  # B, C, D
                "text_en": distractor["lemma_en"],
                "text_es": distractor["lemma_es"] if distractor["lemma_es"] else distractor["lemma_en"],
                "is_correct": False,
                "type": distractor["type"]
            })

        # Shuffle options and reassign labels
        random.shuffle(options)
        for i, option in enumerate(options):
            option["label"] = chr(65 + i)  # A, B, C, D
            if option["is_correct"]:
                correct_label = option["label"]

        # Create prompts
        if include_spanish and root_lemma_es:
            prompt_bilingual = f'Which of the following is a hypernym (broader category) of the Spanish word "{root_lemma_es}" ({root_lemma_en})?'
        else:
            prompt_bilingual = f'Which of the following is a hypernym (broader category) of the English word "{root_lemma_en}"?'

        question_data = {
            "question_id": question_id,
            "root_synset_id": root_synset_id,
            "root_lemma_en": root_lemma_en,
            "root_lemma_es": root_lemma_es,
            "prompt_en": f'Which of the following is a hypernym (broader category) of the English word "{root_lemma_en}"?',
            "prompt_es": f'¿Cuál de las siguientes es un hiperónimo (categoría más amplia) de la palabra española "{root_lemma_es}"?' if root_lemma_es else None,
            "prompt_bilingual": prompt_bilingual,
            "options": options,
            "correct_answer": correct_label,
            "explanation": f'"{options[ord(correct_label) - 65]["text_en"]}" is the hypernym; the others are either co-hyponyms, hyponyms, or unrelated terms.',
            "include_spanish": include_spanish and root_lemma_es is not None
        }

        return question_data

    except Exception as e:
        print(f"[!] Error generating question for {root_synset_id}: {e}")
        return None


def generate_questionnaire(synset_ids, output_file="hypernym_questionnaire.json", include_spanish=True):
    """
    Generate a complete questionnaire from a list of synset IDs.
    """
    questionnaire = {
        "title": "Semantic Hierarchy Validation - Hypernymy Task",
        "description": "Multiple-choice questions testing understanding of hypernym relationships",
        "instructions": "Select the option that represents a hypernym (broader category) of the given word.",
        "questions": []
    }

    successful_questions = 0

    for i, synset_id in enumerate(synset_ids):
        print(f"\nGenerating question {i + 1}/{len(synset_ids)} for {synset_id}...")

        question = generate_hypernym_question(synset_id, question_id=i + 1, include_spanish=include_spanish)
        if question:
            questionnaire["questions"].append(question)
            successful_questions += 1
        else:
            print(f"[!] Failed to generate question for {synset_id}")

    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(questionnaire, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Generated {successful_questions} questions successfully!")
    print(f"📄 Questionnaire saved to: {output_file}")

    return questionnaire["questions"]


def print_sample_question(question_data):
    """Print a formatted sample question."""
    print(f"\n🔹 Question {question_data['question_id']}")
    print(f"📝 Prompt: {question_data['prompt_bilingual']}")
    print("\n📋 Options:")

    for option in question_data['options']:
        marker = "✅" if option['is_correct'] else "  "
        if question_data.get('include_spanish', False):
            print(f"   {marker} {option['label']}. {option['text_es']} ({option['text_en']})")
        else:
            print(f"   {marker} {option['label']}. {option['text_en']}")

    print(f"\n💡 Explanation: {question_data['explanation']}")
    print(f"🎯 Correct Answer: {question_data['correct_answer']}")
    print("-" * 60)


if __name__ == "__main__":
    # Sample synset IDs from your working code
    sample_synsets = [
        'bn:00015267n',  # Dog
        'bn:00015258n',  # Canine
        'bn:00053079n',  # Mammal
        'bn:00016143n',  # Carnivore
        'bn:00028151n',  # Domestic animal
    ]

    print("🚀 Generating Hypernym Questionnaire using BabelNet Python Library...")
    print("=" * 60)

    # Generate questionnaire (try with Spanish first, fallback to English-only)
    questions = generate_questionnaire(
        sample_synsets,
        "hypernym_questionnaire.json",
        include_spanish=True
    )

    # Print sample questions
    if questions:
        print("\n📖 Sample Question:")
        print_sample_question(questions[0])

        print(f"\n📊 Summary:")
        spanish_questions = sum(1 for q in questions if q.get('include_spanish', False))
        print(f"   • Total questions generated: {len(questions)}")
        print(f"   • Questions with Spanish translations: {spanish_questions}")
        print(f"   • Questions English-only: {len(questions) - spanish_questions}")
        print(f"   • Output file: hypernym_questionnaire.json")
        print(f"   • Ready for LLM evaluation! 🎯")

#
# ✅ Generated 5 questions successfully!
# 📄 Questionnaire saved to: hypernym_questionnaire.json
#
# 📖 Sample Question:
#
# 🔹 Question 1
# 📝 Prompt: Which of the following is a hypernym (broader category) of the English word "dog"?
#
# 📋 Options:
#       A. stray
#    ✅ B. pet
#       C. domestic_animal
#       D. bitch
#
# 💡 Explanation: "pet" is the hypernym; the others are either co-hyponyms, hyponyms, or unrelated terms.
# 🎯 Correct Answer: B
# ------------------------------------------------------------
#
# 📊 Summary:
#    • Total questions generated: 5
#    • Questions with Spanish translations: 0
#    • Questions English-only: 5
#    • Output file: hypernym_questionnaire.json
#    • Ready for LLM evaluation! 🎯