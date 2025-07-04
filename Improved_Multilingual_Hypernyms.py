import babelnet as bn
from babelnet import BabelSynsetID, Language
from babelnet.data.relation import BabelPointer
from collections import Counter, defaultdict
import random
import json

# Language configuration with high-resource and low-resource languages
LANGUAGE_CONFIG = {
    'high_resource': {
        Language.EN: {'name': 'English', 'code': 'en'},
        Language.ES: {'name': 'Spanish', 'code': 'es'},
        Language.FR: {'name': 'French', 'code': 'fr'},
        Language.DE: {'name': 'German', 'code': 'de'},
        Language.IT: {'name': 'Italian', 'code': 'it'},
        Language.PT: {'name': 'Portuguese', 'code': 'pt'},
        Language.RU: {'name': 'Russian', 'code': 'ru'}
    },
    'low_resource': {
        Language.SW: {'name': 'Swahili', 'code': 'sw'},
        Language.IS: {'name': 'Icelandic', 'code': 'is'},
        Language.MT: {'name': 'Maltese', 'code': 'mt'}
    }
}

# All languages combined
ALL_LANGUAGES = {**LANGUAGE_CONFIG['high_resource'], **LANGUAGE_CONFIG['low_resource']}

# Hypernym prompts in different languages
HYPERNYM_PROMPTS = {
    'en': 'Which of the following is a hypernym (broader category) of the word "{word}"?',
    'es': '¿Cuál de las siguientes es un hiperónimo (categoría más amplia) de la palabra "{word}"?',
    'fr': 'Lequel des suivants est un hyperonyme (catégorie plus large) du mot "{word}"?',
    'de': 'Welcher der folgenden Begriffe ist ein Hyperonym (übergeordnete Kategorie) des Wortes "{word}"?',
    'it': 'Quale dei seguenti è un iperonimo (categoria più ampia) della parola "{word}"?',
    'pt': 'Qual dos seguintes é um hiperônimo (categoria mais ampla) da palavra "{word}"?',
    'ru': 'Какой из следующих является гиперонимом (более широкой категорией) слова "{word}"?',
    'sw': 'Ni kipi kati ya vifuatavyo ni hypernym (jamii pana) ya neno "{word}"?',
    'is': 'Hvort af eftirfarandi er yfirheiti (víðari flokkur) orðsins "{word}"?',
    'mt': 'Liema mill-ġejjin huwa iperonmu (kategorija usa\') tal-kelma "{word}"?'
}


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
        hypernym_edges = synset.outgoing_edges(BabelPointer.ANY_HYPERNYM)
        for hypernym_edge in hypernym_edges[:3]:
            hypernym_synset = bn.get_synset(hypernym_edge.id_target)
            hyponym_edges = hypernym_synset.outgoing_edges(BabelPointer.ANY_HYPONYM)
            for hyponym_edge in hyponym_edges[:max_cohyponyms]:
                if hyponym_edge.id_target.id != synset.id.id:
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


def fetch_meronyms(synset, max_meronyms=10):
    """Fetch meronyms (part-of relationships) of a given synset."""
    meronyms = []
    try:
        edges = synset.outgoing_edges(BabelPointer.MERONYM)
        for edge in edges[:max_meronyms]:
            target_synset = bn.get_synset(edge.id_target)
            lemma = target_synset.main_sense(Language.EN).full_lemma if target_synset.main_sense(Language.EN) else "N/A"
            if lemma != "N/A":
                meronyms.append({
                    "id": edge.id_target.id,
                    "lemma": lemma,
                    "synset": target_synset
                })
    except Exception as e:
        print(f"[!] Error fetching meronyms for {synset.id}: {e}")
    return meronyms


def fetch_holonyms(synset, max_holonyms=10):
    """Fetch holonyms (whole-of relationships) of a given synset."""
    holonyms = []
    try:
        edges = synset.outgoing_edges(BabelPointer.HOLONYM)
        for edge in edges[:max_holonyms]:
            target_synset = bn.get_synset(edge.id_target)
            lemma = target_synset.main_sense(Language.EN).full_lemma if target_synset.main_sense(Language.EN) else "N/A"
            if lemma != "N/A":
                holonyms.append({
                    "id": edge.id_target.id,
                    "lemma": lemma,
                    "synset": target_synset
                })
    except Exception as e:
        print(f"[!] Error fetching holonyms for {synset.id}: {e}")
    return holonyms


def fetch_related_concepts(synset, max_related=10):
    """Fetch semantically related concepts (similar, also, other relations)."""
    related = []
    try:
        # Try different relation types
        relation_types = [BabelPointer.SIMILAR, BabelPointer.ALSO, BabelPointer.OTHER]

        for relation_type in relation_types:
            edges = synset.outgoing_edges(relation_type)
            for edge in edges[:max_related // len(relation_types)]:
                target_synset = bn.get_synset(edge.id_target)
                lemma = target_synset.main_sense(Language.EN).full_lemma if target_synset.main_sense(
                    Language.EN) else "N/A"
                if lemma != "N/A":
                    related.append({
                        "id": edge.id_target.id,
                        "lemma": lemma,
                        "synset": target_synset,
                        "relation_type": relation_type.name
                    })
    except Exception as e:
        print(f"[!] Error fetching related concepts for {synset.id}: {e}")
    return related


def get_cross_category_distractors(synset, target_categories=None):
    """Get distractors from different semantic categories."""
    if target_categories is None:
        # Common high-level categories that are often confused
        target_categories = [
            'bn:00031027n',  # entity
            'bn:00019128n',  # event
            'bn:00031264n',  # quality
            'bn:00031921n',  # relation
            'bn:00031576n',  # process
            'bn:00031563n',  # property
        ]

    cross_category_distractors = []

    for category_id in target_categories:
        try:
            category_synset = bn.get_synset(BabelSynsetID(category_id))
            # Get some hyponyms of these high-level categories
            hyponyms = fetch_hyponyms(category_synset, max_hyponyms=15)

            # Filter out concepts that are too similar to our target
            for hyponym in hyponyms:
                if hyponym["lemma"] != synset.main_sense(Language.EN).full_lemma:
                    _, translations = calculate_language_coverage(hyponym["synset"], list(ALL_LANGUAGES.keys()))
                    cross_category_distractors.append({
                        "lemma_en": hyponym["lemma"],
                        "translations": translations,
                        "type": "cross_category",
                        "synset": hyponym["synset"]
                    })
        except Exception as e:
            print(f"[DEBUG] Could not fetch from category {category_id}: {e}")
            continue

    return cross_category_distractors


def get_multilingual_translations(synset, target_languages=None):
    """Get translations for a synset in multiple languages."""
    if target_languages is None:
        target_languages = ALL_LANGUAGES.keys()

    translations = {}
    for lang in target_languages:
        try:
            sense = synset.main_sense(lang)
            if sense:
                translations[ALL_LANGUAGES[lang]['code']] = {
                    'lemma': sense.full_lemma,
                    'language_name': ALL_LANGUAGES[lang]['name']
                }
        except Exception as e:
            print(f"[DEBUG] No translation available for {lang}: {e}")

    return translations


def calculate_language_coverage(synset, target_languages=None):
    """Calculate how many target languages have translations for a synset."""
    translations = get_multilingual_translations(synset, target_languages)
    total_languages = len(target_languages) if target_languages else len(ALL_LANGUAGES)
    coverage = len(translations) / total_languages
    return coverage, translations


def generate_multilingual_hypernym_question(root_synset_id, question_id=1,
                                            target_languages=None, min_coverage=0.5):
    """
    Generate a multilingual hypernym validation question with improved distractor selection.

    New distractor strategy:
    1. Alternative hypernyms (sibling categories)
    2. Meronyms/Holonyms (part-whole relationships)
    3. Related concepts from different semantic domains
    4. Cross-category concepts (entities from different top-level categories)
    5. Co-hyponyms (only if not enough alternatives available)
    """
    if target_languages is None:
        target_languages = list(ALL_LANGUAGES.keys())

    try:
        root_synset = bn.get_synset(BabelSynsetID(root_synset_id))

        # Get English lemma as base
        root_lemma_en = None
        if root_synset.main_sense(Language.EN):
            root_lemma_en = root_synset.main_sense(Language.EN).full_lemma

        if not root_lemma_en:
            print(f"[!] Could not get English lemma for {root_synset_id}")
            return None

        # Calculate language coverage
        coverage, root_translations = calculate_language_coverage(root_synset, target_languages)

        if coverage < min_coverage:
            print(f"[!] Insufficient language coverage ({coverage:.2f}) for {root_synset_id}")
            return None

        print(f"[DEBUG] Root concept: {root_lemma_en} ({root_synset_id})")
        print(f"[DEBUG] Language coverage: {coverage:.2f} ({len(root_translations)}/{len(target_languages)} languages)")

        # Get hypernyms
        hypernyms = fetch_hypernyms(root_synset, max_hypernyms=5)
        if not hypernyms:
            print(f"[!] No hypernyms found for {root_synset_id}")
            return None

        # Filter valid hypernyms
        valid_hypernyms = [h for h in hypernyms if h['lemma'] != 'N/A']
        if not valid_hypernyms:
            print(f"[!] No valid hypernyms found for {root_synset_id}")
            return None

        # Select correct hypernym with best language coverage
        best_hypernym = None
        best_coverage = 0

        for hypernym in valid_hypernyms:
            hyp_coverage, hyp_translations = calculate_language_coverage(hypernym["synset"], target_languages)
            if hyp_coverage > best_coverage:
                best_coverage = hyp_coverage
                best_hypernym = {
                    **hypernym,
                    "translations": hyp_translations,
                    "coverage": hyp_coverage
                }

        if not best_hypernym or best_coverage < min_coverage:
            print(f"[!] No hypernym with sufficient coverage for {root_synset_id}")
            return None

        correct_hypernym = best_hypernym
        print(f"[DEBUG] Selected hypernym: {correct_hypernym['lemma']} (coverage: {correct_hypernym['coverage']:.2f})")

        # NEW DISTRACTOR GENERATION STRATEGY
        distractor_pool = []


        # Strategy 2: Meronyms and Holonyms
        print("[DEBUG] Collecting meronyms and holonyms...")
        meronyms = fetch_meronyms(root_synset, max_meronyms=5)
        holonyms = fetch_holonyms(root_synset, max_holonyms=5)

        for meronym in meronyms:
            if meronym["lemma"] != correct_hypernym["lemma"]:
                _, translations = calculate_language_coverage(meronym["synset"], target_languages)
                distractor_pool.append({
                    "lemma_en": meronym["lemma"],
                    "translations": translations,
                    "type": "meronym",
                    "synset": meronym["synset"],
                    "priority": 2  # Good semantic distance
                })

        for holonym in holonyms:
            if holonym["lemma"] != correct_hypernym["lemma"]:
                _, translations = calculate_language_coverage(holonym["synset"], target_languages)
                distractor_pool.append({
                    "lemma_en": holonym["lemma"],
                    "translations": translations,
                    "type": "holonym",
                    "synset": holonym["synset"],
                    "priority": 2
                })

        # Strategy 3: Related concepts from different domains
        print("[DEBUG] Collecting related concepts...")
        related_concepts = fetch_related_concepts(root_synset, max_related=8)
        for related in related_concepts:
            if related["lemma"] != correct_hypernym["lemma"]:
                _, translations = calculate_language_coverage(related["synset"], target_languages)
                distractor_pool.append({
                    "lemma_en": related["lemma"],
                    "translations": translations,
                    "type": f"related_{related['relation_type']}",
                    "synset": related["synset"],
                    "priority": 3  # Good semantic distance
                })

        # Strategy 4: Cross-category distractors
        print("[DEBUG] Collecting cross-category distractors...")
        cross_category = get_cross_category_distractors(root_synset)
        for cross in cross_category[:10]:  # Limit to avoid too many
            if cross["lemma_en"] != correct_hypernym["lemma"]:
                distractor_pool.append({
                    **cross,
                    "priority": 4  # Good for diversity
                })

        # Strategy 5: Co-hyponyms (only if we don't have enough alternatives)
        if len(distractor_pool) < 5:
            print("[DEBUG] Collecting co-hyponyms...")
            cohyponyms = get_cohyponyms(root_synset, max_cohyponyms=8)
            for cohyp in cohyponyms:
                if cohyp["lemma"] != correct_hypernym["lemma"]:
                    _, translations = calculate_language_coverage(cohyp["synset"], target_languages)
                    distractor_pool.append({
                        "lemma_en": cohyp["lemma"],
                        "translations": translations,
                        "type": "co_hyponym",
                        "synset": cohyp["synset"],
                        "priority": 5  # Lower priority - can be confusing
                    })

        if len(distractor_pool) < 2:
            print(f"[!] Not enough distractors for {root_synset_id}")
            return None

        # Smart distractor selection
        print(f"[DEBUG] Total distractor pool: {len(distractor_pool)}")
        for i, distractor in enumerate(distractor_pool, start=1):
            print(f"   [{i}] {distractor['lemma_en']} ({distractor['type']}) "
                  f"- Translations: {list(distractor['translations'].keys())}")

        # NEW: Randomly select 3 distractors from the pool
        if len(distractor_pool) < 3:
            print(f"[!] Not enough distractors for {root_synset_id}")
            return None

        selected_distractors = random.sample(distractor_pool, 3)

        print(f"[DEBUG] Selected distractor types: {[d['type'] for d in selected_distractors]}")

        # Create multilingual options
        options = [
            {
                "label": "A",
                "text_en": correct_hypernym["lemma"],
                "translations": correct_hypernym["translations"],
                "is_correct": True,
                "type": "hypernym"
            }
        ]

        for i, distractor in enumerate(selected_distractors):
            options.append({
                "label": chr(66 + i),  # B, C, D
                "text_en": distractor["lemma_en"],
                "translations": distractor["translations"],
                "is_correct": False,
                "type": distractor["type"]
            })

        # Shuffle options and reassign labels
        random.shuffle(options)
        for i, option in enumerate(options):
            option["label"] = chr(65 + i)
            if option["is_correct"]:
                correct_label = option["label"]

        # Create multilingual prompts
        prompts = {}
        for lang_code, translation in root_translations.items():
            if lang_code in HYPERNYM_PROMPTS:
                prompts[lang_code] = HYPERNYM_PROMPTS[lang_code].format(word=translation['lemma'])

        # Calculate final statistics
        language_stats = {
            'total_languages': len(target_languages),
            'root_coverage': len(root_translations),
            'correct_answer_coverage': len(correct_hypernym["translations"]),
            'avg_distractor_coverage': sum(len(d["translations"]) for d in selected_distractors) / len(
                selected_distractors) if selected_distractors else 0
        }

        # Enhanced explanation based on distractor types
        distractor_types = [d["type"] for d in selected_distractors]
        explanation_parts = [
            f'"{options[ord(correct_label) - 65]["text_en"]}" is the correct hypernym (broader category)']


        if any("meronym" in dt or "holonym" in dt for dt in distractor_types):
            explanation_parts.append("some options represent part-whole relationships")
        if any("related" in dt for dt in distractor_types):
            explanation_parts.append("some options are semantically related but not hierarchical")
        if "cross_category" in distractor_types:
            explanation_parts.append("some options are from different semantic domains")
        if "co_hyponym" in distractor_types:
            explanation_parts.append("some options are sibling concepts at the same level")

        explanation = f"{explanation_parts[0]}. The other options are distractors: {', '.join(explanation_parts[1:])}."

        question_data = {
            "question_id": question_id,
            "root_synset_id": root_synset_id,
            "root_lemma_en": root_lemma_en,
            "root_translations": root_translations,
            "prompts": prompts,
            "options": options,
            "correct_answer": correct_label,
            "explanation_en": explanation,
            "distractor_strategy": {
                "types_used": distractor_types,
                "total_pool_size": len(distractor_pool),
                "selection_strategy": "priority_and_diversity"
            },
            "language_coverage": coverage,
            "language_stats": language_stats,
            "target_languages": [ALL_LANGUAGES[lang]['code'] for lang in target_languages],
            "high_resource_coverage": sum(1 for lang in LANGUAGE_CONFIG['high_resource'].keys() if
                                          ALL_LANGUAGES[lang]['code'] in root_translations),
            "low_resource_coverage": sum(1 for lang in LANGUAGE_CONFIG['low_resource'].keys() if
                                         ALL_LANGUAGES[lang]['code'] in root_translations)
        }

        return question_data

    except Exception as e:
        print(f"[!] Error generating question for {root_synset_id}: {e}")
        return None


def generate_multilingual_questionnaire(synset_ids, output_file="multilingual_hypernym_questionnaire.json",
                                        target_languages=None, min_coverage=0.3):
    """
    Generate a multilingual questionnaire from a list of synset IDs.
    """
    if target_languages is None:
        target_languages = list(ALL_LANGUAGES.keys())

    questionnaire = {
        "title": "Multilingual Semantic Hierarchy Validation - Hypernymy Task",
        "description": "Multiple-choice questions testing understanding of hypernym relationships across 10 languages",
        "instructions": {
            "en": "Select the option that represents a hypernym (broader category) of the given word.",
            "es": "Selecciona la opción que representa un hiperónimo (categoría más amplia) de la palabra dada.",
            "fr": "Sélectionnez l'option qui représente un hyperonyme (catégorie plus large) du mot donné."
        },
        "languages": {
            "high_resource": {code: info['name'] for code, info in
                              {ALL_LANGUAGES[lang]['code']: ALL_LANGUAGES[lang]
                               for lang in LANGUAGE_CONFIG['high_resource'].keys()}.items()},
            "low_resource": {code: info['name'] for code, info in
                             {ALL_LANGUAGES[lang]['code']: ALL_LANGUAGES[lang]
                              for lang in LANGUAGE_CONFIG['low_resource'].keys()}.items()}
        },
        "metadata": {
            "total_target_languages": len(target_languages),
            "min_coverage_threshold": min_coverage,
            "generation_date": "2025-06-26"
        },
        "questions": []
    }

    successful_questions = 0
    coverage_stats = []

    for i, synset_id in enumerate(synset_ids):
        print(f"\nGenerating question {i + 1}/{len(synset_ids)} for {synset_id}...")

        question = generate_multilingual_hypernym_question(
            synset_id,
            question_id=i + 1,
            target_languages=target_languages,
            min_coverage=min_coverage
        )

        if question:
            questionnaire["questions"].append(question)
            successful_questions += 1
            coverage_stats.append(question["language_coverage"])
        else:
            print(f"[!] Failed to generate question for {synset_id}")

    # Add summary statistics
    if coverage_stats:
        questionnaire["metadata"]["coverage_statistics"] = {
            "average_coverage": sum(coverage_stats) / len(coverage_stats),
            "min_coverage": min(coverage_stats),
            "max_coverage": max(coverage_stats),
            "questions_generated": successful_questions
        }

    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(questionnaire, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Generated {successful_questions} multilingual questions successfully!")
    print(f"📄 Questionnaire saved to: {output_file}")

    return questionnaire["questions"]


def print_multilingual_sample_question(question_data):
    """Print a formatted multilingual sample question."""
    print(f"\n🔹 Question {question_data['question_id']}")
    print(f"🌍 Language Coverage: {question_data['language_coverage']:.2f}")
    print(
        f"📊 High-resource: {question_data['high_resource_coverage']}/7, Low-resource: {question_data['low_resource_coverage']}/3")

    # Show root word in available languages
    print(f"\n🎯 Root Word Translations:")
    for lang_code, translation in question_data['root_translations'].items():
        print(f"   {lang_code.upper()}: {translation['lemma']} ({translation['language_name']})")

    # Show prompts in different languages
    print(f"\n📝 Sample Prompts:")
    for lang_code, prompt in list(question_data['prompts'].items())[:3]:  # Show first 3
        print(f"   {lang_code.upper()}: {prompt}")

    print(f"\n📋 Options (English + available translations):")
    for option in question_data['options']:
        marker = "✅" if option['is_correct'] else "  "
        print(f"   {marker} {option['label']}. {option['text_en']} [{option['type']}]")

        # Show translations for this option
        if option['translations']:
            translations_str = ", ".join([f"{code}: {info['lemma']}"
                                          for code, info in list(option['translations'].items())[:3]])
            print(f"      └─ {translations_str}...")

    print(f"\n🔧 Distractor Strategy:")
    print(f"   Types used: {', '.join(question_data['distractor_strategy']['types_used'])}")
    print(f"   Pool size: {question_data['distractor_strategy']['total_pool_size']}")

    print(f"\n💡 Explanation: {question_data['explanation_en']}")
    print(f"🎯 Correct Answer: {question_data['correct_answer']}")
    print("-" * 80)


if __name__ == "__main__":
    # Enhanced sample synset IDs
    sample_synsets = [
        'bn:00015267n',  # Dog
        'bn:00015258n',  # Canine
        'bn:00053079n',  # Mammal
        'bn:00028151n',  # Domestic animal
        'bn:00034516n',  # final_judgment
    ]

    print("🚀 Generating Multilingual Hypernym Questionnaire (10 Languages)...")
    print("🌍 High-resource: English, Spanish, French, German, Italian, Portuguese, Russian")
    print("🌏 Low-resource: Swahili, Icelandic, Maltese")
    print("🔧 NEW: Improved distractor selection strategy")
    print("=" * 80)

    # Generate multilingual questionnaire
    questions = generate_multilingual_questionnaire(
        sample_synsets,
        "multilingual_hypernym_questionnaire.json",
        target_languages=list(ALL_LANGUAGES.keys()),
        min_coverage=0.3  # Require at least 30% language coverage
    )

    # Print sample question
    if questions:
        print("\n📖 Sample Multilingual Question:")
        print_multilingual_sample_question(questions[0])

        # Summary statistics
        total_questions = len(questions)
        avg_coverage = sum(q["language_coverage"] for q in questions) / total_questions if total_questions > 0 else 0
        high_res_avg = sum(
            q["high_resource_coverage"] for q in questions) / total_questions if total_questions > 0 else 0
        low_res_avg = sum(q["low_resource_coverage"] for q in questions) / total_questions if total_questions > 0 else 0

        print(f"\n📊 Multilingual Summary:")
        print(f"   • Total questions generated: {total_questions}")
        print(f"   • Average language coverage: {avg_coverage:.2f}")
        print(f"   • Average high-resource coverage: {high_res_avg:.1f}/7 languages")
        print(f"   • Average low-resource coverage: {low_res_avg:.1f}/3 languages")
        print(f"   • Output file: multilingual_hypernym_questionnaire.json")
        print(f"   • Ready for multilingual LLMs evaluation! 🌍🎯")