import json
import random
from pathlib import Path
import babelnet as bn
from language_config import LANGUAGE_CONFIG
from babelnet import BabelSynsetID, Language
from fetch_relatives_helper import (
    fetch_hypernyms,
    fetch_hyponyms,
    fetch_meronyms,
    get_cohyponyms,
    deduplicate,
    get_lemma
)
import time
from tqdm import tqdm

# Path to your file with 500 BabelNet IDs
BABELNET_IDS_FILE = "../GeneratedFiles/babelnet_with_relations.txt"

# Process all synsets in the file (set to None to process all)
NUM_SYNSETS = None

# Output file
OUTPUT_JSON = "../GeneratedFiles/JsonFiles/multilingual_babelnet_relations.json"

# Gather all languages for multilingual translations
ALL_LANGUAGES = {
    **LANGUAGE_CONFIG['high_resource'],
    **LANGUAGE_CONFIG['medium_resource'],
    **LANGUAGE_CONFIG['low_resource']
}


def load_babelnet_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids


def get_glossary_and_examples(synset):
    """Extract *English only* glossary and examples from a synset."""
    glossary_data = {}
    examples_data = {}

    try:
        glosses = synset.glosses()
        if glosses:
            for gloss in glosses:
                if gloss.language == Language.EN:
                    glossary_data['en'] = {
                        'text': gloss.gloss,
                        'language': gloss.language.name,
                        'source': str(getattr(gloss, 'source', 'Unknown'))
                    }
    except Exception as e:
        print(f"[DEBUG] Error fetching glosses: {e}")

    try:
        examples = synset.examples()
        if examples:
            for example in examples:
                if example.language == Language.EN:
                    if 'en' not in examples_data:
                        examples_data['en'] = []

                    examples_data['en'].append({
                        'text': example.example,
                        'language': example.language.name,
                        'source': str(getattr(example, 'source', 'Unknown'))
                    })
    except Exception as e:
        print(f"[DEBUG] Error fetching examples: {e}")

    return glossary_data, examples_data


def fetch_synset_relations(synset_id_str, max_items=10):
    try:
        synset = bn.get_synset(BabelSynsetID(synset_id_str))
    except Exception as e:
        print(f"[!] Failed to retrieve synset {synset_id_str}: {type(e).__name__}: {e}")
        return None

    lemma = get_lemma(synset)

    # Raw relations
    hypernyms = deduplicate(fetch_hypernyms(synset, max_items))
    hyponyms = deduplicate(fetch_hyponyms(synset, max_items))
    meronyms = deduplicate(fetch_meronyms(synset, max_items))
    cohyponyms = deduplicate(get_cohyponyms(synset, max_items))

    # Enrich relations with translations
    hypernyms = enrich_with_translations(hypernyms)
    hyponyms = enrich_with_translations(hyponyms)
    meronyms = enrich_with_translations(meronyms)
    cohyponyms = enrich_with_translations(cohyponyms)

    # Main synset translations
    translations = get_multilingual_translations(synset, ALL_LANGUAGES)

    # Get glossary and examples
    glossary, examples = get_glossary_and_examples(synset)

    synset_data = {
        "synset_id": synset_id_str,
        "lemma_en": lemma,
        "translations": translations,
        "glossary": glossary,
        "examples": examples,
        "hypernyms": hypernyms,
        "hyponyms": hyponyms,
        "meronyms": meronyms,
        "cohyponyms": cohyponyms
    }

    return synset_data


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


def enrich_with_translations(items):
    """For each item (synset reference), fetch its translations."""
    enriched = []
    for item in items:
        try:
            synset = bn.get_synset(BabelSynsetID(item['id']))
            translations = get_multilingual_translations(synset, ALL_LANGUAGES)
            enriched.append({
                "id": item["id"],
                "lemma": item["lemma"],
                "translations": translations
            })
        except Exception as e:
            print(f"[!] Failed to fetch translations for {item['id']}: {type(e).__name__}: {e}")
    return enriched


def main():
    all_ids = load_babelnet_ids(BABELNET_IDS_FILE)

    # Process all synsets if NUM_SYNSETS is None, otherwise sample
    if NUM_SYNSETS is None:
        synsets_to_process = all_ids
        print(f"📝 Processing all {len(all_ids)} synsets from the file")
    else:
        synsets_to_process = random.sample(all_ids, min(NUM_SYNSETS, len(all_ids)))
        print(f"📝 Processing {len(synsets_to_process)} randomly sampled synsets")

    start_time = time.time()

    dataset = []
    for synset_id in tqdm(synsets_to_process, desc="Processing synsets", unit="synset"):
        data = fetch_synset_relations(synset_id, max_items=5)
        if data:
            dataset.append(data)

    elapsed = time.time() - start_time
    print(f"\n⏱️ Completed in {elapsed:.2f} seconds.")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done. Dataset saved to: {OUTPUT_JSON}")

    # Print some statistics
    glossary_count = sum(1 for item in dataset if item.get('glossary'))
    examples_count = sum(1 for item in dataset if item.get('examples'))
    both_count = sum(1 for item in dataset if item.get('glossary') and item.get('examples'))

    print(f"📊 Statistics:")
    print(f"   - Total synsets processed: {len(dataset)}")
    print(f"   - Synsets with glossary: {glossary_count}")
    print(f"   - Synsets with examples: {examples_count}")
    print(f"   - Synsets with both glossary and examples: {both_count}")


if __name__ == "__main__":
    main()