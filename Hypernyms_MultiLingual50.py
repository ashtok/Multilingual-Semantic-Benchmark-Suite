import json
import random
from pathlib import Path
import babelnet as bn
from language_config import LANGUAGE_CONFIG
from babelnet import BabelSynsetID, Language
from fetch_relatives import (
    fetch_hypernyms,
    fetch_hyponyms,
    fetch_meronyms,
    get_cohyponyms,
    deduplicate,
    get_lemma
)

# Path to your file with 7000 BabelNet IDs
BABELNET_IDS_FILE = "babelnet_with_relations.txt"

# How many synsets to sample
NUM_SYNSETS = 5

# Output file
OUTPUT_JSON = "babelnet_relations_dataset.json"

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

    synset_data = {
        "synset_id": synset_id_str,
        "lemma_en": lemma,
        "translations": translations,
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
    sampled_ids = random.sample(all_ids, min(NUM_SYNSETS, len(all_ids)))

    dataset = []
    for i, synset_id in enumerate(sampled_ids, start=1):
        print(f"[{i}/{len(sampled_ids)}] Processing synset {synset_id}")
        data = fetch_synset_relations(synset_id, max_items=10)
        if data:
            dataset.append(data)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done. Dataset saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
