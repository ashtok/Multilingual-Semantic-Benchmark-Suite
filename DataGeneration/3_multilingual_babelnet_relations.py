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
NUM_SYNSETS = 1000

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
    """Extract *English only* glossary and examples, preferring wn > wn2020 > others."""
    glossary_data = {}
    examples_data = {}

    # Preference order
    SOURCE_PRIORITY = ['wn', 'wn2020']

    try:
        glosses = [g for g in synset.glosses() if g.language == Language.EN]
        if glosses:
            # Group glosses by source
            gloss_by_source = {str(getattr(g, 'source', 'Unknown')).lower(): g for g in glosses}

            # Try preferred sources first
            chosen_gloss = None
            for preferred in SOURCE_PRIORITY:
                if preferred in gloss_by_source:
                    chosen_gloss = gloss_by_source[preferred]
                    break
            # If no preferred, pick the first available gloss
            if not chosen_gloss and gloss_by_source:
                chosen_gloss = next(iter(gloss_by_source.values()))

            if chosen_gloss:
                source_str = str(getattr(chosen_gloss, 'source', 'Unknown'))
                print(f"Glossary: {chosen_gloss.gloss} | Source: {source_str}")
                glossary_data['en'] = {
                    'text': chosen_gloss.gloss,
                    'language': chosen_gloss.language.name,
                    'source': source_str
                }

    except Exception as e:
        print(f"[DEBUG] Error fetching glosses: {e}")

    try:
        examples = [ex for ex in synset.examples() if ex.language == Language.EN]
        if examples:
            # Group examples by source
            examples_by_source = {}
            for ex in examples:
                src = str(getattr(ex, 'source', 'Unknown')).lower()
                examples_by_source.setdefault(src, []).append(ex)

            chosen_examples = None
            for preferred in SOURCE_PRIORITY:
                if preferred in examples_by_source:
                    chosen_examples = examples_by_source[preferred]
                    break
            if not chosen_examples and examples_by_source:
                chosen_examples = next(iter(examples_by_source.values()))

            if chosen_examples:
                for ex in chosen_examples:
                    source_str = str(getattr(ex, 'source', 'Unknown'))
                    print(f"Example: {ex.example} | Source: {source_str}")
                    examples_data.setdefault('en', []).append({
                        'text': ex.example,
                        'language': ex.language.name,
                        'source': source_str
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