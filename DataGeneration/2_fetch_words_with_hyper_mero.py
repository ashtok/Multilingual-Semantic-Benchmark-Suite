import babelnet as bn
from babelnet import BabelSynsetID, Language
from babelnet.data.relation import BabelPointer
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

# === Caching layer ===

@lru_cache(maxsize=100_000)
def cached_get_synset(synset_id: str):
    return bn.get_synset(BabelSynsetID(synset_id))

@lru_cache(maxsize=100_000)
def get_lemma_cached(synset_id: str):
    try:
        synset = cached_get_synset(synset_id)
        main_sense = synset.main_sense(Language.EN)
        return main_sense.full_lemma if main_sense else "N/A"
    except Exception:
        return "N/A"

# === Edge Fetchers ===

def fetch_edges(synset, pointer, relation_type, max_items=10):
    items = []
    try:
        edges = synset.outgoing_edges(pointer)
        for edge in edges:
            if len(items) >= max_items:
                break
            lemma = get_lemma_cached(edge.id_target.id)
            if lemma != "N/A":
                items.append({
                    "id": edge.id_target.id,
                    "lemma": lemma
                })
    except Exception as e:
        print(f"[!] Error fetching {relation_type} for {synset.id}: {type(e).__name__}: {e}")
    return items

def fetch_hypernyms(synset, max_items=10):
    return fetch_edges(synset, pointer=BabelPointer.ANY_HYPERNYM, relation_type="hypernym", max_items=max_items)

def fetch_hyponyms(synset, max_items=10):
    return fetch_edges(synset, pointer=BabelPointer.ANY_HYPONYM, relation_type="hyponym", max_items=max_items)

def fetch_meronyms(synset, max_items=10):
    meronym_pointers = [
        BabelPointer.PART_MERONYM,
        BabelPointer.MEMBER_MERONYM,
        BabelPointer.SUBSTANCE_MERONYM
    ]
    items = []
    try:
        for pointer in meronym_pointers:
            items.extend(fetch_edges(synset, pointer=pointer, relation_type="meronym"))
            if len(items) >= max_items:
                break
    except Exception as e:
        print(f"[!] Error fetching meronyms for {synset.id}: {type(e).__name__}: {e}")
    return items[:max_items]

def get_cohyponyms(synset, max_items=10):
    cohyponyms = []
    try:
        for hypernym_edge in synset.outgoing_edges(BabelPointer.ANY_HYPERNYM):
            hypernym_synset = cached_get_synset(hypernym_edge.id_target.id)
            for hyponym_edge in hypernym_synset.outgoing_edges(BabelPointer.ANY_HYPONYM):
                if hyponym_edge.id_target.id != synset.id.id:
                    lemma = get_lemma_cached(hyponym_edge.id_target.id)
                    if lemma != "N/A":
                        cohyponyms.append({
                            "id": hyponym_edge.id_target.id,
                            "lemma": lemma
                        })
                if len(cohyponyms) >= max_items:
                    break
            if len(cohyponyms) >= max_items:
                break
    except Exception as e:
        print(f"[!] Error fetching co-hyponyms for {synset.id}: {type(e).__name__}: {e}")
    return cohyponyms

# === Main filter logic ===

def has_all_relations(synset, max_items=1):
    try:
        return all([
            bool(fetch_hypernyms(synset, max_items)),
            bool(fetch_hyponyms(synset, max_items)),
            bool(fetch_meronyms(synset, max_items)),
            bool(get_cohyponyms(synset, max_items))
        ])
    except Exception as e:
        print(f"[!] Error checking relations for {synset.id}: {type(e).__name__}: {e}")
        return False

# === Threaded line processor ===

def process_synset_line(line, line_number):
    synset_id = line.split("\t")[0]
    try:
        synset = cached_get_synset(synset_id)
        if has_all_relations(synset):
            print(f"[Line {line_number}] ✅ Synset {synset_id} has all required relations.")
            return synset_id
        else:
            print(f"[Line {line_number}] ❌ Synset {synset_id} missing one or more relations.")
    except Exception as e:
        print(f"[Line {line_number}] 💥 Error processing {synset_id}: {type(e).__name__}: {e}")
    return None

# === File processor ===

def process_file(input_file, output_file, max_lines=100):
    with open(input_file, "r", encoding="utf-8") as infile:
        lines = [line.strip() for i, line in enumerate(infile) if line.strip() and i < max_lines]

    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_synset_line, line, i + 1): i + 1 for i, line in enumerate(lines)}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    with open(output_file, "w", encoding="utf-8") as outfile:
        for sid in results:
            outfile.write(sid + "\n")

    print(f"\n✅ Processing complete. {len(results)} synsets with all relations written to {output_file}.")

# === Main entry point ===

if __name__ == "__main__":
    input_path = "../GeneratedFiles/assembled_words.txt"
    output_path = "../GeneratedFiles/babelnet_with_relations.txt"
    process_file(input_path, output_path, max_lines=24983)
