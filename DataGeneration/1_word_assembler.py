import babelnet as bn
from babelnet import BabelSynsetID, Language
from babelnet.data.relation import BabelPointer
from tqdm import tqdm
from collections import deque # More efficient for queue operations

# ---------------------------------------------------
# Helper functions
# ---------------------------------------------------

# Cache for synset objects and their lemmas to avoid redundant lookups
_synset_cache = {}
_lemma_cache = {}

def get_cached_synset(synset_id_obj):
    """Retrieves a synset from cache or BabelNet, then caches it."""
    synset_id_str = synset_id_obj.id # Use the string representation for dict key
    if synset_id_str not in _synset_cache:
        try:
            _synset_cache[synset_id_str] = bn.get_synset(synset_id_obj)
        except Exception as e:
            print(f"[!] Could not retrieve synset {synset_id_str}: {type(e).__name__}: {e}")
            return None
    return _synset_cache[synset_id_str]

def get_lemma(synset):
    """Retrieves lemma from cache or synset, then caches it."""
    if synset is None:
        return "N/A"
    synset_id_str = synset.id.id
    if synset_id_str not in _lemma_cache:
        main_sense = synset.main_sense(Language.EN)
        _lemma_cache[synset_id_str] = main_sense.full_lemma if main_sense else "N/A"
    return _lemma_cache[synset_id_str]

def fetch_edges(synset, pointer, relation_type, max_items=50):
    """
    Fetch outgoing edges of a given pointer type.
    """
    items = []
    if synset is None:
        return items

    try:
        edges = synset.outgoing_edges(pointer)
        for edge in edges:
            target_synset = get_cached_synset(edge.id_target)
            lemma = get_lemma(target_synset)
            if lemma != "N/A":
                items.append({
                    "id": edge.id_target.id,
                    "lemma": lemma,
                    "relation": relation_type
                })
            if len(items) >= max_items:
                break
    except Exception as e:
        print(f"[!] Error fetching {relation_type} for {synset.id}: {type(e).__name__}: {e}")
    return items

def fetch_hypernyms(synset, max_items=50):
    return fetch_edges(synset, pointer=BabelPointer.ANY_HYPERNYM, relation_type="hypernym", max_items=max_items)

def fetch_hyponyms(synset, max_items=50):
    return fetch_edges(synset, pointer=BabelPointer.ANY_HYPONYM, relation_type="hyponym", max_items=max_items)

def fetch_meronyms(synset, max_items=50):
    meronym_pointers = [
        BabelPointer.PART_MERONYM,
        BabelPointer.MEMBER_MERONYM,
        BabelPointer.SUBSTANCE_MERONYM
    ]
    items = []
    if synset is None:
        return items

    try:
        for pointer in meronym_pointers:
            edges = fetch_edges(synset, pointer=pointer, relation_type="meronym", max_items=max_items)
            items.extend(edges)
            if len(items) >= max_items:
                break
    except Exception as e:
        print(f"[!] Error fetching meronyms for {synset.id}: {type(e).__name__}: {e}")
    return items[:max_items]

def get_cohyponyms(synset, max_items=50):
    """
    Get cohyponyms:
    - for each hypernym of this synset
        - find all hyponyms of that hypernym
        - exclude this synset itself
    """
    cohyponyms = []
    if synset is None:
        return cohyponyms

    try:
        hypernym_edges = synset.outgoing_edges(BabelPointer.ANY_HYPERNYM)
        for hypernym_edge in hypernym_edges:
            hypernym_synset = get_cached_synset(hypernym_edge.id_target)
            if hypernym_synset is None:
                continue

            hyponym_edges = hypernym_synset.outgoing_edges(BabelPointer.ANY_HYPONYM)
            for hyponym_edge in hyponym_edges:
                if hyponym_edge.id_target.id != synset.id.id:
                    target_synset = get_cached_synset(hyponym_edge.id_target)
                    lemma = get_lemma(target_synset)
                    if lemma != "N/A":
                        cohyponyms.append({
                            "id": hyponym_edge.id_target.id,
                            "lemma": lemma,
                            "relation": "cohyponym"
                        })
                if len(cohyponyms) >= max_items:
                    break
            if len(cohyponyms) >= max_items:
                break
    except Exception as e:
        print(f"[!] Error fetching cohyponyms for {synset.id}: {type(e).__name__}: {e}")
    return cohyponyms

# ---------------------------------------------------
# Recursive traversal
# ---------------------------------------------------

def traverse_synset(synset_id, max_depth, visited, max_items):
    """
    Recursively traverse all relations of a synset up to max_depth.
    """
    # Using deque for efficient append and pop from both ends (BFS behavior)
    queue = deque([(synset_id, 0)])

    pbar = tqdm(desc=f"⤵ Traversing from root {synset_id[:8]}...", unit="synset", position=1, leave=False)

    while queue:
        current_id, current_depth = queue.popleft() # Use popleft for BFS

        if current_id in visited:
            pbar.update(1) # Still update progress for skipped items
            continue

        synset = get_cached_synset(BabelSynsetID(current_id))
        if synset is None:
            pbar.update(1)
            continue

        lemma = get_lemma(synset)
        visited[current_id] = lemma
        print(f"[+] Discovered synset {current_id} → {lemma}")

        if current_depth >= max_depth:
            pbar.update(1)
            continue

        all_relations = []
        all_relations.extend(fetch_hypernyms(synset, max_items=max_items))
        all_relations.extend(fetch_hyponyms(synset, max_items=max_items))
        all_relations.extend(fetch_meronyms(synset, max_items=max_items))
        all_relations.extend(get_cohyponyms(synset, max_items=max_items))

        for relation in all_relations:
            rel_id = relation["id"]
            rel_lemma = relation["lemma"]
            rel_type = relation["relation"]

            # print newly discovered word
            print(f"    [→] {rel_type.upper()}: {rel_id} → {rel_lemma}")

            if rel_id not in visited:
                queue.append((rel_id, current_depth + 1))

        pbar.update(1)

    pbar.close()

# ---------------------------------------------------
# File processing
# ---------------------------------------------------

def process_file(input_file, output_file, max_depth=4, max_items=50):
    global _synset_cache, _lemma_cache
    visited_synsets = dict()

    with open(input_file, "r", encoding="utf-8") as infile:
        synset_ids = [line.strip().split("\t")[0] for line in infile if line.strip()]

    with tqdm(total=len(synset_ids), desc="🔍 Processing root synsets", unit="root") as main_bar:
        for synset_id in synset_ids:
            # Clear caches for each root synset if memory is an issue,
            # otherwise keep them to benefit from inter-root overlaps.
            # For deeper traversals and many root synsets, clearing might be necessary.
            # For now, let's keep it to maximize caching benefits.
            # _synset_cache = {}
            # _lemma_cache = {}

            traverse_synset(synset_id, max_depth, visited_synsets, max_items=max_items)
            main_bar.update(1)

    with open(output_file, "w", encoding="utf-8") as outfile:
        for synset_id, lemma in visited_synsets.items():
            outfile.write(f"{synset_id}\t{lemma}\n")

    print(f"[✓] Written {len(visited_synsets)} unique synsets to {output_file}")

# ---------------------------------------------------
# Main
# ---------------------------------------------------

if __name__ == "__main__":
    input_path = "../GeneratedFiles/seed_words_10.txt"
    output_path = "../GeneratedFiles/assembled_words.txt"
    max_depth = 5
    max_items = 6   # ← adjust this as desired!

    process_file(input_path, output_path, max_depth=max_depth, max_items=max_items)