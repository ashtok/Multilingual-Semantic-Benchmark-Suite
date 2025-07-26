import babelnet as bn
from babelnet import BabelSynsetID, Language
from babelnet.data.relation import BabelPointer
from tqdm import tqdm

# ---------------------------------------------------
# Helper functions
# ---------------------------------------------------

def fetch_edges(synset, pointer, relation_type, max_items=50):
    """
    Fetch outgoing edges of a given pointer type.
    """
    items = []
    try:
        edges = synset.outgoing_edges(pointer)
        for edge in edges:
            target_synset = bn.get_synset(edge.id_target)
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
    try:
        hypernym_edges = synset.outgoing_edges(BabelPointer.ANY_HYPERNYM)
        for hypernym_edge in hypernym_edges:
            hypernym_synset = bn.get_synset(hypernym_edge.id_target)
            hyponym_edges = hypernym_synset.outgoing_edges(BabelPointer.ANY_HYPONYM)
            for hyponym_edge in hyponym_edges:
                if hyponym_edge.id_target.id != synset.id.id:
                    target_synset = bn.get_synset(hyponym_edge.id_target)
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

def get_lemma(synset):
    main_sense = synset.main_sense(Language.EN)
    if main_sense:
        return main_sense.full_lemma
    return "N/A"

# ---------------------------------------------------
# Recursive traversal
# ---------------------------------------------------

def traverse_synset(synset_id, max_depth, visited, max_items):
    """
    Recursively traverse all relations of a synset up to max_depth.
    """
    queue = [(synset_id, 0)]

    pbar = tqdm(desc=f"⤵ Traversing from root {synset_id[:8]}...", unit="synset", position=1, leave=False)

    while queue:
        current_id, current_depth = queue.pop(0)

        if current_id in visited:
            continue

        try:
            synset = bn.get_synset(BabelSynsetID(current_id))
        except Exception as e:
            print(f"[!] Could not retrieve synset {current_id}: {type(e).__name__}: {e}")
            continue

        lemma = get_lemma(synset)
        visited[current_id] = lemma
        print(f"[+] Discovered synset {current_id} → {lemma}")

        if current_depth >= max_depth:
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
    visited_synsets = dict()

    with open(input_file, "r", encoding="utf-8") as infile:
        synset_ids = [line.strip().split("\t")[0] for line in infile if line.strip()]

    with tqdm(total=len(synset_ids), desc="🔍 Processing root synsets", unit="root") as main_bar:
        for synset_id in synset_ids:
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
    max_items = 10   # ← adjust this as desired!

    process_file(input_path, output_path, max_depth=max_depth, max_items=max_items)
