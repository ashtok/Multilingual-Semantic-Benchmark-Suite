import babelnet as bn
from babelnet import BabelSynsetID, Language
from babelnet.data.relation import BabelPointer

def fetch_hypernyms(synset, max_items=10):
    """Fetch hypernyms (broader categories) of the synset."""
    return fetch_edges(
        synset,
        pointer=BabelPointer.ANY_HYPERNYM,
        relation_type="hypernym",
        max_items=max_items
    )


def fetch_hyponyms(synset, max_items=10):
    """Fetch hyponyms (narrower terms) of the synset."""
    return fetch_edges(
        synset,
        pointer=BabelPointer.ANY_HYPONYM,
        relation_type="hyponym",
        max_items=max_items
    )


def fetch_meronyms(synset, max_items=10):
    """Fetch meronyms (part-of relationships) of the synset from all meronym pointers."""
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
    """Fetch co-hyponyms (siblings) of the synset."""
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
                            "lemma": lemma
                        })
                if len(cohyponyms) >= max_items:
                    break
            if len(cohyponyms) >= max_items:
                break
    except Exception as e:
        print(f"[!] Error fetching co-hyponyms for {synset.id}: {type(e).__name__}: {e}")
    return cohyponyms


def fetch_edges(synset, pointer, relation_type, max_items=10):
    """Generic edge fetcher for a synset given a pointer and relation type."""
    items = []
    try:
        edges = synset.outgoing_edges(pointer)
        for edge in edges:
            if len(items) >= max_items:
                break
            target_synset = bn.get_synset(edge.id_target)
            lemma = get_lemma(target_synset)
            if lemma != "N/A":
                items.append({
                    "id": edge.id_target.id,
                    "lemma": lemma
                })
    except Exception as e:
        print(f"[!] Error fetching {relation_type} for {synset.id}: {type(e).__name__}: {e}")
    return items


def get_lemma(synset):
    """Get the English lemma for a synset, or 'N/A' if not found."""
    main_sense = synset.main_sense(Language.EN)
    if main_sense:
        return main_sense.full_lemma
    return "N/A"


def deduplicate(items):
    """Remove duplicates from a list of dicts by lemma."""
    seen = set()
    deduped = []
    for item in items:
        if item["lemma"] not in seen:
            deduped.append(item)
            seen.add(item["lemma"])
    return deduped


def has_all_relations(synset, max_items=1):
    """
    Return True if synset has at least:
        - one hypernym
        - one hyponym
        - one meronym
        - one co-hyponym
    Otherwise return False.
    """
    try:
        has_hypernyms = bool(fetch_hypernyms(synset, max_items))
        has_hyponyms = bool(fetch_hyponyms(synset, max_items))
        has_meronyms = bool(fetch_meronyms(synset, max_items))
        has_cohyponyms = bool(get_cohyponyms(synset, max_items))

        return all([has_hypernyms, has_hyponyms, has_meronyms, has_cohyponyms])

    except Exception as e:
        print(f"[!] Error checking relations for {synset.id}: {type(e).__name__}: {e}")
        return False


def process_file(input_file, output_file, max_lines=100):
    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:
        for i, line in enumerate(infile):
            if i >= max_lines:
                break
            line = line.strip()
            if not line:
                print(f"[Line {i+1}] Skipped empty line.")
                continue
            synset_id = line.split("\t")[0]
            print(f"[Line {i+1}] Processing synset ID: {synset_id}")
            try:
                synset = bn.get_synset(BabelSynsetID(synset_id))
                if has_all_relations(synset):
                    print(f"[Line {i+1}] Synset {synset_id} has all required relations. Writing to output.")
                    outfile.write(synset_id + "\n")
                else:
                    print(f"[Line {i+1}] Synset {synset_id} missing one or more required relations.")
            except Exception as e:
                print(f"[Line {i+1}] Failed to process {synset_id}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    input_path = "../GeneratedFiles/assembled_words.txt"  # Change to your input filename
    output_path = "../GeneratedFiles/babelnet_with_relations.txt"
    process_file(input_path, output_path, max_lines=6499)
    print(f"Processed up to 6499 lines. Output written to {output_path}")
