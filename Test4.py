import babelnet as bn
from babelnet import BabelSynsetID, Language
from babelnet.data.relation import BabelPointer

def fetch_hypernyms(synset, max_items=10):
    return fetch_edges(
        synset,
        pointer=BabelPointer.ANY_HYPERNYM,
        relation_type="hypernym",
        max_items=max_items
    )

def fetch_hyponyms(synset, max_items=10):
    return fetch_edges(
        synset,
        pointer=BabelPointer.ANY_HYPONYM,
        relation_type="hyponym",
        max_items=max_items
    )

def fetch_antonyms(synset, max_items=10):
    return fetch_edges(
        synset,
        pointer=BabelPointer.ANTONYM,
        relation_type="antonym",
        max_items=max_items
    )

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

def fetch_relations_of_synsets(synset_ids, max_items=10):
    results = []
    for synset_id in synset_ids:
        try:
            synset = bn.get_synset(BabelSynsetID(synset_id))
            lemma = get_lemma(synset)

            hypernyms = deduplicate(fetch_hypernyms(synset, max_items))
            meronyms = deduplicate(fetch_meronyms(synset, max_items))

            results.append({
                "id": synset_id,
                "lemma": lemma,
                "hypernyms": hypernyms,
                "meronyms": meronyms
            })
        except Exception as e:
            print(f"[!] Error fetching relations for synset {synset_id}: {type(e).__name__}: {e}")
    return results

def get_lemma(synset):
    main_sense = synset.main_sense(Language.EN)
    if main_sense:
        return main_sense.full_lemma
    return "N/A"

def deduplicate(items):
    seen = set()
    deduped = []
    for item in items:
        if item["lemma"] not in seen:
            deduped.append(item)
            seen.add(item["lemma"])
    return deduped

# <<< CHANGED FUNCTION:
def has_required_relations(synset, max_items=10):
    """
    Return True if synset has at least:
        - one hypernym
        - one meronym
        - at least 3 distractor words drawn from hyponyms, cohyponyms, or antonyms
    """
    try:
        hypernyms = deduplicate(fetch_hypernyms(synset, max_items))
        meronyms = deduplicate(fetch_meronyms(synset, max_items))
        hyponyms = deduplicate(fetch_hyponyms(synset, max_items))
        cohyponyms = deduplicate(get_cohyponyms(synset, max_items))
        antonyms = deduplicate(fetch_antonyms(synset, max_items))

        distractors = hyponyms + cohyponyms + antonyms

        has_hypernyms = len(hypernyms) >= 1
        has_meronyms = len(meronyms) >= 1
        has_enough_distractors = len(distractors) >= 3

        if not has_hypernyms:
            print(f"[!] Missing hypernyms for {synset.id}")
        if not has_meronyms:
            print(f"[!] Missing meronyms for {synset.id}")
        if not has_enough_distractors:
            print(f"[!] Not enough distractors for {synset.id} — found {len(distractors)}")

        return has_hypernyms and has_meronyms and has_enough_distractors

    except Exception as e:
        print(f"[!] Error checking required relations for {synset.id}: {type(e).__name__}: {e}")
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
                if has_required_relations(synset):
                    print(f"[Line {i+1}] Synset {synset_id} meets all criteria. Writing to output.")
                    outfile.write(synset_id + "\n")
                else:
                    print(f"[Line {i+1}] Synset {synset_id} does not meet required criteria.")
            except Exception as e:
                print(f"[Line {i+1}] Failed to process {synset_id}: {type(e).__name__}: {e}")

if __name__ == "__main__":
    input_path = "synset_sememes.txt"
    output_path = "babelnet_with_relations.txt"
    process_file(input_path, output_path, max_lines=10)
    print(f"Processed up to 15000 lines. Output written to {output_path}")
