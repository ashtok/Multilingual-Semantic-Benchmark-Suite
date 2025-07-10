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


def print_list(title, items):
    """Nicely print a list of relation items."""
    print(f"\n{title}:")
    if not items:
        print("   (none found)")
    else:
        for i, item in enumerate(items, start=1):
            print(f"   [{i}] {item['lemma']} ({item['id']})")


def print_relations(synset_id_str, max_items=10):
    """Fetch and print all relevant relations for a given synset ID."""
    try:
        synset = bn.get_synset(BabelSynsetID(synset_id_str))
        lemma = get_lemma(synset)
        print(f"\n=== Synset: {lemma} ({synset_id_str}) ===")
    except Exception as e:
        print(f"[!] Failed to retrieve synset {synset_id_str}: {type(e).__name__}: {e}")
        return

    # Fetch each relation type with isolated error handling
    try:
        hypernyms = deduplicate(fetch_hypernyms(synset, max_items))
    except Exception as e:
        print(f"[!] Failed to fetch hypernyms: {type(e).__name__}: {e}")
        hypernyms = []

    try:
        hyponyms = deduplicate(fetch_hyponyms(synset, max_items))
    except Exception as e:
        print(f"[!] Failed to fetch hyponyms: {type(e).__name__}: {e}")
        hyponyms = []

    try:
        meronyms = deduplicate(fetch_meronyms(synset, max_items))
    except Exception as e:
        print(f"[!] Failed to fetch meronyms: {type(e).__name__}: {e}")
        meronyms = []

    try:
        cohyponyms = deduplicate(get_cohyponyms(synset, max_items))
    except Exception as e:
        print(f"[!] Failed to fetch co-hyponyms: {type(e).__name__}: {e}")
        cohyponyms = []

    print_list("Hypernyms", hypernyms)
    print_list("Hyponyms", hyponyms)
    print_list("Meronyms", meronyms)
    print_list("Co-Hyponyms", cohyponyms)


if __name__ == "__main__":
    # Example BabelNet synset IDs (feel free to add more or replace)
    synset_ids = [
        "bn:00015267n",     # Dog
        "bn:00042379n",     # Water
        "bn:00104078a",
    ]

    for sid in synset_ids:
        print_relations(sid)
