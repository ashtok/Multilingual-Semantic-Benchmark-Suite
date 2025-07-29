import babelnet as bn
from babelnet import Language, BabelSynsetID, BabelSenseSource
from babelnet.data.relation import BabelPointer


def print_all_synset_data(synset_id: str):
    """
    Print all available data for a BabelNet synset ID
    Usage: print_all_synset_data('bn:00000356n')
    """

    # Get the synset
    synset = bn.get_synset(BabelSynsetID(synset_id))

    print("=" * 60)
    print(f"SYNSET DATA FOR: {synset_id}")
    print("=" * 60)

    # Basic synset info
    print(f"Synset ID: {synset.id}")
    print()

    # Main sense in English
    main_sense = synset.main_sense(Language.EN)
    if main_sense:
        print(f"Main sense (EN): {main_sense.full_lemma}")
    print()

    # All senses
    print("ALL SENSES:")
    print("-" * 40)
    for sense in synset:
        print(f"Sense: {sense.full_lemma}\tLanguage: {sense.language}\tSource: {sense.source}")

        # Pronunciations if available
        if hasattr(sense, 'pronunciations') and sense.pronunciations:
            for audio in sense.pronunciations.audios:
                print(f"  Audio URL: {audio.validated_url}")
    print()

    # Wikidata senses
    print("WIKIDATA SENSES:")
    print("-" * 40)
    for sense in synset.senses(source=BabelSenseSource.WIKIDATA):
        sensekey = sense.sensekey
        print(f"{sense.full_lemma}\t{sense.language}\t{sensekey}")
    print()

    # Glosses in different languages
    print("GLOSSES (DEFINITIONS):")
    print("-" * 40)
    for lang in [Language.EN, Language.IT, Language.FR, Language.ES, Language.DE]:
        try:
            gloss = synset.main_gloss(lang)
            if gloss:
                print(f"{lang}: {gloss.gloss}")
        except:
            continue
    print()

    # All relation types
    print("SEMANTIC RELATIONS:")
    print("-" * 40)

    # Hypernyms (IS-A relationships)
    print("HYPERNYMS (IS-A):")
    for edge in synset.outgoing_edges(BabelPointer.ANY_HYPERNYM):
        target_synset = bn.get_synset(edge.id_target)
        target_main = target_synset.main_sense(Language.EN)
        target_lemma = target_main.full_lemma if target_main else "N/A"
        print(f"{synset.id} - {edge.pointer} - {edge.id_target} - {target_lemma}")

    # Hyponyms (HAS-KIND relationships)
    print("\nHYPONYMS (HAS-KIND):")
    for edge in synset.outgoing_edges(BabelPointer.ANY_HYPONYM):
        target_synset = bn.get_synset(edge.id_target)
        target_main = target_synset.main_sense(Language.EN)
        target_lemma = target_main.full_lemma if target_main else "N/A"
        print(f"{synset.id} - {edge.pointer} - {edge.id_target} - {target_lemma}")

    # Meronyms (HAS-PART relationships)
    print("\nMERONYMS (HAS-PART):")
    for edge in synset.outgoing_edges(BabelPointer.ANY_MERONYM):
        target_synset = bn.get_synset(edge.id_target)
        target_main = target_synset.main_sense(Language.EN)
        target_lemma = target_main.full_lemma if target_main else "N/A"
        print(f"{synset.id} - {edge.pointer} - {edge.id_target} - {target_lemma}")

    # Holonyms (PART-OF relationships)
    print("\nHOLONYMS (PART-OF):")
    for edge in synset.outgoing_edges(BabelPointer.ANY_HOLONYM):
        target_synset = bn.get_synset(edge.id_target)
        target_main = target_synset.main_sense(Language.EN)
        target_lemma = target_main.full_lemma if target_main else "N/A"
        print(f"{synset.id} - {edge.pointer} - {edge.id_target} - {target_lemma}")

    # Similar relationships
    print("\nSIMILAR:")
    for edge in synset.outgoing_edges(BabelPointer.SIMILAR_TO):
        target_synset = bn.get_synset(edge.id_target)
        target_main = target_synset.main_sense(Language.EN)
        target_lemma = target_main.full_lemma if target_main else "N/A"
        print(f"{synset.id} - {edge.pointer} - {edge.id_target} - {target_lemma}")

    # Also relationships
    print("\nALSO:")
    for edge in synset.outgoing_edges(BabelPointer.ALSO):
        target_synset = bn.get_synset(edge.id_target)
        target_main = target_synset.main_sense(Language.EN)
        target_lemma = target_main.full_lemma if target_main else "N/A"
        print(f"{synset.id} - {edge.pointer} - {edge.id_target} - {target_lemma}")

    # Derivation relationships
    print("\nDERIVATION:")
    for edge in synset.outgoing_edges(BabelPointer.DERIVATION):
        target_synset = bn.get_synset(edge.id_target)
        target_main = target_synset.main_sense(Language.EN)
        target_lemma = target_main.full_lemma if target_main else "N/A"
        print(f"{synset.id} - {edge.pointer} - {edge.id_target} - {target_lemma}")

    # Other relationships
    print("\nOTHER RELATIONS:")
    for edge in synset.outgoing_edges(BabelPointer.OTHER):
        target_synset = bn.get_synset(edge.id_target)
        target_main = target_synset.main_sense(Language.EN)
        target_lemma = target_main.full_lemma if target_main else "N/A"
        print(f"{synset.id} - {edge.pointer} - {edge.id_target} - {target_lemma}")

    print()

    # Images
    print("IMAGES:")
    print("-" * 40)
    try:
        for image in synset.images():
            print(f"Image URL: {image.url}")
            if hasattr(image, 'name'):
                print(f"  Name: {image.name}")
            if hasattr(image, 'validated'):
                print(f"  Validated: {image.validated}")
    except:
        print("No images available or method not supported")
    print()

    # Categories
    print("CATEGORIES:")
    print("-" * 40)
    try:
        for category in synset.categories():
            print(f"Category: {category}")
    except:
        print("No categories available or method not supported")
    print()

    # Domains
    print("DOMAINS:")
    print("-" * 40)
    try:
        for domain in synset.domains():
            print(f"Domain: {domain}")
    except:
        print("No domains available or method not supported")
    print()

    print("=" * 60)
    print("END OF DATA")
    print("=" * 60)


# Example usage
if __name__ == "__main__":
    # Example synset IDs to test
    test_ids = [
        'bn:00000356n',  # home
        'bn:00015556n',  # car
        'bn:00017222n'  # book
    ]

    # Choose which synset to analyze
    synset_id = input("Enter BabelNet synset ID (or press Enter for 'bn:00000356n'): ").strip()
    if not synset_id:
        synset_id = 'bn:00000356n'

    try:
        print_all_synset_data(synset_id)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the synset ID is valid and BabelNet is properly configured")