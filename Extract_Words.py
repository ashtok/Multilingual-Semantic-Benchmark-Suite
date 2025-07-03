import collections
from nltk.corpus import wordnet as wn
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

CATEGORY_KEYWORDS = {
    'animal': [
        'animal', 'mammal', 'bird', 'fish', 'reptile', 'insect', 'amphibian',
        'rodent', 'dog', 'cat', 'horse', 'livestock', 'pet'
    ],
    'plant': [
        'plant', 'tree', 'flower', 'shrub', 'herb', 'grass', 'crop', 'vine',
        'fungus', 'moss', 'bush', 'weed'
    ],
    'vehicle': [
        'vehicle', 'car', 'truck', 'automobile', 'bike', 'bicycle', 'motorcycle',
        'bus', 'train', 'boat', 'ship', 'submarine', 'aircraft', 'plane', 'helicopter'
    ],
    'machine': [
        'machine', 'device', 'engine', 'motor', 'robot', 'mechanism',
        'appliance', 'tool', 'instrument', 'equipment', 'gadget'
    ],
    'food': [
        'food', 'fruit', 'vegetable', 'dish', 'grain', 'meat', 'spice',
        'drink', 'beverage', 'dessert', 'snack', 'dairy', 'seafood'
    ],
    'person': [
        'person', 'human', 'man', 'woman', 'boy', 'girl', 'child',
        'adult', 'worker', 'athlete', 'politician', 'scientist'
    ],
    'profession': [
        'profession', 'occupation', 'job', 'career', 'artist', 'musician',
        'engineer', 'doctor', 'nurse', 'teacher', 'lawyer', 'driver'
    ],
    'place': [
        'place', 'city', 'country', 'state', 'province', 'region',
        'village', 'continent', 'island', 'river', 'mountain', 'lake',
        'park', 'building', 'monument'
    ],
    'body_part': [
        'body_part', 'organ', 'limb', 'arm', 'leg', 'eye', 'ear', 'mouth',
        'head', 'heart', 'skin', 'bone', 'blood'
    ],
    'clothing': [
        'clothing', 'garment', 'shirt', 'pants', 'dress', 'coat', 'hat',
        'shoe', 'sock', 'glove', 'scarf'
    ],
    'emotion': [
        'emotion', 'feeling', 'joy', 'anger', 'fear', 'sadness',
        'happiness', 'anxiety', 'surprise'
    ],
    'material': [
        'material', 'metal', 'wood', 'plastic', 'glass', 'stone',
        'fabric', 'paper', 'cloth', 'ceramic'
    ],
    'shape': [
        'shape', 'form', 'circle', 'square', 'triangle', 'rectangle',
        'sphere', 'cube', 'cylinder', 'cone'
    ],
    'color': [
        'color', 'red', 'blue', 'green', 'yellow', 'black', 'white',
        'gray', 'brown', 'orange', 'pink', 'purple'
    ],
    'event': [
        'event', 'festival', 'holiday', 'war', 'battle', 'ceremony',
        'meeting', 'competition', 'celebration'
    ],
    'time': [
        'time', 'day', 'week', 'month', 'year', 'century', 'minute',
        'second', 'hour', 'era', 'season', 'date'
    ],
    'art': [
        'art', 'painting', 'sculpture', 'drawing', 'photograph',
        'architecture', 'music', 'dance', 'poem', 'novel'
    ],
    'technology': [
        'technology', 'computer', 'software', 'internet', 'phone',
        'application', 'website', 'hardware', 'network'
    ],
    'organization': [
        'organization', 'company', 'corporation', 'institution',
        'association', 'party', 'agency', 'group'
    ],
    'language': [
        'language', 'dialect', 'tongue'
    ],
    'currency': [
        'currency', 'money', 'coin', 'banknote', 'dollar', 'euro',
        'yen', 'peso', 'rupee'
    ]
}

def belongs_to_category(word, category_keywords):
    """
    Check if a word belongs to any of the desired categories
    by checking its hypernyms in WordNet.
    """
    synsets = wn.synsets(word, pos=wn.NOUN)
    for syn in synsets:
        hypernyms = syn.hypernym_paths()
        for path in hypernyms:
            for hyper in path:
                for cat, keywords in category_keywords.items():
                    if any(keyword in hyper.lemma_names() for keyword in keywords):
                        return cat
    return None

def process_file(filename, category_keywords, top_n=1000):
    id_counter = collections.Counter()
    category_ids = collections.defaultdict(set)

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue

            babelnet_id = parts[0]  # e.g. bn:00104083a
            eng_ch_pairs = parts[1].split()

            belongs_to_any_category = False
            for pair in eng_ch_pairs:
                if '|' in pair:
                    eng, ch = pair.split('|', 1)
                    eng_lower = eng.lower()

                    category = belongs_to_category(eng_lower, category_keywords)
                    if category:
                        category_ids[category].add(babelnet_id)
                        belongs_to_any_category = True

            # Count the ID if any word belonged to a category
            if belongs_to_any_category:
                id_counter[babelnet_id] += 1

    top_ids = id_counter.most_common(top_n)
    return top_ids, category_ids

def save_top_ids(top_ids, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for babelnet_id, count in top_ids:
            f.write(f"{babelnet_id}\t{count}\n")
    print(f"Saved top {len(top_ids)} BabelNet IDs to {output_file}")

def save_category_ids(category_ids, prefix="category_"):
    for category, ids in category_ids.items():
        filename = f"{prefix}{category}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            for bn_id in sorted(ids):
                f.write(f"{bn_id}\n")
        print(f"Saved {len(ids)} BabelNet IDs for category '{category}' to {filename}")

if __name__ == "__main__":
    filename = "synset_sememes.txt"  # replace with your actual file path

    top_ids, category_ids = process_file(filename, CATEGORY_KEYWORDS, top_n=1000)

    print("=== Top 1000 BabelNet IDs ===")
    for bn_id, count in top_ids:
        print(f"{bn_id}\t{count}")

    print("\n=== Category BabelNet IDs ===")
    for category, ids in category_ids.items():
        print(f"{category}: {', '.join(sorted(ids))}")

    save_top_ids(top_ids, "top_1000_babelnet_ids.txt")
    save_category_ids(category_ids, prefix="category_")
