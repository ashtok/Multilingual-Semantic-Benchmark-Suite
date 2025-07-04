import babelnet as bn
from babelnet import BabelSynsetID, Language
from tqdm import tqdm  # progress bar

import requests
import time

INPUT_FILE = 'category_keywords_babelnet.txt'
OUTPUT_FILE = 'babelnet_ids_output.txt'


def get_babelnet_ids(word):
    url = 'https://babelnet.io/v6/getSynsetIds'
    params = {
        'lemma': word,
        'searchLang': 'EN',
        'key': API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()  # List of synsets
    else:
        print(f"Failed to get BabelNet IDs for {word}: Status {response.status_code}")
        return []


def main():
    # Read the input file and extract words (just a simple approach)
    words = set()
    with open(INPUT_FILE, 'r') as file:
        for line in file:
            # split by common delimiters
            for part in line.strip().split():
                # clean punctuation and commas
                cleaned = part.strip(',').lower()
                if cleaned.isalpha():
                    words.add(cleaned)

    print(f"Total unique words to query: {len(words)}")

    with open(OUTPUT_FILE, 'w') as outfile:
        for word in sorted(words):
            synsets = get_babelnet_ids(word)
            if synsets:
                ids = [entry['id'] for entry in synsets]
                outfile.write(f"{word}: {', '.join(ids)}\n")
            else:
                outfile.write(f"{word}: No BabelNet ID found\n")
            time.sleep(0.25)  # polite delay to avoid API throttling


if __name__ == "__main__":
    main()


