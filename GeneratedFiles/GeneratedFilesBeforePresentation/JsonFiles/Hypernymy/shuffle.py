import json
import random

# ABSOLUTE paths
INPUT_PATH = r"D:\Masters In Germany\Computer Science\Semester 4\Practical_NLP\Babelnet_Client\GeneratedFiles\JsonFiles\Hypernymy\hypernymy_questions_all.json"
OUTPUT_PATH = r"D:\Masters In Germany\Computer Science\Semester 4\Practical_NLP\Babelnet_Client\GeneratedFiles\JsonFiles\Hypernymy\hypernymy_questions_all4000.json"

N_SAMPLES = 4000
SEED = 42

# load JSON array
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# shuffle
random.seed(SEED)
random.shuffle(data)

# take the first N
subset = data[:N_SAMPLES]

# write JSON array
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(subset, f, indent=2, ensure_ascii=False)

print(f"Wrote {len(subset)} records to {OUTPUT_PATH}")
