import json

# Replace these with your real file paths
input_file = "hypernymy_questions.json"
output_file = "hypernymy_questions.jsonl"

# Step 1: Load JSON data
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Step 2: Write each object as a separate line
with open(output_file, "w", encoding="utf-8") as f_out:
    for item in data:
        json_line = json.dumps(item, ensure_ascii=False)
        f_out.write(json_line + "\n")

print("Conversion complete. JSONL file saved to", output_file)
