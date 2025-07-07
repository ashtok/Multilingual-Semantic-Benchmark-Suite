import json

input_file = 'D:\\Masters In Germany\\Computer Science\\Semester 4\\Practical_NLP\\Babelnet_Client\\hypernymy_questions.json'

output_file = 'hypernymy_multilingual.jsonl'

with open(input_file, 'r', encoding='utf-8') as infile:
    data = json.load(infile)  # Load entire array

with open(output_file, 'w', encoding='utf-8') as outfile:
    for entry in data:
        json_line = json.dumps(entry, ensure_ascii=False)
        outfile.write(json_line + '\n')