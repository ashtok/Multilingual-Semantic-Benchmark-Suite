# Define input and output file paths
input_file = 'synset_sememes.txt'    # your original file
output_file = 'babelnet_ids.txt'  # file to save extracted IDs

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        # Each line starts with the BabelNet ID followed by a tab
        # Extract the part before the tab character
        babelnet_id = line.split('\t')[0].strip()
        outfile.write(babelnet_id + '\n')

print(f"Extraction done! IDs saved to {output_file}")
