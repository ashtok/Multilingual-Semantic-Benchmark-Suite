import json
import hashlib
import csv
import os
import glob


def create_unique_key(entry):
    """Create a unique hash key based on identifying fields to detect duplicates."""
    key_fields = [
        entry.get("model_name", ""),
        entry.get("task_name", ""),
        str(entry.get("acc,none", "")),
        str(entry.get("acc_stderr,none", "")),
        str(entry.get("acc_norm,none", "")),
        str(entry.get("acc_norm_stderr,none", "")),
        str(entry.get("n_samples_original", "")),
        str(entry.get("n_samples_effective", "")),
        str(entry.get("n_shot", ""))
    ]
    return hashlib.md5("|".join(key_fields).encode()).hexdigest()


def read_input_file(file_path):
    """Read input JSON or CSV file and normalize to dict with 'results' key."""
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.json':
        with open(file_path, 'r') as file:
            data = json.load(file)
            if isinstance(data, list):
                return {'results': data}
            elif isinstance(data, dict) and 'results' in data:
                return data
            else:
                raise ValueError("JSON must contain a list or a dict with 'results' key")

    elif file_ext == '.csv':
        results = []
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                converted_row = {}
                for key, value in row.items():
                    if value == '':
                        converted_row[key] = None
                    elif key in ['date', 'evaluation_time', 'acc,none', 'acc_stderr,none',
                                 'acc_norm,none', 'acc_norm_stderr,none']:
                        try:
                            converted_row[key] = float(value)
                        except ValueError:
                            converted_row[key] = value
                    elif key in ['n_samples_original', 'n_samples_effective', 'n_shot']:
                        try:
                            converted_row[key] = int(value)
                        except ValueError:
                            converted_row[key] = value
                    else:
                        converted_row[key] = value
                results.append(converted_row)
        return {'results': results}

    else:
        raise ValueError(f"Unsupported format: {file_ext}")


def remove_duplicates(input_file_path, output_json_path=None, output_csv_path=None):
    """Remove duplicates and optionally save cleaned results."""
    data = read_input_file(input_file_path)
    seen_keys = set()
    unique_results = []
    duplicates_removed = 0

    print(f"Processing {input_file_path}")
    print(f"Original entries: {len(data['results'])}")

    for entry in data['results']:
        unique_key = create_unique_key(entry)
        if unique_key not in seen_keys:
            seen_keys.add(unique_key)
            unique_results.append(entry)
        else:
            duplicates_removed += 1
            print(f"Duplicate found: {entry['model_name']} - {entry['task_name']}")

    data['results'] = unique_results
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Final entries: {len(unique_results)}")

    if output_json_path:
        with open(output_json_path, 'w') as file:
            json.dump(data, file, indent=2)
        print(f"Saved cleaned JSON: {output_json_path}")

    if output_csv_path:
        save_to_csv(unique_results, output_csv_path)
        print(f"Saved cleaned CSV: {output_csv_path}")

    return data


def save_to_csv(results, csv_file_path):
    """Save results to CSV with a defined column order."""
    fieldnames = [
        'model_name', 'task_name', 'file_path', 'date', 'evaluation_time',
        'acc,none', 'acc_stderr,none', 'acc_norm,none', 'acc_norm_stderr,none',
        'n_samples_original', 'n_samples_effective', 'n_shot'
    ]

    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {field: result.get(field, '') for field in fieldnames}
            writer.writerow(row)


def display_summary(data):
    """Print summary of cleaned results."""
    print("\n" + "=" * 50)
    print("SUMMARY OF CLEANED DATA")
    print("=" * 50)
    models = {}
    for entry in data['results']:
        models.setdefault(entry['model_name'], set()).add(entry['task_name'])
    for model, tasks in models.items():
        print(f"\n{model}:")
        for task in sorted(tasks):
            print(f"  - {task}")


if __name__ == "__main__":
    # Automatically pick the latest collected_data_*.json
    consolidated_dir = r"D:\Masters In Germany\Computer Science\Semester 4\Practical_NLP\Babelnet_Client\results\CompiledResults\consolidated_results"
    json_files = glob.glob(os.path.join(consolidated_dir, "collected_data_*.json"))

    if not json_files:
        raise FileNotFoundError(f"No collected_data_*.json found in {consolidated_dir}")

    # Sort newest first
    json_files.sort(key=os.path.getmtime, reverse=True)
    input_file = json_files[0]
    print(f"Using latest collected data: {input_file}")

    # Output filenames based on timestamp in collected_data filename
    timestamp = os.path.splitext(os.path.basename(input_file))[0].replace("collected_data_", "")
    output_json_file = f"cleaned_results_{timestamp}.json"
    output_csv_file = f"cleaned_results_{timestamp}.csv"

    cleaned_data = remove_duplicates(input_file, output_json_file, output_csv_file)
    display_summary(cleaned_data)
