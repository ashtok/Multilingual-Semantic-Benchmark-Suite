import json
import hashlib
import csv
import os


def create_unique_key(entry):
    """
    Create a unique key for each entry based on model_name, task_name,
    and the accuracy-related fields that should be considered for duplicates.
    """
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

    # Create a hash of the key fields for consistent comparison
    key_string = "|".join(key_fields)
    return hashlib.md5(key_string.encode()).hexdigest()


def read_input_file(file_path):
    """
    Read input file (JSON or CSV) and return data in standardized format.

    Args:
        file_path (str): Path to input file

    Returns:
        dict: Data with 'results' key containing list of entries
    """
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.json':
        with open(file_path, 'r') as file:
            data = json.load(file)
            # Handle both formats: direct list or dict with 'results' key
            if isinstance(data, list):
                return {'results': data}
            elif isinstance(data, dict) and 'results' in data:
                return data
            else:
                raise ValueError("JSON file must contain either a list of results or a dict with 'results' key")

    elif file_ext == '.csv':
        results = []
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Convert string numbers back to appropriate types
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
        raise ValueError(f"Unsupported file format: {file_ext}. Please use .json or .csv files.")


def remove_duplicates(input_file_path, output_json_path=None, output_csv_path=None):
    """
    Remove duplicate entries from input file (JSON or CSV) based on model_name, task_name,
    and accuracy-related fields.

    Args:
        input_file_path (str): Path to input file (JSON or CSV)
        output_json_path (str): Path to output JSON file (optional)
        output_csv_path (str): Path to output CSV file (optional)

    Returns:
        dict: Cleaned data with duplicates removed
    """

    # Read the input file (JSON or CSV)
    data = read_input_file(input_file_path)

    file_ext = os.path.splitext(input_file_path)[1].lower()
    print(f"Processing {file_ext.upper()} file: {input_file_path}")

    # Track seen entries and store unique ones
    seen_keys = set()
    unique_results = []
    duplicates_removed = 0

    print(f"Original number of entries: {len(data['results'])}")

    for entry in data['results']:
        # Create unique key for this entry
        unique_key = create_unique_key(entry)

        # If we haven't seen this combination before, add it
        if unique_key not in seen_keys:
            seen_keys.add(unique_key)
            unique_results.append(entry)
        else:
            duplicates_removed += 1
            print(f"Duplicate found: {entry['model_name']} - {entry['task_name']}")

    # Update the data with unique results
    data['results'] = unique_results

    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Final number of entries: {len(unique_results)}")

    # Save to output files if specified
    if output_json_path:
        with open(output_json_path, 'w') as file:
            json.dump(data, file, indent=2)
        print(f"Cleaned JSON data saved to: {output_json_path}")

    if output_csv_path:
        save_to_csv(unique_results, output_csv_path)
        print(f"Cleaned CSV data saved to: {output_csv_path}")

    return data


def save_to_csv(results, csv_file_path):
    """
    Save results to CSV file with specified column order.

    Args:
        results (list): List of result dictionaries
        csv_file_path (str): Path to output CSV file
    """
    # Define the column order as specified
    fieldnames = [
        'model_name',
        'task_name',
        'file_path',
        'date',
        'evaluation_time',
        'acc,none',
        'acc_stderr,none',
        'acc_norm,none',
        'acc_norm_stderr,none',
        'n_samples_original',
        'n_samples_effective',
        'n_shot'
    ]

    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            # Create a row with only the fields we want
            row = {field: result.get(field, '') for field in fieldnames}
            writer.writerow(row)


def display_summary(data):
    """Display a summary of the cleaned data."""
    print("\n" + "=" * 50)
    print("SUMMARY OF CLEANED DATA")
    print("=" * 50)

    # Group by model
    models = {}
    for entry in data['results']:
        model = entry['model_name']
        task = entry['task_name']

        if model not in models:
            models[model] = []
        models[model].append(task)

    for model, tasks in models.items():
        print(f"\n{model}:")
        for task in sorted(set(tasks)):
            print(f"  - {task}")


# Example usage
if __name__ == "__main__":
    # Input file can be either JSON or CSV
    input_file = "./output/collected_data_20250715_114802.json"
    output_json_file = "cleaned_results_meronymy.json"
    output_csv_file = "cleaned_results_meronymy.csv"

    try:
        # Remove duplicates and save to both JSON and CSV
        cleaned_data = remove_duplicates(input_file, output_json_file, output_csv_file)

        # Display summary
        display_summary(cleaned_data)

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        print("Please make sure the file exists and the path is correct.")
        print("Supported formats: .json, .csv")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{input_file}'.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")