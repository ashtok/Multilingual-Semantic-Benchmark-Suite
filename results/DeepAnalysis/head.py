import os
import pandas as pd

# Base path to your detailed_model_analysis directory
BASE_DIR = r"D:\Masters In Germany\Computer Science\Semester 4\Practical_NLP\Babelnet_Client\results\DeepAnalysis\detailed_model_analysis"


def print_csv_heads(base_path, head_rows=5):
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)

        # Skip non-directories
        if not os.path.isdir(folder_path):
            continue

        # Loop through files inside the model folder
        for file in os.listdir(folder_path):
            if file.lower().endswith(".csv"):
                file_path = os.path.join(folder_path, file)

                try:
                    df = pd.read_csv(file_path)
                    print(f"\n=== {file} ({folder}) ===")
                    print(df.head(head_rows))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")


if __name__ == "__main__":
    print_csv_heads(BASE_DIR)
