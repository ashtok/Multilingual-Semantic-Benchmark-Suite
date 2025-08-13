import pandas as pd
import glob
import os

# Directory where cleaned CSVs are saved (adjust path if needed)
cleaned_dir = r"D:\Masters In Germany\Computer Science\Semester 4\Practical_NLP\Babelnet_Client\results\CompiledResults"

# Find all cleaned_results_*.csv files dynamically
files = glob.glob(os.path.join(cleaned_dir, "cleaned_results_*.csv"))

if not files:
    raise FileNotFoundError(f"No cleaned_results_*.csv files found in {cleaned_dir}")

print(f"Found {len(files)} cleaned CSV files to merge.")

# Load CSVs
dfs = [pd.read_csv(f) for f in files]

# Concatenate all dataframes
merged_df = pd.concat(dfs, ignore_index=True)

# Drop unnecessary columns if they exist, including the new ones requested
columns_to_remove = [
    "file_path",
    "date",
    "evaluation_time",
    "acc_stderr,none",
    "acc_norm,none",
    "acc_norm_stderr,none"
]

columns_to_drop = [col for col in columns_to_remove if col in merged_df.columns]
merged_df.drop(columns=columns_to_drop, inplace=True)

# Now drop duplicates again (to handle duplicates across files)
merged_df.drop_duplicates(inplace=True)

# Save the final CSV
output_path = os.path.join(cleaned_dir, "merged_results.csv")
merged_df.to_csv(output_path, index=False)

print(f"✅ Merge complete. Duplicates removed. File saved as {output_path}")
