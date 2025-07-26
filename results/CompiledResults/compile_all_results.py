import pandas as pd

# CSVs to read
files = [
    "cleaned_results_meronymy.csv",
    "cleaned_results_hypernymy.csv",
    "cleaned_results_analogy.csv"
]

# Load CSVs
dfs = [pd.read_csv(f) for f in files]

# Concatenate all dataframes
merged_df = pd.concat(dfs, ignore_index=True)

# Drop unnecessary columns if they exist
columns_to_remove = ["file_path", "date", "evaluation_time"]

# Only drop columns that are actually present
columns_to_drop = [col for col in columns_to_remove if col in merged_df.columns]

merged_df.drop(columns=columns_to_drop, inplace=True)

# Now drop duplicates
merged_df = merged_df.drop_duplicates()

# Save the final CSV
merged_df.to_csv("merged_results.csv", index=False)

print("✅ Merge complete. Duplicates removed. File saved as merged_results.csv.")
