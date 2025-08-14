import pandas as pd

# --- Load data ---
df = pd.read_csv("merged_results.csv")

# Extract task category and difficulty
df['task_category'] = df['task_name'].apply(lambda x: x.split('_')[0])
df['difficulty'] = df['task_name'].apply(lambda x: x.split('_')[-1] if '_' in x else 'all')

# Take only one row per task (ignore duplicates across models)
unique_tasks = df[['task_category', 'difficulty', 'n_samples_original']].drop_duplicates()

# Pivot for table-like view
table = unique_tasks.pivot(index='task_category', columns='difficulty', values='n_samples_original').fillna(0).astype(int)

# Optional: add 'all' column as sum across difficulties
table['all'] = table.sum(axis=1)

# Reorder columns
cols = ['all', 'high', 'low', 'medium', 'mono']
table = table[cols]

# Save and print
table.to_csv("questions_count_table_corrected.csv")
print("Number of Questions per Task Category and Difficulty:\n")
print(table)
