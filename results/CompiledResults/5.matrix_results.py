import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==== CONFIGURATION ====
# Choose color palette for all heatmaps:
# Examples: "Blues", "BuGn", "coolwarm", "viridis", "YlGnBu"
heatmap_cmap = "YlGnBu"

# --- Load & preprocess ---
df = pd.read_csv("merged_results.csv")
df['shot_type'] = df['n_shot'].apply(lambda x: 'few-shot' if x > 0 else 'zero-shot')

# Extract main task category and difficulty
df['task_category'] = df['task_name'].apply(lambda x: x.split('_')[0])
df['difficulty'] = df['task_name'].apply(lambda x: x.split('_')[-1] if '_' in x else 'misc')

# --- 1. Per-task few-shot ranking ---
few_shot_df = df[df['shot_type'] == 'few-shot']
few_shot_ranking = few_shot_df.groupby(['task_name', 'model_name'])['acc,none'].mean().reset_index()
few_shot_ranking['rank'] = few_shot_ranking.groupby('task_name')['acc,none'].rank(ascending=False, method='min')
few_shot_ranking = few_shot_ranking.sort_values(['task_name', 'rank'])
few_shot_ranking.to_csv("few_shot_ranking_per_task.csv", index=False)

# --- 2. Heatmap matrix for each model ---
for model in df['model_name'].unique():
    model_data = few_shot_df[few_shot_df['model_name'] == model]
    heatmap_data = model_data.pivot_table(index='difficulty',
                                          columns='task_category',
                                          values='acc,none')
    plt.figure(figsize=(8, 5))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap=heatmap_cmap, cbar_kws={'label': 'Accuracy'})
    plt.title(f"Accuracy Heatmap (Few-Shot) for {model}")
    plt.tight_layout()
    plt.savefig(f"{model.replace('/', '_')}_heatmap.png")
    plt.close()

# --- 3. Model comparison for _all tasks ---
all_tasks_df = df[(df['shot_type'] == 'few-shot') & (df['difficulty'] == 'all')]

# Bar chart version
plt.figure(figsize=(8, 6))
sns.barplot(data=all_tasks_df, x='task_category', y='acc,none', hue='model_name')
plt.title("Few-Shot Accuracy on '_all' Tasks by Model")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("model_comparison_all_tasks.png")
plt.close()

# Heatmap version
heatmap_all_tasks = all_tasks_df.pivot(index='model_name', columns='task_category', values='acc,none')
plt.figure(figsize=(8, 5))
sns.heatmap(heatmap_all_tasks, annot=True, fmt=".2f", cmap=heatmap_cmap, cbar_kws={'label': 'Accuracy'})
plt.title("Few-Shot Accuracy Heatmap ('_all' Tasks)")
plt.tight_layout()
plt.savefig("model_comparison_all_tasks_heatmap.png")
plt.close()

print("✅ Outputs:")
print("- few_shot_ranking_per_task.csv")
print(f"- Heatmaps per model (PNG, palette={heatmap_cmap})")
print("- model_comparison_all_tasks.png (bar chart)")
print(f"- model_comparison_all_tasks_heatmap.png (heatmap, palette={heatmap_cmap})")
