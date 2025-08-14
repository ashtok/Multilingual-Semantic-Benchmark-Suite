import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==== CONFIGURATION ====
INPUT_FILE = "merged_results.csv"
OUTPUT_DIR = "results"
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
HEATMAP_DIR = os.path.join(PLOT_DIR, "heatmaps")
BAR_DIR = os.path.join(PLOT_DIR, "bars")

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)
os.makedirs(BAR_DIR, exist_ok=True)

BAR_PALETTE = "YlGnBu"
HEATMAP_PALETTE = "Blues"

# ==== LOAD DATA ====
df = pd.read_csv(INPUT_FILE)
df['shot_type'] = df['n_shot'].apply(lambda x: 'few-shot' if x > 0 else 'zero-shot')
df['task_category'] = df['task_name'].apply(lambda x: x.split('_')[0])
df['language_resource'] = df['task_name'].apply(lambda x: x.split('_')[-1] if '_' in x else 'misc')

# ==== 1. Summary by model and task ====
summary_df = df.groupby(['model_name', 'task_name', 'shot_type'])['acc,none'].mean().reset_index()
summary_csv = os.path.join(CSV_DIR, "summary_by_model_task.csv")
summary_df.to_csv(summary_csv, index=False)

# Pivot for zero vs few-shot comparison
comparison_df = summary_df.pivot_table(
    index=['model_name', 'task_name'],
    columns='shot_type',
    values='acc,none'
).reset_index()

comparison_df['few_minus_zero'] = comparison_df['few-shot'] - comparison_df['zero-shot']
comparison_df['few_vs_zero_pct'] = (comparison_df['few-shot'] - comparison_df['zero-shot']) / comparison_df[
    'zero-shot'] * 100

comparison_csv = os.path.join(CSV_DIR, "zero_vs_few_comparison.csv")
comparison_df.to_csv(comparison_csv, index=False)

# ==== 2. Per-task few-shot ranking ====
few_shot_df = df[df['shot_type'] == 'few-shot'].copy()
few_shot_ranking = few_shot_df.groupby(['task_name', 'model_name'])['acc,none'].mean().reset_index()
few_shot_ranking['rank'] = few_shot_ranking.groupby('task_name')['acc,none'].rank(ascending=False, method='min')
few_shot_ranking = few_shot_ranking.sort_values(['task_name', 'rank'])
ranking_csv = os.path.join(CSV_DIR, "few_shot_ranking_per_task.csv")
few_shot_ranking.to_csv(ranking_csv, index=False)

# ==== Plotting label mappings ====
language_resource_plot_labels = {
    'mono': 'English',
    'high': 'High',
    'medium': 'Medium',
    'all': 'All',
    'low': 'Low'
}

task_category_plot_labels = {
    'gloss': 'Gloss',
    'hypernymy': 'Hypernymy',
    'meronymy': 'Meronymy',
    'analogies': 'Analogies'
}

language_resource_order = ['mono', 'high', 'medium', 'all', 'low']
task_category_order = ['gloss', 'hypernymy', 'meronymy', 'analogies']

# ==== 3. Heatmaps per model (ordered) ====
for model in df['model_name'].unique():
    model_data = few_shot_df[few_shot_df['model_name'] == model]
    heatmap_data = model_data.pivot_table(
        index='language_resource',
        columns='task_category',
        values='acc,none'
    ).reindex(index=language_resource_order, columns=task_category_order)

    # Map labels for plotting
    heatmap_data.index = heatmap_data.index.map(language_resource_plot_labels)
    heatmap_data.columns = heatmap_data.columns.map(task_category_plot_labels)

    plt.figure(figsize=(8, 5))
    sns.heatmap(
        heatmap_data,
        annot=True, fmt=".2f",
        cmap=HEATMAP_PALETTE,
        cbar_kws={'label': 'Accuracy'}
    )
    plt.title(f"Accuracy Heatmap (Few-Shot) for {model}", fontsize=14)
    plt.xlabel("Task Category", fontsize=12)
    plt.ylabel("Language Resource", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(HEATMAP_DIR, f"{model.replace('/', '_')}_heatmap.png"))
    plt.close()

# ==== 4. Model comparison for '_all' tasks ====
all_tasks_df = df[(df['shot_type'] == 'few-shot') & (df['language_resource'] == 'all')].copy()

# Mapping and simplified model names
model_name_map = {
    'meta-llama/Llama-3.1-8B-Instruct': 'Llama-3.1-8B-Instruct',
    'Qwen/Qwen3-8B': 'Qwen3-8B',
    'mistralai/Mistral-7B-Instruct-v0.3': 'Mistral-7B-Instruct-v0.3',
    'google/gemma-7b-it': 'Gemma-7B-IT',
    'google/gemma-3-1b-it': 'Gemma-3-1B-IT'
}
all_tasks_df['model_short'] = all_tasks_df['model_name'].map(model_name_map)
model_order_full = list(model_name_map.keys())
model_order_short = [model_name_map[m] for m in model_order_full]

# Bar chart
plt.figure(figsize=(8, 6))
sns.barplot(
    data=all_tasks_df,
    x='task_category',
    y='acc,none',
    hue='model_short',
    palette=BAR_PALETTE,
    order=task_category_order,
    hue_order=model_order_short
)
plt.title("Few-Shot Accuracy on '_All' Tasks by Model", fontsize=14)
plt.ylabel("Accuracy", fontsize=12)
plt.xlabel("Task Category", fontsize=12)
plt.xticks(ticks=range(len(task_category_order)),
           labels=[task_category_plot_labels[t] for t in task_category_order])
plt.tight_layout()
plt.savefig(os.path.join(BAR_DIR, "model_comparison_all_tasks.png"))
plt.close()

# Heatmap for '_all' tasks
heatmap_all_tasks = all_tasks_df.pivot(
    index='model_short',
    columns='task_category',
    values='acc,none'
).reindex(index=model_order_short, columns=task_category_order)

heatmap_all_tasks.columns = heatmap_all_tasks.columns.map(task_category_plot_labels)

plt.figure(figsize=(8, 5))
sns.heatmap(
    heatmap_all_tasks,
    annot=True, fmt=".2f",
    cmap=HEATMAP_PALETTE,
    cbar_kws={'label': 'Accuracy'}
)
plt.title("Few-Shot Accuracy Heatmap ('_All' Tasks)", fontsize=14)
plt.xlabel("Task Category", fontsize=12)
plt.ylabel("Model Name", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(HEATMAP_DIR, "model_comparison_all_tasks_heatmap.png"))
plt.close()


# ==== 5. Bar plots per model ====
def plot_bar_accuracy(summary_df, palette=BAR_PALETTE):
    models = summary_df['model_name'].unique()
    for model in models:
        model_data = summary_df[summary_df['model_name'] == model].copy()

        # Map labels for plotting
        model_data['task_name_plot'] = model_data['task_name'].map(
            lambda x: task_category_plot_labels.get(x.split('_')[0], x.title())
        )
        model_data['shot_type_plot'] = model_data['shot_type'].str.title()

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=model_data,
            x='task_name_plot',
            y='acc,none',
            hue='shot_type_plot',
            palette=palette
        )
        plt.title(f"Accuracy by Task for {model}", fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Accuracy", fontsize=12)
        plt.xlabel("Task Name", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(BAR_DIR, f"{model.replace('/', '_')}_accuracy_comparison.png"))
        plt.close()


plot_bar_accuracy(summary_df, palette=BAR_PALETTE)

print(f"✅ Outputs saved in '{OUTPUT_DIR}':")
print(f"- CSVs: {CSV_DIR}")
print(f"- Bar charts: {BAR_DIR}")
print(f"- Heatmaps: {HEATMAP_DIR}")
