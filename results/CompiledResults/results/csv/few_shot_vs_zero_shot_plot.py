import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set the style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Read the data from CSV file
csv_dir = r"D:\Masters In Germany\Computer Science\Semester 4\Practical_NLP\Babelnet_Client\results\CompiledResults\results\csv"
csv_path = Path(csv_dir) / "zero_vs_few_comparison.csv"
df = pd.read_csv(csv_path)

# Clean model names for better display
df['model_short'] = df['model_name'].str.split('/').str[-1].str.replace('-Instruct-v0.3', '').str.replace('-Instruct', '')

print("Data Overview:")
print(f"Shape: {df.shape}")
print(f"Models: {df['model_name'].unique()}")
print(f"Tasks: {df['task_name'].unique()}")
print("\nFirst few rows:")
print(df.head())

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 24))

# 1. Overall comparison by model (average across all tasks)
plt.subplot(4, 2, 1)
model_avg = df.groupby('model_short')[['zero-shot', 'few-shot']].mean().sort_values('few-shot', ascending=True)
x_pos = np.arange(len(model_avg))
width = 0.35

bars1 = plt.barh(x_pos - width/2, model_avg['zero-shot'], width, label='Zero-shot', alpha=0.8)
bars2 = plt.barh(x_pos + width/2, model_avg['few-shot'], width, label='Few-shot', alpha=0.8)

plt.xlabel('Average Performance Score')
plt.title('Overall Model Performance: Zero-shot vs Few-shot\n(Averaged across all tasks)')
plt.yticks(x_pos, model_avg.index, rotation=0)
plt.legend()
plt.grid(axis='x', alpha=0.3)

# Add value labels on bars
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    plt.text(bar1.get_width() + 0.01, bar1.get_y() + bar1.get_height()/2,
             f'{model_avg.iloc[i]["zero-shot"]:.3f}', va='center', fontsize=8)
    plt.text(bar2.get_width() + 0.01, bar2.get_y() + bar2.get_height()/2,
             f'{model_avg.iloc[i]["few-shot"]:.3f}', va='center', fontsize=8)

# 2. Performance by task type (aggregated across models)
plt.subplot(4, 2, 2)
# Extract main task type (before '_')
df['task_type'] = df['task_name'].str.split('_').str[0]
task_avg = df.groupby('task_type')[['zero-shot', 'few-shot']].mean().sort_values('few-shot', ascending=True)
x_pos = np.arange(len(task_avg))

bars1 = plt.barh(x_pos - width/2, task_avg['zero-shot'], width, label='Zero-shot', alpha=0.8)
bars2 = plt.barh(x_pos + width/2, task_avg['few-shot'], width, label='Few-shot', alpha=0.8)

plt.xlabel('Average Performance Score')
plt.title('Task Type Performance: Zero-shot vs Few-shot\n(Averaged across all models)')
plt.yticks(x_pos, task_avg.index, rotation=0)
plt.legend()
plt.grid(axis='x', alpha=0.3)

# 3. Improvement percentage by model
plt.subplot(4, 2, 3)
model_improvement = df.groupby('model_short')['few_vs_zero_pct'].mean().sort_values(ascending=True)
bars = plt.barh(range(len(model_improvement)), model_improvement.values, alpha=0.8)
plt.xlabel('Average Improvement (%)')
plt.title('Few-shot Improvement over Zero-shot by Model\n(Average % improvement across tasks)')
plt.yticks(range(len(model_improvement)), model_improvement.index)
plt.grid(axis='x', alpha=0.3)

# Add value labels
for i, bar in enumerate(bars):
    plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f'{model_improvement.iloc[i]:.1f}%', va='center', fontsize=9)

# 4. Detailed heatmap of performance differences
plt.subplot(4, 2, 4)
# Create pivot table for heatmap
pivot_data = df.pivot_table(values='few_vs_zero_pct', index='model_short', columns='task_name', fill_value=0)
sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', center=0, fmt='.1f',
            cbar_kws={'label': 'Few-shot improvement (%)'})
plt.title('Few-shot Improvement Heatmap\n(% improvement over zero-shot)')
plt.xlabel('Tasks')
plt.ylabel('Models')
plt.xticks(rotation=45, ha='right')

# 5. Distribution of improvements
plt.subplot(4, 2, 5)
plt.hist(df['few_vs_zero_pct'], bins=20, alpha=0.7, edgecolor='black')
plt.axvline(df['few_vs_zero_pct'].mean(), color='red', linestyle='--',
            label=f'Mean: {df["few_vs_zero_pct"].mean():.1f}%')
plt.xlabel('Few-shot Improvement (%)')
plt.ylabel('Frequency')
plt.title('Distribution of Few-shot Improvements')
plt.legend()
plt.grid(alpha=0.3)

# 6. Best and worst performing model-task combinations
plt.subplot(4, 2, 6)
# Top 10 improvements
top_improvements = df.nlargest(10, 'few_vs_zero_pct')[['model_short', 'task_name', 'few_vs_zero_pct']]
top_improvements['label'] = top_improvements['model_short'] + '\n' + top_improvements['task_name']

bars = plt.barh(range(len(top_improvements)), top_improvements['few_vs_zero_pct'], alpha=0.8)
plt.xlabel('Improvement (%)')
plt.title('Top 10 Model-Task Combinations\n(Highest few-shot improvements)')
plt.yticks(range(len(top_improvements)), top_improvements['label'], fontsize=8)
plt.grid(axis='x', alpha=0.3)

# 7. Model comparison on specific task types
plt.subplot(4, 2, 7)
# Focus on 'all' tasks (overall performance indicators)
all_tasks = df[df['task_name'].str.contains('_all')]
if not all_tasks.empty:
    pivot_all = all_tasks.pivot(index='model_short', columns='task_name', values='few-shot')
    pivot_all.plot(kind='bar', width=0.8, alpha=0.8)
    plt.title('Model Performance on Overall Tasks\n(Few-shot scores)')
    plt.xlabel('Models')
    plt.ylabel('Few-shot Performance')
    plt.legend(title='Tasks', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)

# 8. Zero-shot vs Few-shot scatter plot
plt.subplot(4, 2, 8)
colors = plt.cm.Set3(np.linspace(0, 1, len(df['model_short'].unique())))
model_colors = dict(zip(df['model_short'].unique(), colors))

for model in df['model_short'].unique():
    model_data = df[df['model_short'] == model]
    plt.scatter(model_data['zero-shot'], model_data['few-shot'],
                label=model, alpha=0.7, s=60, color=model_colors[model])

# Add diagonal line (y=x) for reference
min_val = min(df['zero-shot'].min(), df['few-shot'].min())
max_val = max(df['zero-shot'].max(), df['few-shot'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Equal performance')

plt.xlabel('Zero-shot Performance')
plt.ylabel('Few-shot Performance')
plt.title('Zero-shot vs Few-shot Performance\n(Points above diagonal show few-shot advantage)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.grid(alpha=0.3)

plt.tight_layout()

# Save the comprehensive plot
plot_path = Path(csv_dir) / "zero_vs_few_shot_comprehensive_analysis.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n📁 Plot saved as: {plot_path}")

plt.show()

# Create and save individual focused plots

# Individual Plot 1: Model Performance Comparison
fig1, ax1 = plt.subplots(figsize=(12, 8))
model_avg = df.groupby('model_short')[['zero-shot', 'few-shot']].mean().sort_values('few-shot', ascending=True)
x_pos = np.arange(len(model_avg))
width = 0.35

bars1 = ax1.barh(x_pos - width/2, model_avg['zero-shot'], width, label='Zero-shot', alpha=0.8)
bars2 = ax1.barh(x_pos + width/2, model_avg['few-shot'], width, label='Few-shot', alpha=0.8)

ax1.set_xlabel('Average Performance Score', fontsize=12)
ax1.set_title('Model Performance Comparison: Zero-shot vs Few-shot', fontsize=14, fontweight='bold')
ax1.set_yticks(x_pos)
ax1.set_yticklabels(model_avg.index, fontsize=10)
ax1.legend(fontsize=11)
ax1.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    ax1.text(bar1.get_width() + 0.01, bar1.get_y() + bar1.get_height()/2,
             f'{model_avg.iloc[i]["zero-shot"]:.3f}', va='center', fontsize=9)
    ax1.text(bar2.get_width() + 0.01, bar2.get_y() + bar2.get_height()/2,
             f'{model_avg.iloc[i]["few-shot"]:.3f}', va='center', fontsize=9)

plt.tight_layout()
individual_plot1_path = Path(csv_dir) / "model_performance_comparison.png"
plt.savefig(individual_plot1_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"📁 Individual plot saved as: {individual_plot1_path}")
plt.close()

# Individual Plot 2: Improvement Heatmap
fig2, ax2 = plt.subplots(figsize=(14, 8))
pivot_data = df.pivot_table(values='few_vs_zero_pct', index='model_short', columns='task_name', fill_value=0)
sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', center=0, fmt='.1f',
            cbar_kws={'label': 'Few-shot improvement (%)'}, ax=ax2)
ax2.set_title('Few-shot Learning Improvement Heatmap\n(% improvement over zero-shot)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Tasks', fontsize=12)
ax2.set_ylabel('Models', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
individual_plot2_path = Path(csv_dir) / "improvement_heatmap.png"
plt.savefig(individual_plot2_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"📁 Individual plot saved as: {individual_plot2_path}")
plt.close()

# Individual Plot 3: Task Type Analysis
fig3, ax3 = plt.subplots(figsize=(10, 6))
task_avg = df.groupby('task_type')[['zero-shot', 'few-shot']].mean().sort_values('few-shot', ascending=True)
x_pos = np.arange(len(task_avg))

bars1 = ax3.barh(x_pos - width/2, task_avg['zero-shot'], width, label='Zero-shot', alpha=0.8)
bars2 = ax3.barh(x_pos + width/2, task_avg['few-shot'], width, label='Few-shot', alpha=0.8)

ax3.set_xlabel('Average Performance Score', fontsize=12)
ax3.set_title('Task Type Performance: Zero-shot vs Few-shot', fontsize=14, fontweight='bold')
ax3.set_yticks(x_pos)
ax3.set_yticklabels(task_avg.index, fontsize=11)
ax3.legend(fontsize=11)
ax3.grid(axis='x', alpha=0.3)

plt.tight_layout()
individual_plot3_path = Path(csv_dir) / "task_type_analysis.png"
plt.savefig(individual_plot3_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"📁 Individual plot saved as: {individual_plot3_path}")
plt.close()

# Print summary statistics
print("\n" + "="*80)
print("PERFORMANCE ANALYSIS SUMMARY")
print("="*80)

print(f"\n📊 Overall Statistics:")
print(f"   • Average zero-shot performance: {df['zero-shot'].mean():.3f}")
print(f"   • Average few-shot performance: {df['few-shot'].mean():.3f}")
print(f"   • Average improvement: {df['few_vs_zero_pct'].mean():.1f}%")
print(f"   • Best improvement: {df['few_vs_zero_pct'].max():.1f}% ({df.loc[df['few_vs_zero_pct'].idxmax(), 'model_short']} on {df.loc[df['few_vs_zero_pct'].idxmax(), 'task_name']})")

print(f"\n🏆 Best Performing Models (by average few-shot score):")
best_models = df.groupby('model_short')['few-shot'].mean().sort_values(ascending=False)
for i, (model, score) in enumerate(best_models.head(3).items(), 1):
    print(f"   {i}. {model}: {score:.3f}")

print(f"\n📈 Highest Improvement Models (by average % gain):")
best_improvement = df.groupby('model_short')['few_vs_zero_pct'].mean().sort_values(ascending=False)
for i, (model, improvement) in enumerate(best_improvement.head(3).items(), 1):
    print(f"   {i}. {model}: {improvement:.1f}% improvement")

print(f"\n🎯 Task Type Analysis:")
task_analysis = df.groupby('task_type').agg({
    'few_vs_zero_pct': 'mean',
    'few-shot': 'mean',
    'zero-shot': 'mean'
}).round(3)
for task, stats in task_analysis.iterrows():
    print(f"   • {task}: {stats['few_vs_zero_pct']:.1f}% avg improvement (few-shot: {stats['few-shot']:.3f}, zero-shot: {stats['zero-shot']:.3f})")

# Cases where zero-shot outperformed few-shot
negative_cases = df[df['few_vs_zero_pct'] < 0]
if not negative_cases.empty:
    print(f"\n⚠️  Cases where zero-shot outperformed few-shot ({len(negative_cases)} cases):")
    for _, case in negative_cases.iterrows():
        print(f"   • {case['model_short']} on {case['task_name']}: {case['few_vs_zero_pct']:.1f}%")
else:
    print(f"\n✅ Few-shot learning improved performance in ALL cases!")