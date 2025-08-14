import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Base path to your detailed_model_analysis directory
# IMPORTANT: Update this path to your specific directory
BASE_DIR = r"D:\Masters In Germany\Computer Science\Semester 4\Practical_NLP\Babelnet_Client\results\DeepAnalysis\detailed_model_analysis"


def generate_plots(base_path):
    """
    Iterates through model analysis folders, reads the CSV files,
    and generates bar plots and heatmaps for each model's 'acc_mean'.
    """
    # Create a directory to save the plots if it doesn't exist
    plots_dir = os.path.join(base_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Saving plots to: {plots_dir}")

    # Files to be ignored
    ignored_files = ['_metrics.csv', '_stats.csv', '_raw_data.csv']

    # Handle special 'model_comparisons' directory first for heatmaps
    model_comparisons_dir = os.path.join(base_path, "model_comparisons")
    if os.path.isdir(model_comparisons_dir):
        print("\n--- Generating Model Comparison Heatmaps ---")
        try:
            # Read the overall comparison file to get model names
            overall_path = os.path.join(model_comparisons_dir, "models_overall_comparison.csv")
            if os.path.exists(overall_path):
                models_df = pd.read_csv(overall_path, low_memory=False)
                model_names = models_df['model_name'].tolist()
            else:
                print(f"Warning: Could not find '{overall_path}' to get model names.")
                model_names = []

            # Process 'models_by_difficulty'
            difficulty_path = os.path.join(model_comparisons_dir, "models_by_difficulty.csv")
            if os.path.exists(difficulty_path) and model_names:
                df = pd.read_csv(difficulty_path, low_memory=False)
                df.index = model_names

                plt.figure(figsize=(10, 8))
                # Use 'Blues' colormap for the heatmap
                sns.heatmap(df, annot=True, fmt=".2f", cmap="Blues")
                plt.title("Model Accuracy by Difficulty")
                plt.xlabel("Difficulty Level")
                plt.ylabel("Model Name")
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, "model_comparison_heatmap_difficulty.png"))
                plt.close()
                print("Generated heatmap for models by difficulty.")

            # Process 'models_by_task_type'
            task_type_path = os.path.join(model_comparisons_dir, "models_by_task_type.csv")
            if os.path.exists(task_type_path) and model_names:
                df = pd.read_csv(task_type_path, low_memory=False)
                df.index = model_names

                plt.figure(figsize=(10, 8))
                # Use 'Blues' colormap for the heatmap
                sns.heatmap(df, annot=True, fmt=".2f", cmap="Blues")
                plt.title("Model Accuracy by Task Type")
                plt.xlabel("Task Type")
                plt.ylabel("Model Name")
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, "model_comparison_heatmap_task_type.png"))
                plt.close()
                print("Generated heatmap for models by task type.")

        except Exception as e:
            print(f"Error generating comparison plots: {e}")

    # --- NEW: Generate plots for top 10 and top 20 language pairs across all models ---
    print("\n--- Generating Plots for Top Language Pairs ---")
    try:
        # Assuming language pair data is available in the overall comparison file
        lang_pair_path = os.path.join(base_path, "model_comparisons", "models_overall_comparison.csv")
        if os.path.exists(lang_pair_path):
            df_overall = pd.read_csv(lang_pair_path, low_memory=False)

            if 'language_pair' in df_overall.columns and 'acc_mean' in df_overall.columns:
                # Group by language_pair and calculate the mean accuracy across all models
                lang_pair_acc = df_overall.groupby('language_pair')['acc_mean'].mean().reset_index()
                lang_pair_acc_sorted = lang_pair_acc.sort_values(by='acc_mean', ascending=False)

                # Plot Top 10 language pairs
                top_10_df = lang_pair_acc_sorted.head(10)
                plt.figure(figsize=(12, 7))
                # Use default seaborn colors for bar plots
                sns.barplot(x='language_pair', y='acc_mean', data=top_10_df, hue='language_pair', legend=False)
                plt.title("Top 10 Language Pairs by Average Accuracy")
                plt.xlabel("Language Pair")
                plt.ylabel("Average Accuracy Mean")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, "top_10_language_pairs_barplot.png"))
                plt.close()
                print("Generated plot for top 10 language pairs.")

                # Plot Top 20 language pairs
                top_20_df = lang_pair_acc_sorted.head(20)
                plt.figure(figsize=(15, 8))
                # Use default seaborn colors for bar plots
                sns.barplot(x='language_pair', y='acc_mean', data=top_20_df, hue='language_pair', legend=False)
                plt.title("Top 20 Language Pairs by Average Accuracy")
                plt.xlabel("Language Pair")
                plt.ylabel("Average Accuracy Mean")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, "top_20_language_pairs_barplot.png"))
                plt.close()
                print("Generated plot for top 20 language pairs.")
            else:
                print("Warning: 'language_pair' or 'acc_mean' column not found in overall comparison data.")
        else:
            print("Warning: 'models_overall_comparison.csv' not found for language pair analysis.")
    except Exception as e:
        print(f"Error generating language pair plots: {e}")
    # Now, loop through each model's individual analysis folder
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)

        # Skip non-directories and the comparison folder
        if not os.path.isdir(folder_path) or folder == "model_comparisons":
            continue

        model_name = folder
        print(f"\n--- Processing model: {model_name} ---")

        # Loop through files inside the model folder
        for file in os.listdir(folder_path):
            if file.lower().endswith(".csv") and not any(f in file for f in ignored_files):
                file_path = os.path.join(folder_path, file)

                try:
                    df = pd.read_csv(file_path, low_memory=False)

                    # Ensure 'acc_mean' is in the DataFrame
                    if 'acc_mean' not in df.columns:
                        print(f"Skipping {file}: 'acc_mean' column not found.")
                        continue

                    # The first column is the categorical variable for the bar plot
                    category_col = df.columns[0]

                    # Create a bar plot
                    plt.figure(figsize=(12, 7))
                    # Use the default seaborn color scheme
                    sns.barplot(x=category_col, y='acc_mean', data=df, hue=category_col, legend=False)
                    plt.title(f"{model_name} Accuracy by {category_col.replace('_', ' ').title()}")
                    plt.xlabel(category_col.replace('_', ' ').title())
                    plt.ylabel("Accuracy Mean")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()

                    # Save the plot with a descriptive filename
                    plot_filename = f"{model_name}_{category_col}_barplot.png"
                    plt.savefig(os.path.join(plots_dir, plot_filename))
                    plt.close()
                    print(f"Generated bar plot for {file}.")

                except Exception as e:
                    print(f"Error reading or plotting {file_path}: {e}")


if __name__ == "__main__":
    generate_plots(BASE_DIR)
