import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Base path to your detailed_model_analysis directory
# IMPORTANT: Update this path to your specific directory
BASE_DIR = r"D:\Masters In Germany\Computer Science\Semester 4\Practical_NLP\Babelnet_Client\results\DeepAnalysis\detailed_model_analysis"


def generate_language_pair_plots(base_path):
    """
    Finds and aggregates language pair data from individual model CSVs,
    then plots the top 10 and top 20 performing pairs.
    """
    plots_dir = os.path.join(base_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Saving plots to: {plots_dir}")

    all_data = []

    # Iterate through each folder to find model-specific data
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)

        # Skip non-directories and the 'model_comparisons' folder
        if not os.path.isdir(folder_path) or folder == "model_comparisons":
            continue

        model_name = folder
        print(f"Processing data for model: {model_name}")

        # Construct the file name based on the model folder name
        # We're looking for a file like 'model_name_by_language_pair.csv'
        lang_pair_file = os.path.join(folder_path, f"{model_name}_by_language_pair.csv")

        if os.path.exists(lang_pair_file):
            try:
                df_model = pd.read_csv(lang_pair_file, low_memory=False)

                # Ensure required columns are present
                if 'language_pair' in df_model.columns and 'acc_mean' in df_model.columns:
                    # Add a column for the model name to track the origin of the data
                    df_model['model_name'] = model_name
                    all_data.append(df_model[['model_name', 'language_pair', 'acc_mean']])
                else:
                    print(f"Warning: 'language_pair' or 'acc_mean' not found in {lang_pair_file}. Skipping.")
            except Exception as e:
                print(f"Error reading {lang_pair_file}: {e}")
        else:
            print(f"Warning: Language pair file not found for model: {model_name}. Skipping.")

    if not all_data:
        print("\nCould not find any valid language pair data to plot. Please check your file structure.")
        return

    # Concatenate all data into a single DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)

    # Calculate the average accuracy for each language pair across all models
    lang_pair_acc = combined_df.groupby('language_pair')['acc_mean'].mean().reset_index()
    lang_pair_acc_sorted = lang_pair_acc.sort_values(by='acc_mean', ascending=False)

    print("\n--- Generating Plots for Top Language Pairs ---")

    # Plot Top 10 language pairs
    top_10_df = lang_pair_acc_sorted.head(10)
    plt.figure(figsize=(12, 7))
    sns.barplot(x='language_pair', y='acc_mean', data=top_10_df, hue='language_pair', legend=False)
    plt.title("Top 10 Language Pairs by Average Accuracy Across All Models")
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
    sns.barplot(x='language_pair', y='acc_mean', data=top_20_df, hue='language_pair', legend=False)
    plt.title("Top 20 Language Pairs by Average Accuracy Across All Models")
    plt.xlabel("Language Pair")
    plt.ylabel("Average Accuracy Mean")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "top_20_language_pairs_barplot.png"))
    plt.close()
    print("Generated plot for top 20 language pairs.")


if __name__ == "__main__":
    generate_language_pair_plots(BASE_DIR)