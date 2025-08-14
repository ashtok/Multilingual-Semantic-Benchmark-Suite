import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class ModelAnalysisVisualizer:
    def __init__(self, analysis_folder='detailed_model_analysis'):
        """
        Initialize visualizer with path to analysis results folder

        Args:
            analysis_folder (str): Path to folder containing model analysis results
        """
        self.base_path = Path(analysis_folder)
        self.model_data = {}
        self.comparison_data = {}

    def discover_models(self):
        """Discover all model directories"""
        model_dirs = [d for d in self.base_path.iterdir() if
                      d.is_dir() and d.name != 'model_comparisons' and d.name != 'visualizations']
        return [d.name for d in model_dirs]

    def load_model_data(self, model_name):
        """Load all CSV files for a specific model"""
        model_path = self.base_path / model_name
        if not model_path.exists():
            print(f"Model directory {model_name} not found")
            return {}

        data = {}
        csv_files = list(model_path.glob("*.csv"))

        for csv_file in csv_files:
            try:
                # Extract analysis type from filename
                analysis_type = csv_file.name.replace(f'{model_name}_', '').replace('.csv', '')
                data[analysis_type] = pd.read_csv(csv_file)
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")

        return data

    def load_comparison_data(self):
        """Load model comparison data"""
        comparison_path = self.base_path / 'model_comparisons'
        if not comparison_path.exists():
            print("Model comparisons folder not found")
            return {}

        data = {}
        csv_files = list(comparison_path.glob("*.csv"))

        for csv_file in csv_files:
            try:
                analysis_type = csv_file.name.replace('models_', '').replace('.csv', '')
                data[analysis_type] = pd.read_csv(csv_file)
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")

        return data

    def load_all_data(self):
        """Load data for all models and comparisons"""
        models = self.discover_models()
        print(f"Loading data for models: {models}")

        for model in models:
            self.model_data[model] = self.load_model_data(model)

        self.comparison_data = self.load_comparison_data()
        print("Data loading complete")

    def create_model_performance_heatmap(self, output_folder='advanced_visualizations'):
        """Create heatmap showing model performance across different dimensions"""
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)

        # Create overall performance heatmap
        if 'overall_comparison' in self.comparison_data:
            overall_df = self.comparison_data['overall_comparison']

            # Create a matrix for heatmap
            metrics = ['acc_mean', 'acc_norm_mean']
            heatmap_data = overall_df.set_index('model_name')[metrics].T

            plt.figure(figsize=(12, 6))
            sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', fmt='.4f',
                        cbar_kws={'label': 'Accuracy Score'})
            plt.title('Model Performance Heatmap - Overall Metrics')
            plt.xlabel('Models')
            plt.ylabel('Metrics')
            plt.tight_layout()
            plt.savefig(output_path / 'model_performance_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()

    def create_task_performance_heatmap(self, output_folder='advanced_visualizations'):
        """Create heatmap showing model performance by task type"""
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)

        if 'by_task_type' in self.comparison_data:
            task_df = self.comparison_data['by_task_type']

            plt.figure(figsize=(10, 8))
            sns.heatmap(task_df, annot=True, cmap='RdYlGn', fmt='.4f',
                        cbar_kws={'label': 'Mean Accuracy'})
            plt.title('Model Performance by Task Type')
            plt.xlabel('Task Type')
            plt.ylabel('Model')
            plt.tight_layout()
            plt.savefig(output_path / 'task_performance_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()

    def create_difficulty_performance_heatmap(self, output_folder='advanced_visualizations'):
        """Create heatmap showing model performance by difficulty"""
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)

        if 'by_difficulty' in self.comparison_data:
            difficulty_df = self.comparison_data['by_difficulty']

            plt.figure(figsize=(8, 8))
            sns.heatmap(difficulty_df, annot=True, cmap='RdYlGn', fmt='.4f',
                        cbar_kws={'label': 'Mean Accuracy'})
            plt.title('Model Performance by Difficulty Level')
            plt.xlabel('Difficulty Level')
            plt.ylabel('Model')
            plt.tight_layout()
            plt.savefig(output_path / 'difficulty_performance_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()

    def create_language_pair_analysis(self, output_folder='advanced_visualizations'):
        """Create visualizations for language pair performance"""
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)

        # Collect language pair data from all models
        all_lang_pairs = {}
        for model_name, model_data in self.model_data.items():
            if 'by_language_pair' in model_data:
                lang_df = model_data['by_language_pair']
                # Only include pairs with sufficient data
                lang_df_filtered = lang_df[lang_df['count'] >= 5].copy()
                all_lang_pairs[model_name] = lang_df_filtered.set_index('language_pair')['acc_mean']

        if all_lang_pairs:
            # Create combined DataFrame
            combined_lang_df = pd.DataFrame(all_lang_pairs)

            # Get top 20 language pairs by average performance
            avg_performance = combined_lang_df.mean(axis=1).sort_values(ascending=False)
            top_pairs = avg_performance.head(20).index

            # Create heatmap for top language pairs
            plt.figure(figsize=(12, 16))
            sns.heatmap(combined_lang_df.loc[top_pairs].fillna(0),
                        annot=True, cmap='RdYlGn', fmt='.3f',
                        cbar_kws={'label': 'Mean Accuracy'})
            plt.title('Top 20 Language Pairs Performance Across Models')
            plt.xlabel('Model')
            plt.ylabel('Language Pair')
            plt.tight_layout()
            plt.savefig(output_path / 'language_pairs_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Create bar plot for best performing language pairs
            plt.figure(figsize=(14, 8))
            avg_performance.head(15).plot(kind='bar')
            plt.title('Top 15 Language Pairs by Average Performance')
            plt.xlabel('Language Pair')
            plt.ylabel('Average Accuracy')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_path / 'top_language_pairs_bar.png', dpi=300, bbox_inches='tight')
            plt.close()

    def create_resource_level_analysis(self, output_folder='advanced_visualizations'):
        """Create analysis for resource level performance"""
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)

        # Collect resource level data
        resource_data = {}
        for model_name, model_data in self.model_data.items():
            if 'by_resource_level' in model_data:
                resource_df = model_data['by_resource_level']
                resource_data[model_name] = resource_df.set_index('resource_level')['acc_mean']

        if resource_data:
            resource_combined = pd.DataFrame(resource_data)

            # Create grouped bar plot
            plt.figure(figsize=(12, 8))
            resource_combined.T.plot(kind='bar', width=0.8)
            plt.title('Model Performance by Resource Level')
            plt.xlabel('Model')
            plt.ylabel('Mean Accuracy')
            plt.legend(title='Resource Level')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_path / 'resource_level_performance.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Create heatmap
            plt.figure(figsize=(8, 10))
            sns.heatmap(resource_combined.T, annot=True, cmap='RdYlGn', fmt='.4f',
                        cbar_kws={'label': 'Mean Accuracy'})
            plt.title('Resource Level Performance Heatmap')
            plt.xlabel('Resource Level')
            plt.ylabel('Model')
            plt.tight_layout()
            plt.savefig(output_path / 'resource_level_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()

    def create_comprehensive_dashboard(self, output_folder='advanced_visualizations'):
        """Create a comprehensive multi-plot dashboard"""
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))

        # 1. Overall performance comparison (top left)
        if 'overall_comparison' in self.comparison_data:
            ax1 = plt.subplot(2, 3, 1)
            overall_df = self.comparison_data['overall_comparison']
            bars = plt.bar(overall_df['model_name'], overall_df['acc_mean'])
            plt.title('Overall Model Performance')
            plt.ylabel('Mean Accuracy')
            plt.xticks(rotation=45, ha='right')

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.3f}', ha='center', va='bottom')

        # 2. Task type performance (top middle)
        if 'by_task_type' in self.comparison_data:
            ax2 = plt.subplot(2, 3, 2)
            task_df = self.comparison_data['by_task_type']
            sns.heatmap(task_df, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax2)
            plt.title('Performance by Task Type')

        # 3. Sample count distribution (top right)
        ax3 = plt.subplot(2, 3, 3)
        if 'overall_comparison' in self.comparison_data:
            overall_df = self.comparison_data['overall_comparison']
            plt.pie(overall_df['total_samples'], labels=overall_df['model_name'], autopct='%1.1f%%')
            plt.title('Sample Distribution Across Models')

        # 4. Difficulty performance (bottom left)
        if 'by_difficulty' in self.comparison_data:
            ax4 = plt.subplot(2, 3, 4)
            difficulty_df = self.comparison_data['by_difficulty']
            sns.heatmap(difficulty_df, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax4)
            plt.title('Performance by Difficulty')

        # 5. Performance variance (bottom middle)
        ax5 = plt.subplot(2, 3, 5)
        if 'overall_comparison' in self.comparison_data:
            overall_df = self.comparison_data['overall_comparison']
            plt.errorbar(range(len(overall_df)), overall_df['acc_mean'],
                         yerr=overall_df['acc_std'], fmt='o', capsize=5)
            plt.xticks(range(len(overall_df)), overall_df['model_name'], rotation=45, ha='right')
            plt.title('Performance with Standard Deviation')
            plt.ylabel('Mean Accuracy ± Std')

        # 6. Top language pairs summary (bottom right)
        ax6 = plt.subplot(2, 3, 6)
        # Get average performance across top language pairs
        if self.model_data:
            top_lang_pairs = []
            for model_name, model_data in self.model_data.items():
                if 'by_language_pair' in model_data:
                    lang_df = model_data['by_language_pair']
                    lang_df_filtered = lang_df[lang_df['count'] >= 10]
                    if not lang_df_filtered.empty:
                        top_pair = lang_df_filtered.loc[lang_df_filtered['acc_mean'].idxmax()]
                        top_lang_pairs.append({
                            'model': model_name,
                            'best_pair': top_pair['language_pair'],
                            'accuracy': top_pair['acc_mean']
                        })

            if top_lang_pairs:
                top_df = pd.DataFrame(top_lang_pairs)
                bars = plt.bar(top_df['model'], top_df['accuracy'])
                plt.title('Best Language Pair per Model')
                plt.ylabel('Best Accuracy')
                plt.xticks(rotation=45, ha='right')

                # Add best pair labels
                for i, (bar, pair) in enumerate(zip(bars, top_df['best_pair'])):
                    plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                             pair.replace('_to_', '→'), ha='center', va='bottom',
                             rotation=45, fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path / 'comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_statistical_analysis(self, output_folder='advanced_visualizations'):
        """Create statistical analysis plots"""
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)

        if 'overall_comparison' not in self.comparison_data:
            return

        overall_df = self.comparison_data['overall_comparison']

        # 1. Performance distribution
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.scatter(overall_df['total_samples'], overall_df['acc_mean'], s=100, alpha=0.7)
        for i, model in enumerate(overall_df['model_name']):
            plt.annotate(model, (overall_df.iloc[i]['total_samples'], overall_df.iloc[i]['acc_mean']),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)
        plt.xlabel('Total Samples')
        plt.ylabel('Mean Accuracy')
        plt.title('Accuracy vs Sample Size')

        plt.subplot(1, 3, 2)
        plt.scatter(overall_df['acc_std'], overall_df['acc_mean'], s=100, alpha=0.7)
        for i, model in enumerate(overall_df['model_name']):
            plt.annotate(model, (overall_df.iloc[i]['acc_std'], overall_df.iloc[i]['acc_mean']),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)
        plt.xlabel('Standard Deviation')
        plt.ylabel('Mean Accuracy')
        plt.title('Accuracy vs Consistency')

        plt.subplot(1, 3, 3)
        # Efficiency score (accuracy / std deviation)
        efficiency = overall_df['acc_mean'] / overall_df['acc_std']
        plt.bar(overall_df['model_name'], efficiency)
        plt.ylabel('Efficiency Score (Accuracy/Std)')
        plt.title('Model Efficiency')
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(output_path / 'statistical_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_summary_report(self, output_folder='advanced_visualizations'):
        """Generate a summary report with key insights"""
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)

        report = []
        report.append("MULTILINGUAL MODEL ANALYSIS REPORT")
        report.append("=" * 50)
        report.append("")

        if 'overall_comparison' in self.comparison_data:
            overall_df = self.comparison_data['overall_comparison']

            # Best performing model
            best_model = overall_df.loc[overall_df['acc_mean'].idxmax()]
            report.append(f"BEST PERFORMING MODEL: {best_model['model_name']}")
            report.append(f"  - Mean Accuracy: {best_model['acc_mean']:.4f}")
            report.append(f"  - Standard Deviation: {best_model['acc_std']:.4f}")
            report.append(f"  - Total Samples: {best_model['total_samples']:,}")
            report.append("")

            # Most consistent model
            most_consistent = overall_df.loc[overall_df['acc_std'].idxmin()]
            report.append(f"MOST CONSISTENT MODEL: {most_consistent['model_name']}")
            report.append(f"  - Standard Deviation: {most_consistent['acc_std']:.4f}")
            report.append(f"  - Mean Accuracy: {most_consistent['acc_mean']:.4f}")
            report.append("")

        # Task-specific insights
        if 'by_task_type' in self.comparison_data:
            task_df = self.comparison_data['by_task_type']
            report.append("TASK-SPECIFIC PERFORMANCE:")
            for task in task_df.columns:
                best_model_for_task = task_df[task].idxmax()
                best_score = task_df[task].max()
                report.append(f"  - Best at {task}: {best_model_for_task} ({best_score:.4f})")
            report.append("")

        # Difficulty insights
        if 'by_difficulty' in self.comparison_data:
            difficulty_df = self.comparison_data['by_difficulty']
            report.append("DIFFICULTY-SPECIFIC PERFORMANCE:")
            for difficulty in difficulty_df.columns:
                best_model_for_diff = difficulty_df[difficulty].idxmax()
                best_score = difficulty_df[difficulty].max()
                report.append(f"  - Best at difficulty {difficulty}: {best_model_for_diff} ({best_score:.4f})")
            report.append("")

        # Save report
        with open(output_path / 'analysis_summary_report.txt', 'w') as f:
            f.write('\n'.join(report))

        # Also print to console
        print('\n'.join(report))

    def create_all_visualizations(self, output_folder='advanced_visualizations'):
        """Create all visualizations and reports"""
        print("Creating comprehensive visualizations...")

        self.create_model_performance_heatmap(output_folder)
        print("✓ Model performance heatmap created")

        self.create_task_performance_heatmap(output_folder)
        print("✓ Task performance heatmap created")

        self.create_difficulty_performance_heatmap(output_folder)
        print("✓ Difficulty performance heatmap created")

        self.create_language_pair_analysis(output_folder)
        print("✓ Language pair analysis created")

        self.create_resource_level_analysis(output_folder)
        print("✓ Resource level analysis created")

        self.create_comprehensive_dashboard(output_folder)
        print("✓ Comprehensive dashboard created")

        self.create_statistical_analysis(output_folder)
        print("✓ Statistical analysis created")

        self.generate_summary_report(output_folder)
        print("✓ Summary report generated")

        print(f"\nAll visualizations saved to '{output_folder}' folder!")


# Usage example
def main():
    # Initialize visualizer
    visualizer = ModelAnalysisVisualizer('detailed_model_analysis')

    # Load all data
    visualizer.load_all_data()

    # Create all visualizations
    visualizer.create_all_visualizations('advanced_visualizations')


if __name__ == "__main__":
    main()