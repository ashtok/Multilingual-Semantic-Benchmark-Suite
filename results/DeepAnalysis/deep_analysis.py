import json
import pandas as pd
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


class MultiModelMultilingualAnalyzer:
    def __init__(self, base_results_path):
        """
        Initialize analyzer with path to base results directory containing model subdirectories

        Args:
            base_results_path (str): Path to directory containing model subdirectories with JSONL files
        """
        self.base_path = Path(base_results_path)
        self.models_data = {}
        self.models_df = {}
        self.combined_df = None

    def discover_models(self):
        """Discover all model directories in the base path"""
        model_dirs = [d for d in self.base_path.iterdir() if d.is_dir()]
        model_names = [d.name for d in model_dirs]
        print(f"Discovered {len(model_names)} model directories: {model_names}")
        return model_names

    def load_model_results(self, model_name):
        """Load all JSONL files for a specific model"""
        model_path = self.base_path / model_name
        jsonl_files = list(model_path.glob("*.jsonl"))

        if not jsonl_files:
            print(f"No .jsonl files found for model {model_name}")
            return []

        print(f"Loading {len(jsonl_files)} JSONL files for {model_name}")

        model_data = []
        for file_path in jsonl_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            data = json.loads(line.strip())
                            # Add model and source file information
                            data['model_name'] = model_name
                            data['source_file'] = file_path.name
                            model_data.append(data)
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line {line_num} in {file_path.name}: {e}")
            except Exception as e:
                print(f"Error reading {file_path.name}: {e}")

        print(f"Loaded {len(model_data)} records for {model_name}")
        return model_data

    def load_all_models(self):
        """Load data for all discovered models"""
        model_names = self.discover_models()

        for model_name in model_names:
            self.models_data[model_name] = self.load_model_results(model_name)

        # Create combined dataset
        all_data = []
        for model_name, model_data in self.models_data.items():
            all_data.extend(model_data)

        print(f"Total records across all models: {len(all_data)}")
        return all_data

    def process_model_data(self, model_name, model_data):
        """Process data for a specific model into DataFrame"""
        processed_records = []

        for record in model_data:
            # Extract basic information
            doc_info = record.get('doc', {})
            metadata = doc_info.get('metadata', {})

            # Create processed record
            processed_record = {
                'model_name': model_name,
                'doc_id': record.get('doc_id'),
                'prompt_id': doc_info.get('id'),
                'accuracy': record.get('acc', 0),
                'accuracy_norm': record.get('acc_norm', 0),
                'source_file': record.get('source_file'),

                # Metadata fields
                'difficulty': metadata.get('difficulty'),
                'distractor_type': metadata.get('distractor_type'),
                'from_lang': metadata.get('from_lang'),
                'to_lang': metadata.get('to_lang'),
                'relation_type': metadata.get('relation_type'),
                'resource_pair': metadata.get('resource_pair'),
                'prompt_lang': metadata.get('prompt_lang'),
                'multilingual_mode': metadata.get('multilingual_mode'),
                'generation_time': metadata.get('generation_time'),
                'synset_id': metadata.get('synset_id'),

                # Derived fields
                'language_pair': f"{metadata.get('from_lang', 'unknown')}_to_{metadata.get('to_lang', 'unknown')}",
                'answer_index': doc_info.get('answer_index'),
                'target': record.get('target'),
                'num_options': len(doc_info.get('options', [])) if doc_info.get('options') else 0,

                # Extract task type from filename
                'task_type': self._extract_task_type(record.get('source_file', '')),
                'resource_level': self._extract_resource_level(record.get('source_file', ''))
            }

            processed_records.append(processed_record)

        return pd.DataFrame(processed_records)

    def _extract_task_type(self, filename):
        """Extract task type from filename (e.g., 'analogies', 'gloss', 'hypernymy', 'meronymy')"""
        if 'analogies' in filename:
            return 'analogies'
        elif 'gloss' in filename:
            return 'gloss'
        elif 'hypernymy' in filename:
            return 'hypernymy'
        elif 'meronymy' in filename:
            return 'meronymy'
        return 'unknown'

    def _extract_resource_level(self, filename):
        """Extract resource level from filename (e.g., 'high', 'medium', 'low', 'all', 'mono')"""
        for level in ['high', 'medium', 'low', 'all', 'mono']:
            if level in filename:
                return level
        return 'unknown'

    def process_all_models(self):
        """Process data for all models"""
        for model_name, model_data in self.models_data.items():
            if model_data:
                self.models_df[model_name] = self.process_model_data(model_name, model_data)
                print(f"Processed {len(self.models_df[model_name])} records for {model_name}")

        # Create combined DataFrame
        if self.models_df:
            combined_dfs = list(self.models_df.values())
            self.combined_df = pd.concat(combined_dfs, ignore_index=True)
            print(f"Combined DataFrame: {len(self.combined_df)} total records")

    def analyze_model_performance(self, model_name):
        """Analyze performance for a specific model"""
        if model_name not in self.models_df:
            print(f"No data found for model {model_name}")
            return None

        df = self.models_df[model_name]
        results = {}

        # Overall stats
        results['overall'] = {
            'total_records': len(df),
            'accuracy_mean': df['accuracy'].mean(),
            'accuracy_std': df['accuracy'].std(),
            'accuracy_norm_mean': df['accuracy_norm'].mean(),
            'accuracy_norm_std': df['accuracy_norm'].std()
        }

        # By language pair
        lang_pair_stats = df.groupby('language_pair').agg({
            'accuracy': ['count', 'mean', 'std'],
            'accuracy_norm': ['mean', 'std']
        }).round(4)
        lang_pair_stats.columns = ['count', 'acc_mean', 'acc_std', 'acc_norm_mean', 'acc_norm_std']
        results['by_language_pair'] = lang_pair_stats.reset_index()

        # By difficulty
        if 'difficulty' in df.columns and not df['difficulty'].isna().all():
            difficulty_stats = df.groupby('difficulty').agg({
                'accuracy': ['count', 'mean', 'std'],
                'accuracy_norm': ['mean', 'std']
            }).round(4)
            difficulty_stats.columns = ['count', 'acc_mean', 'acc_std', 'acc_norm_mean', 'acc_norm_std']
            results['by_difficulty'] = difficulty_stats.reset_index()

        # By task type
        if 'task_type' in df.columns:
            task_stats = df.groupby('task_type').agg({
                'accuracy': ['count', 'mean', 'std'],
                'accuracy_norm': ['mean', 'std']
            }).round(4)
            task_stats.columns = ['count', 'acc_mean', 'acc_std', 'acc_norm_mean', 'acc_norm_std']
            results['by_task_type'] = task_stats.reset_index()

        # By resource level
        if 'resource_level' in df.columns:
            resource_stats = df.groupby('resource_level').agg({
                'accuracy': ['count', 'mean', 'std'],
                'accuracy_norm': ['mean', 'std']
            }).round(4)
            resource_stats.columns = ['count', 'acc_mean', 'acc_std', 'acc_norm_mean', 'acc_norm_std']
            results['by_resource_level'] = resource_stats.reset_index()

        return results

    def compare_models(self):
        """Compare performance across all models"""
        if not self.combined_df is not None:
            print("No combined data available")
            return None

        # Overall comparison
        model_comparison = self.combined_df.groupby('model_name').agg({
            'accuracy': ['count', 'mean', 'std'],
            'accuracy_norm': ['mean', 'std']
        }).round(4)
        model_comparison.columns = ['total_samples', 'acc_mean', 'acc_std', 'acc_norm_mean', 'acc_norm_std']

        # By task type
        task_comparison = self.combined_df.groupby(['model_name', 'task_type'])['accuracy'].agg(
            ['count', 'mean']).reset_index()
        task_pivot = task_comparison.pivot(index='model_name', columns='task_type', values='mean').round(4)

        # By difficulty
        if 'difficulty' in self.combined_df.columns:
            difficulty_comparison = self.combined_df.groupby(['model_name', 'difficulty'])['accuracy'].agg(
                ['count', 'mean']).reset_index()
            difficulty_pivot = difficulty_comparison.pivot(index='model_name', columns='difficulty',
                                                           values='mean').round(4)
        else:
            difficulty_pivot = None

        return {
            'overall_comparison': model_comparison.reset_index(),
            'by_task_type': task_pivot,
            'by_difficulty': difficulty_pivot
        }

    def save_model_results(self, model_name, output_base_folder='model_analysis'):
        """Save analysis results for a specific model"""
        output_path = Path(output_base_folder) / model_name
        output_path.mkdir(parents=True, exist_ok=True)

        if model_name not in self.models_df:
            print(f"No data found for model {model_name}")
            return

        # Save raw data
        self.models_df[model_name].to_csv(output_path / f'{model_name}_raw_data.csv', index=False)

        # Save analysis results
        results = self.analyze_model_performance(model_name)
        if results:
            # Save overall stats
            pd.DataFrame([results['overall']]).to_csv(output_path / f'{model_name}_overall_stats.csv', index=False)

            # Save detailed analyses
            for analysis_name, df in results.items():
                if analysis_name != 'overall' and df is not None:
                    filename = f'{model_name}_{analysis_name}.csv'
                    if hasattr(df, 'to_csv'):
                        df.to_csv(output_path / filename, index=False)

        print(f"Saved results for {model_name} to {output_path}")

    def save_all_model_results(self, output_base_folder='model_analysis'):
        """Save results for all models"""
        for model_name in self.models_df.keys():
            self.save_model_results(model_name, output_base_folder)

        # Save model comparison
        if self.combined_df is not None:
            comparison_results = self.compare_models()
            if comparison_results:
                output_path = Path(output_base_folder) / 'model_comparisons'
                output_path.mkdir(parents=True, exist_ok=True)

                for comp_name, df in comparison_results.items():
                    if df is not None:
                        filename = f'models_{comp_name}.csv'
                        if hasattr(df, 'to_csv'):
                            df.to_csv(output_path / filename, index=False)
                        else:
                            df.reset_index().to_csv(output_path / filename, index=False)

                print(f"Saved model comparisons to {output_path}")

    def create_comparison_visualizations(self, output_folder='model_analysis'):
        """Create visualizations comparing models"""
        if self.combined_df is None:
            print("No combined data for visualizations")
            return

        output_path = Path(output_folder) / 'visualizations'
        output_path.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. Overall model comparison
        model_stats = self.combined_df.groupby('model_name')['accuracy'].agg(['mean', 'count']).reset_index()

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(model_stats)), model_stats['mean'])
        plt.xticks(range(len(model_stats)), model_stats['model_name'], rotation=45, ha='right')
        plt.ylabel('Mean Accuracy')
        plt.title('Overall Model Performance Comparison')
        plt.tight_layout()
        plt.savefig(output_path / 'model_comparison_overall.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Performance by task type
        if 'task_type' in self.combined_df.columns:
            task_stats = self.combined_df.groupby(['model_name', 'task_type'])['accuracy'].mean().unstack()

            plt.figure(figsize=(12, 8))
            task_stats.plot(kind='bar', ax=plt.gca())
            plt.title('Model Performance by Task Type')
            plt.xlabel('Model')
            plt.ylabel('Mean Accuracy')
            plt.legend(title='Task Type')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_path / 'model_comparison_by_task.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 3. Performance by difficulty (if available)
        if 'difficulty' in self.combined_df.columns and not self.combined_df['difficulty'].isna().all():
            difficulty_stats = self.combined_df.groupby(['model_name', 'difficulty'])['accuracy'].mean().unstack()

            plt.figure(figsize=(12, 8))
            difficulty_stats.plot(kind='bar', ax=plt.gca())
            plt.title('Model Performance by Difficulty Level')
            plt.xlabel('Model')
            plt.ylabel('Mean Accuracy')
            plt.legend(title='Difficulty')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_path / 'model_comparison_by_difficulty.png', dpi=300, bbox_inches='tight')
            plt.close()

        print(f"Saved comparison visualizations to {output_path}")

    def print_model_summary(self, model_name):
        """Print summary for a specific model"""
        if model_name not in self.models_df:
            print(f"No data found for model {model_name}")
            return

        df = self.models_df[model_name]
        print(f"\n{'=' * 60}")
        print(f"SUMMARY FOR {model_name.upper()}")
        print(f"{'=' * 60}")

        print(f"Total Records: {len(df):,}")
        print(f"Overall Accuracy: {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
        print(f"Normalized Accuracy: {df['accuracy_norm'].mean():.4f} ± {df['accuracy_norm'].std():.4f}")

        # Task type breakdown
        if 'task_type' in df.columns:
            task_stats = df.groupby('task_type')['accuracy'].agg(['count', 'mean']).round(4)
            print(f"\nPerformance by Task Type:")
            print(task_stats.to_string())

        # Difficulty breakdown
        if 'difficulty' in df.columns and not df['difficulty'].isna().all():
            difficulty_stats = df.groupby('difficulty')['accuracy'].agg(['count', 'mean']).round(4)
            print(f"\nPerformance by Difficulty:")
            print(difficulty_stats.to_string())

    def print_comparison_summary(self):
        """Print comparison summary across all models"""
        if self.combined_df is None:
            print("No combined data available for comparison")
            return

        print(f"\n{'=' * 60}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'=' * 60}")

        comparison = self.compare_models()
        if comparison and 'overall_comparison' in comparison:
            print("\nOverall Performance:")
            overall = comparison['overall_comparison'][['model_name', 'total_samples', 'acc_mean', 'acc_std']]
            overall = overall.sort_values('acc_mean', ascending=False)
            print(overall.to_string(index=False))


# Usage example and main function
def main():
    # Initialize analyzer with base results directory
    analyzer = MultiModelMultilingualAnalyzer(
        r'D:\Masters In Germany\Computer Science\Semester 4\Practical_NLP\Babelnet_Client\results\DeepAnalysis')

    # Load and process all models
    analyzer.load_all_models()
    analyzer.process_all_models()

    # Save individual model results
    analyzer.save_all_model_results('detailed_model_analysis')

    # Create comparison visualizations
    analyzer.create_comparison_visualizations('detailed_model_analysis')

    # Print summaries
    for model_name in analyzer.models_df.keys():
        analyzer.print_model_summary(model_name)

    analyzer.print_comparison_summary()

    print(f"\nAnalysis complete! Results saved in 'detailed_model_analysis' folder.")


if __name__ == "__main__":
    main()