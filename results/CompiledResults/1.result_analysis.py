#!/usr/bin/env python3
"""
LM Eval Results Collector

This script collects and organizes results from multiple LM evaluation JSON files,
creating consolidated datasets for analysis and comparison.
"""

import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LMEvalResultsCollector:
    """Collects and processes LM evaluation results from multiple JSON files."""

    def __init__(self, results_dir: str = ".", output_dir: str = "consolidated_results"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Storage for collected data
        self.results_data = []
        self.model_configs = []
        self.task_configs = []

        # Track duplicates
        self.duplicate_results = []
        self.seen_results = set()  # Track unique model-task combinations

    def find_result_files(self) -> List[Path]:
        """Find all JSON result files in the results directory and subfolders."""
        json_files = list(self.results_dir.rglob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files in {self.results_dir}")
        return json_files

    def create_result_key(self, model_name: str, task_name: str, result_row: Dict[str, Any]) -> str:
        """Create a unique key for a result to detect duplicates."""
        # Use model name, task name, and key metrics to create unique identifier
        key_components = [model_name, task_name]

        # Add key metrics to the identifier
        for metric in ["acc,none", "acc_norm,none"]:
            if metric in result_row:
                key_components.append(f"{metric}:{result_row[metric]}")

        # Add sample size and n-shot info if available
        if "n_samples_original" in result_row and result_row["n_samples_original"]:
            key_components.append(f"samples:{result_row['n_samples_original']}")
        if "n_shot" in result_row and result_row["n_shot"]:
            key_components.append(f"nshot:{result_row['n_shot']}")

        return "|".join(str(c) for c in key_components)

    def is_duplicate_result(self, result_key: str, result_row: Dict[str, Any], file_path: Path) -> bool:
        """Check if a result is a duplicate and decide how to handle it."""
        if result_key in self.seen_results:
            # Find the existing result
            existing_result = None
            for existing in self.results_data:
                existing_key = self.create_result_key(existing["model_name"], existing["task_name"], existing)
                if existing_key == result_key:
                    existing_result = existing
                    break

            if existing_result:
                # Log the duplicate
                duplicate_info = {
                    "model_name": result_row["model_name"],
                    "task_name": result_row["task_name"],
                    "existing_file": existing_result["file_path"],
                    "duplicate_file": str(file_path),
                    "result_key": result_key
                }
                self.duplicate_results.append(duplicate_info)

                # Keep the newer result (based on date if available)
                existing_date = existing_result.get("date", 0)
                new_date = result_row.get("date", 0)

                if new_date and existing_date and new_date > existing_date:
                    # Replace older result with newer one
                    logger.info(
                        f"Replacing older result for {result_row['model_name']}/{result_row['task_name']} with newer version")
                    self.results_data.remove(existing_result)
                    return False  # Not a duplicate to skip
                else:
                    logger.info(f"Skipping duplicate result for {result_row['model_name']}/{result_row['task_name']}")
                    return True  # This is a duplicate to skip

    def extract_model_name(self, file_path: Path, data: Dict[str, Any]) -> str:
        """Extract model name from file path or data."""
        # Try to get from config first
        if "config" in data and "model_args" in data["config"]:
            model_args = data["config"]["model_args"]
            if "pretrained=" in model_args:
                return model_args.split("pretrained=")[1].split(",")[0]

        # Try to get from model_name
        if "model_name" in data:
            return data["model_name"]

        # Fallback to filename
        return file_path.stem

    def process_file(self, file_path: Path) -> bool:
        """Process a single JSON result file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            model_name = self.extract_model_name(file_path, data)
            logger.info(f"Processing {file_path.name} - Model: {model_name}")

            # Extract results for each task
            if "results" in data:
                for task_name, task_results in data["results"].items():
                    result_row = {
                        "model_name": model_name,
                        "task_name": task_name,
                        "file_path": str(file_path),
                        "date": data.get("date", None),
                        "evaluation_time": data.get("total_evaluation_time_seconds", None),
                    }

                    # Add all metrics
                    for metric_name, metric_value in task_results.items():
                        if metric_name != "alias":  # Skip alias field
                            result_row[metric_name] = metric_value

                    # Add sample information
                    if "n-samples" in data and task_name in data["n-samples"]:
                        result_row["n_samples_original"] = data["n-samples"][task_name].get("original", None)
                        result_row["n_samples_effective"] = data["n-samples"][task_name].get("effective", None)

                    # Add n-shot information
                    if "n-shot" in data and task_name in data["n-shot"]:
                        result_row["n_shot"] = data["n-shot"][task_name]

                    # Check for duplicates before adding
                    result_key = self.create_result_key(model_name, task_name, result_row)
                    if not self.is_duplicate_result(result_key, result_row, file_path):
                        self.results_data.append(result_row)

            # Extract model configuration
            if "config" in data:
                model_config = {
                    "model_name": model_name,
                    "file_path": str(file_path),
                    **data["config"]
                }
                self.model_configs.append(model_config)

            # Extract task configurations
            if "configs" in data:
                for task_name, task_config in data["configs"].items():
                    task_config_row = {
                        "model_name": model_name,
                        "task_name": task_name,
                        "file_path": str(file_path),
                        **task_config
                    }
                    self.task_configs.append(task_config_row)

            return True

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return False

    def collect_all_results(self) -> None:
        """Collect results from all JSON files."""
        result_files = self.find_result_files()

        successful = 0
        for file_path in result_files:
            if self.process_file(file_path):
                successful += 1

        logger.info(f"Successfully processed {successful}/{len(result_files)} files")
        logger.info(f"Collected {len(self.results_data)} result entries")

    def create_summary_table(self) -> pd.DataFrame:
        """Create a summary table with key metrics."""
        if not self.results_data:
            logger.warning("No results data available for summary")
            return pd.DataFrame()

        df = pd.DataFrame(self.results_data)

        # Create pivot table for main accuracy metrics
        metrics_of_interest = ["acc,none", "acc_norm,none"]
        available_metrics = [m for m in metrics_of_interest if m in df.columns]

        if available_metrics:
            summary = df.pivot_table(
                index=["model_name"],
                columns=["task_name"],
                values=available_metrics[0],  # Use first available metric
                aggfunc="first"
            )
            return summary

        return df

    def save_results(self) -> None:
        """Save all collected results to various formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        if self.results_data:
            results_df = pd.DataFrame(self.results_data)
            results_path = self.output_dir / f"detailed_results_{timestamp}.csv"
            results_df.to_csv(results_path, index=False)
            logger.info(f"Saved detailed results to {results_path}")

            # Save Excel version with multiple sheets
            excel_path = self.output_dir / f"lm_eval_results_{timestamp}.xlsx"
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='Detailed Results', index=False)

                # Create summary sheet
                summary_df = self.create_summary_table()
                if not summary_df.empty:
                    summary_df.to_excel(writer, sheet_name='Summary')

                # Model configs sheet
                if self.model_configs:
                    model_configs_df = pd.DataFrame(self.model_configs)
                    model_configs_df.to_excel(writer, sheet_name='Model Configs', index=False)

                # Task configs sheet (sample)
                if self.task_configs:
                    task_configs_df = pd.DataFrame(self.task_configs)
                    # Only include a sample due to potential size
                    sample_tasks = task_configs_df.head(100)
                    sample_tasks.to_excel(writer, sheet_name='Task Configs Sample', index=False)

                # Save duplicate information if any found
                if self.duplicate_results:
                    duplicates_df = pd.DataFrame(self.duplicate_results)
                    duplicates_df.to_excel(writer, sheet_name='Duplicates Found', index=False)

            logger.info(f"Saved Excel results to {excel_path}")

        # Save raw collected data as JSON for further processing
        collected_data = {
            "timestamp": timestamp,
            "results": self.results_data,
            "model_configs": self.model_configs,
            "task_configs": self.task_configs,
            "duplicates_found": self.duplicate_results
        }

        json_path = self.output_dir / f"collected_data_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(collected_data, f, indent=2, default=str)

        logger.info(f"Saved raw collected data to {json_path}")

        # Save duplicate report if any found
        if self.duplicate_results:
            duplicate_report_path = self.output_dir / f"duplicate_report_{timestamp}.csv"
            duplicates_df = pd.DataFrame(self.duplicate_results)
            duplicates_df.to_csv(duplicate_report_path, index=False)
            logger.info(f"Saved duplicate report to {duplicate_report_path}")

    def print_summary(self) -> None:
        """Print a summary of collected results."""
        if not self.results_data:
            print("No results data collected.")
            return

        df = pd.DataFrame(self.results_data)

        print("\n" + "=" * 60)
        print("LM EVALUATION RESULTS SUMMARY")
        print("=" * 60)

        print(f"Total result entries: {len(self.results_data)}")
        print(f"Unique models: {df['model_name'].nunique()}")
        print(f"Unique tasks: {df['task_name'].nunique()}")

        print("\nModels found:")
        for model in sorted(df['model_name'].unique()):
            model_tasks = df[df['model_name'] == model]['task_name'].nunique()
            print(f"  - {model} ({model_tasks} tasks)")

        print("\nTasks found:")
        for task in sorted(df['task_name'].unique()):
            task_models = df[df['task_name'] == task]['model_name'].nunique()
            print(f"  - {task} ({task_models} models)")

        # Show sample of best performing results
        if "acc,none" in df.columns:
            print("\nTop 10 Results by Accuracy:")
            top_results = df.nlargest(10, "acc,none")[["model_name", "task_name", "acc,none"]]
            print(top_results.to_string(index=False))


def main():
    """Main function to run the collector."""
    parser = argparse.ArgumentParser(description="Collect and organize LM evaluation results")

    # Change the default results_dir here
    default_results_dir = r"D:\Masters In Germany\Computer Science\Semester 4\Practical_NLP\Babelnet_Client\results"
    default_output_dir = r"D:\Masters In Germany\Computer Science\Semester 4\Practical_NLP\Babelnet_Client\results\CompiledResults\consolidated_results"

    parser.add_argument("--results_dir", "-r", default=default_results_dir,
                        help="Directory containing JSON result files")
    parser.add_argument("--output_dir", "-o", default=default_output_dir,
                        help="Output directory for consolidated results")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Run in quiet mode (less output)")

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Create collector and run
    collector = LMEvalResultsCollector(args.results_dir, args.output_dir)

    print("Starting LM Eval Results Collection...")
    collector.collect_all_results()
    collector.save_results()
    collector.print_summary()

    print(f"\nResults saved to: {collector.output_dir}")
    print("Collection complete!")



if __name__ == "__main__":
    main()