import os
import json
import pandas as pd
from glob import glob

class DeepAnalyzer:
    def __init__(self, base_path):
        self.base_path = base_path
        self.models_data = {}
        self.combined_df = pd.DataFrame()

    def load_jsonl_files(self, model_name):
        """Load all JSONL records for a given model."""
        model_path = os.path.join(self.base_path, model_name)
        jsonl_files = glob(os.path.join(model_path, "*.jsonl"))
        all_records = []

        print(f"Loading {len(jsonl_files)} JSONL files for {model_name}")
        for file_path in jsonl_files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        record["model_name"] = model_name
                        all_records.append(record)
                    except json.JSONDecodeError:
                        continue
        print(f"Loaded {len(all_records)} records for {model_name}")
        return all_records

    def process_model_data(self, model_name, records):
        """Process model records into DataFrame with only required fields."""
        processed = []

        for r in records:
            doc = r.get("doc", {})
            metadata = doc.get("metadata", {})
            answer_index = doc.get("answer_index", None)
            filtered_resps = r.get("filtered_resps", [])

            # Predicted index from lowest score
            predicted_index = None
            if isinstance(filtered_resps, list) and filtered_resps:
                try:
                    predicted_index = filtered_resps.index(min(filtered_resps, key=lambda x: float(x[0])))
                except (ValueError, TypeError):
                    pass

            # Accuracy
            acc = 1 if predicted_index == answer_index else 0

            processed.append({
                "model_name": model_name,
                "acc": acc,
                "difficulty": metadata.get("difficulty"),
                "language_pair": f"{metadata.get('from_lang', 'unknown')}_{metadata.get('to_lang', 'unknown')}",
                "resource_level": metadata.get("resource_pair"),
                "task_type": metadata.get("relation_type"),
            })

        return pd.DataFrame(processed)

    def compare_models(self):
        """Cross-model comparison based on mean accuracy."""
        return self.combined_df.groupby("model_name").agg({
            "acc": "mean",
            "difficulty": lambda x: x.mode()[0] if not x.mode().empty else None,
            "language_pair": lambda x: x.mode()[0] if not x.mode().empty else None,
            "resource_level": lambda x: x.mode()[0] if not x.mode().empty else None,
            "task_type": lambda x: x.mode()[0] if not x.mode().empty else None
        }).reset_index()

    def save_all_model_results(self, output_dir):
        """Save per-model CSVs and comparison CSV."""
        os.makedirs(output_dir, exist_ok=True)

        for model_name, df in self.models_data.items():
            model_path = os.path.join(output_dir, model_name)
            os.makedirs(model_path, exist_ok=True)
            df.to_csv(os.path.join(model_path, f"{model_name}_metrics.csv"), index=False)
            print(f"Saved metrics CSV for {model_name} to {model_path}")

        comp_df = self.compare_models()
        comp_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
        print(f"Saved cross-model comparison to {os.path.join(output_dir, 'model_comparison.csv')}")

def main():
    base_path = "results"  # Change to your JSONL results directory
    analyzer = DeepAnalyzer(base_path)

    # Discover models
    model_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    print(f"Discovered {len(model_dirs)} model directories: {model_dirs}")

    # Process each model
    for model_name in model_dirs:
        records = analyzer.load_jsonl_files(model_name)
        df = analyzer.process_model_data(model_name, records)
        analyzer.models_data[model_name] = df
        print(f"Processed {len(df)} records for {model_name}")

    # Combine all models into one DF
    analyzer.combined_df = pd.concat(analyzer.models_data.values(), ignore_index=True)
    print(f"Combined DataFrame: {len(analyzer.combined_df)} total records")

    # Save outputs
    analyzer.save_all_model_results("detailed_model_analysis")

if __name__ == "__main__":
    main()
