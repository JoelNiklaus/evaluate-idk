#!/usr/bin/env python3
"""
Analysis script to examine answer endings from evaluation details files.

This script samples answers from different categories:
- IDK score = 1 (model answered correctly)
- IDK score = -1 (model answered when it should have said "I don't know")
- IDK score = 0 (model appropriately said "I don't know")
- Extract fail = 1 (failed to extract an answer)
"""

import pandas as pd
from pathlib import Path
import tiktoken
import argparse


def find_all_parquet_files(base_dir="results/details", benchmark_filter=None, model_filter=None):
    """Find all parquet files in the details directory.
    
    Args:
        base_dir: Base directory to search
        benchmark_filter: Optional benchmark name to filter by (e.g., "gpqa-diamond-idk", "lexam-en-idk")
        model_filter: Optional model name to filter by (e.g., "gemini-2.5-flash")
    """
    base_path = Path(base_dir)
    all_files = list(base_path.rglob("*.parquet"))
    
    if benchmark_filter:
        # Filter files that contain the benchmark name in their filename
        all_files = [f for f in all_files if benchmark_filter in f.name]
    
    if model_filter:
        # Filter files that contain the model name in their path
        all_files = [f for f in all_files if model_filter in str(f)]
    
    return all_files


def load_all_data(parquet_files):
    """Load all parquet files and combine them."""
    all_data = []
    
    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            # Extract relevant information
            for idx, row in df.iterrows():
                answer_text = row['model_response']['text']
                # Handle both list and string formats
                if isinstance(answer_text, list):
                    answer_text = answer_text[0] if answer_text else ""
                
                all_data.append({
                    'file': str(file),
                    'answer': answer_text,
                    'idk_score': row['metric']['idk_score'],
                    'extract_fail': row['metric']['extract_fail'],
                    'trad_score': row['metric']['trad_score'],
                    'idk_freq': row['metric']['idk_freq']
                })
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return pd.DataFrame(all_data)


def sample_and_display(df, category_name, condition, n_samples=3):
    """Sample and display answers from a specific category."""
    filtered = df[condition]
    
    print(f"\n{'='*80}")
    print(f"{category_name}")
    print(f"{'='*80}")
    print(f"Total count: {len(filtered)}")
    
    if len(filtered) == 0:
        print("No samples found in this category.")
        return
    
    # Initialize GPT-5 tokenizer (uses o200k_base encoding)
    tokenizer = tiktoken.get_encoding("o200k_base")
    
    # Show average answer length for this category
    avg_length = filtered['answer'].apply(lambda x: len(tokenizer.encode(str(x)))).mean()
    print(f"Average answer length: {avg_length:.1f} tokens")
    
    # Sample (or take all if less than n_samples)
    sample_size = min(n_samples, len(filtered))
    samples = filtered.sample(n=sample_size, random_state=42)
    
    for i, (_, row) in enumerate(samples.iterrows(), 1):
        print(f"\n{'-'*80}")
        print(f"Sample {i}/{sample_size}")
        print(f"{'-'*80}")
        print(f"File: {Path(row['file']).parent.parent.name}/{Path(row['file']).parent.name}/{Path(row['file']).name}")
        print(f"IDK Score: {row['idk_score']}")
        print(f"Extract Fail: {row['extract_fail']}")
        print(f"Trad Score: {row['trad_score']}")
        print(f"IDK Freq: {row['idk_freq']}")
        answer = str(row['answer'])
        answer_tokens = tokenizer.encode(answer)
        print(f"Answer length: {len(answer_tokens)} tokens")
        print(f"\nLast 100 tokens of answer:")
        print(f"{'─'*80}")
        last_100_tokens = answer_tokens[-100:] if len(answer_tokens) > 100 else answer_tokens
        last_text = tokenizer.decode(last_100_tokens)
        if len(answer_tokens) > 100:
            print(f"...{last_text}")
        else:
            print(last_text)
        print(f"{'─'*80}")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(
        description="Analyze answer patterns from evaluation results"
    )
    parser.add_argument(
        "model",
        type=str,
        help="Filter results by model name (e.g., 'gemini-2.5-flash', 'claude-sonnet-4.5')"
    )
    parser.add_argument(
        "benchmark",
        type=str,
        nargs="?",
        default="lexam",
        choices=["gpqa", "lexam"],
        help="Filter results by benchmark name: 'gpqa' or 'lexam' (default: lexam)"
    )
    args = parser.parse_args()
    
    # Map short benchmark names to full names
    benchmark_map = {
        "gpqa": "gpqa-diamond-idk",
        "lexam": "lexam-en-idk"
    }
    full_benchmark_name = benchmark_map[args.benchmark]
    
    print(f"Filtering for benchmark: {full_benchmark_name}")
    print(f"Filtering for model: {args.model}")
    
    print("\nFinding all parquet files...")
    parquet_files = find_all_parquet_files(
        benchmark_filter=full_benchmark_name,
        model_filter=args.model
    )
    print(f"Found {len(parquet_files)} parquet files")
    
    if len(parquet_files) == 0:
        print("\nNo parquet files found.")
        print(f"Try checking if the benchmark name '{full_benchmark_name}' and model name '{args.model}' are correct.")
        return
    
    print("\nLoading data from all files...")
    df = load_all_data(parquet_files)
    print(f"Loaded {len(df)} total samples")
    
    # Compute token lengths for all answers
    print("\nComputing token lengths...")
    tokenizer = tiktoken.get_encoding("o200k_base") # GPT-5 tokenizer
    df['answer_tokens'] = df['answer'].apply(lambda x: len(tokenizer.encode(str(x))))
    
    # Sample from each category
    sample_and_display(
        df,
        "CATEGORY 1: IDK Score = 1 (Model answered correctly)",
        df['idk_score'] == 1,
        n_samples=2
    )

    sample_and_display(
        df,
        "CATEGORY 2: IDK Score = -1 (Model answered incorrectly)",
        df['idk_score'] == -1,
        n_samples=2
    )

    sample_and_display(
        df,
        "CATEGORY 1: IDK Score = 0 (Model appropriately said 'I don't know')",
        df['idk_score'] == 0,
        n_samples=10
    )
    
    sample_and_display(
        df,
        "CATEGORY 3: Extract Fail = 1 (Failed to extract answer)",
        df['extract_fail'] == 1,
        n_samples=10
    )
    
    # Display overall statistics and answer length summary
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print(f"\nIDK Score distribution:")
    print(df['idk_score'].value_counts().sort_index())
    print(f"\nExtract Fail distribution:")
    print(df['extract_fail'].value_counts().sort_index())
    
    # Answer length statistics table
    print("\n" + "="*80)
    print("ANSWER LENGTH STATISTICS (in tokens)")
    print("="*80)
    print()
    
    # Prepare data for table
    categories = [
        ("IDK Score = 1", df['idk_score'] == 1),
        ("IDK Score = -1", df['idk_score'] == -1),
        ("IDK Score = 0", df['idk_score'] == 0),
        ("Extract Fail = 1", df['extract_fail'] == 1),
    ]
    
    table_rows = []
    for cat_name, condition in categories:
        filtered = df[condition]
        if len(filtered) > 0:
            mean = filtered['answer_tokens'].mean()
            std = filtered['answer_tokens'].std()
            count = len(filtered)
            table_rows.append((cat_name, str(count), f"{mean:.1f} ± {std:.1f}"))
        else:
            table_rows.append((cat_name, "0", "N/A"))
    
    # Calculate column widths
    col1_width = max(len("Category"), max(len(row[0]) for row in table_rows))
    col2_width = max(len("Count"), max(len(row[1]) for row in table_rows))
    col3_width = max(len("Mean ± Std"), max(len(row[2]) for row in table_rows))
    
    # Print header
    print(f"| {'Category'.ljust(col1_width)} | {'Count'.ljust(col2_width)} | {'Mean ± Std'.ljust(col3_width)} |")
    print(f"|{'-' * (col1_width + 2)}|{'-' * (col2_width + 2)}|{'-' * (col3_width + 2)}|")
    
    # Print rows
    for cat_name, count, mean_std in table_rows:
        print(f"| {cat_name.ljust(col1_width)} | {count.ljust(col2_width)} | {mean_std.ljust(col3_width)} |")


if __name__ == "__main__":
    main()

