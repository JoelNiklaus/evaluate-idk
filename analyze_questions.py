#!/usr/bin/env python3
"""
Analysis script to examine question performance across models.

This script identifies which questions are most challenging by looking at
which questions appear across different score categories (idk_score 1, -1, 0, extract_fail)
when evaluated by different models.
"""

import pandas as pd
from pathlib import Path
import argparse


def find_all_parquet_files(base_dir="results/details", benchmark_filter=None):
    """Find all parquet files in the details directory."""
    base_path = Path(base_dir)
    all_files = list(base_path.rglob("*.parquet"))
    
    if benchmark_filter:
        all_files = [f for f in all_files if benchmark_filter in f.name]
    
    return all_files


def extract_model_name(file_path):
    """Extract model name from file path."""
    # Path structure: results/details/openrouter/provider/model/timestamp/file.parquet
    parts = Path(file_path).parts
    if len(parts) >= 5:
        # Get provider/model
        provider = parts[-4]
        model = parts[-3]
        return f"{provider}/{model}"
    return "unknown"


def load_all_data(parquet_files):
    """Load all parquet files and combine them with model information."""
    all_data = []
    
    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            model_name = extract_model_name(file)
            
            # Extract relevant information
            for idx, row in df.iterrows():
                question_id = row['doc'].get('id', 'unknown')
                task_name = row['doc'].get('task_name', 'unknown')
                
                all_data.append({
                    'question_id': question_id,
                    'task_name': task_name,
                    'model': model_name,
                    'idk_score': row['metric']['idk_score'],
                    'extract_fail': row['metric']['extract_fail'],
                    'trad_score': row['metric']['trad_score'],
                    'idk_freq': row['metric']['idk_freq'],
                    'query': row['doc'].get('query', '')[:500],  # First 500 chars for display
                    'full_query': row['doc'].get('query', ''),
                    'model_response': row.get('model_response', ''),
                    'choices': str(row['doc'].get('choices', [])),
                    'gold_index': row['doc'].get('gold_index', -1)
                })
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return pd.DataFrame(all_data)


def analyze_question_diversity(df):
    """Analyze which questions have diverse outcomes across models."""
    
    # Group by question and count unique scores
    question_stats = []
    
    for question_id in df['question_id'].unique():
        q_data = df[df['question_id'] == question_id]
        
        # Count how many models scored it in each category
        idk_1_count = len(q_data[q_data['idk_score'] == 1])
        idk_minus1_count = len(q_data[q_data['idk_score'] == -1])
        idk_0_count = len(q_data[q_data['idk_score'] == 0])
        extract_fail_count = len(q_data[q_data['extract_fail'] == 1])
        
        total_models = len(q_data)
        
        # Calculate "diversity score" - how many different categories appear
        categories_present = sum([
            idk_1_count > 0,
            idk_minus1_count > 0,
            idk_0_count > 0,
            extract_fail_count > 0
        ])
        
        # Check if ALL models have the same outcome
        all_correct = (idk_1_count == total_models)
        all_wrong = (idk_minus1_count == total_models)
        all_said_idk = (idk_0_count == total_models)
        all_extract_fail = (extract_fail_count == total_models)
        
        question_stats.append({
            'question_id': question_id,
            'task_name': q_data['task_name'].iloc[0],
            'total_evaluations': total_models,
            'idk_1_count': idk_1_count,
            'idk_minus1_count': idk_minus1_count,
            'idk_0_count': idk_0_count,
            'extract_fail_count': extract_fail_count,
            'categories_present': categories_present,
            'all_correct': all_correct,
            'all_wrong': all_wrong,
            'all_said_idk': all_said_idk,
            'all_extract_fail': all_extract_fail,
            'query_snippet': q_data['query'].iloc[0][:200],
            'full_query': q_data['query'].iloc[0]
        })
    
    return pd.DataFrame(question_stats)


def print_unanimous_overview(question_stats):
    """Print overview of questions where ALL models agreed on the outcome."""
    print(f"\n{'='*100}")
    print(f"📊 UNANIMOUS RESULTS OVERVIEW (All Models Agreed)")
    print(f"{'='*100}\n")
    
    all_correct = question_stats[question_stats['all_correct'] == True]
    all_wrong = question_stats[question_stats['all_wrong'] == True]
    all_said_idk = question_stats[question_stats['all_said_idk'] == True]
    all_extract_fail = question_stats[question_stats['all_extract_fail'] == True]
    
    total_questions = len(question_stats)
    
    print(f"✅ Questions where ALL models answered CORRECTLY:           {len(all_correct):>4} ({len(all_correct)/total_questions*100:5.1f}%)")
    print(f"❌ Questions where ALL models answered WRONG:               {len(all_wrong):>4} ({len(all_wrong)/total_questions*100:5.1f}%)")
    print(f"🤷 Questions where ALL models said 'I DON'T KNOW':          {len(all_said_idk):>4} ({len(all_said_idk)/total_questions*100:5.1f}%)")
    print(f"⚠️  Questions where ALL models FAILED TO EXTRACT:           {len(all_extract_fail):>4} ({len(all_extract_fail)/total_questions*100:5.1f}%)")
    print(f"\n{'─'*100}")
    total_unanimous = len(all_correct) + len(all_wrong) + len(all_said_idk) + len(all_extract_fail)
    mixed_results = total_questions - total_unanimous
    print(f"Total unanimous questions: {total_unanimous} ({total_unanimous/total_questions*100:5.1f}%)")
    print(f"Questions with mixed results: {mixed_results} ({mixed_results/total_questions*100:5.1f}%)")
    
    # Show one random sample from each category
    import random
    random.seed(42)
    
    categories = [
        ("✅ ALL MODELS CORRECT", all_correct),
        ("❌ ALL MODELS WRONG", all_wrong),
        ("🤷 ALL MODELS SAID 'I DON'T KNOW'", all_said_idk),
        ("⚠️  ALL MODELS FAILED TO EXTRACT", all_extract_fail),
    ]
    
    for title, df_cat in categories:
        print(f"\n{'='*100}")
        print(f"{title}")
        print(f"{'='*100}\n")
        
        if len(df_cat) == 0:
            print("No questions in this category.\n")
            continue
        
        # Pick random sample
        sample = df_cat.sample(n=1, random_state=random.randint(0, 10000)).iloc[0]
        
        print(f"Question ID: {sample['question_id']}")
        print(f"Total evaluations: {sample['total_evaluations']} models")
        print(f"\nQuestion:")
        print(f"{'─'*100}")
        # Print first 600 characters of the query
        query_text = sample['full_query']
        if len(query_text) > 600:
            print(f"{query_text[:600]}...")
        else:
            print(query_text)
        print(f"{'─'*100}\n")


def print_top_questions(question_stats, title, sort_by, n=10):
    """Print top N questions based on specified criteria."""
    print(f"\n{'='*100}")
    print(f"{title}")
    print(f"{'='*100}\n")
    
    sorted_stats = question_stats.sort_values(sort_by, ascending=False).head(n)
    
    for idx, row in sorted_stats.iterrows():
        print(f"Question ID: {row['question_id']} | Task: {row['task_name']}")
        print(f"  Total evaluations: {row['total_evaluations']}")
        print(f"  IDK=1 (correct): {row['idk_1_count']} | "
              f"IDK=-1 (wrong): {row['idk_minus1_count']} | "
              f"IDK=0 (said IDK): {row['idk_0_count']} | "
              f"Extract fail: {row['extract_fail_count']}")
        print(f"  Diversity: {row['categories_present']}/4 categories present")
        print(f"  Query: {row['query_snippet']}...")
        print()


def print_question_detail(df, question_id):
    """Print detailed breakdown for a specific question across all models."""
    q_data = df[df['question_id'] == question_id].sort_values('idk_score', ascending=False)
    
    if len(q_data) == 0:
        print(f"No data found for question {question_id}")
        return
    
    print(f"\n{'='*100}")
    print(f"DETAILED BREAKDOWN: Question {question_id}")
    print(f"{'='*100}\n")
    
    print(f"Task: {q_data['task_name'].iloc[0]}\n")
    print(f"Query snippet:\n{q_data['query'].iloc[0][:300]}...\n")
    print(f"\n{'-'*100}")
    print(f"{'Model':<40} | {'IDK Score':<10} | {'Extract Fail':<12} | {'Trad Score':<10}")
    print(f"{'-'*100}")
    
    for _, row in q_data.iterrows():
        status = ""
        if row['extract_fail'] == 1:
            status = "EXTRACT FAIL"
        elif row['idk_score'] == 1:
            status = "✓ Correct"
        elif row['idk_score'] == -1:
            status = "✗ Wrong"
        elif row['idk_score'] == 0:
            status = "Said IDK"
        
        print(f"{row['model']:<40} | {row['idk_score']:<10.1f} | {row['extract_fail']:<12.1f} | {row['trad_score']:<10.1f} | {status}")


def extract_text_from_response(model_response):
    """Extract the actual text from model_response dictionary."""
    if isinstance(model_response, dict):
        # Try to get text_post_processed first
        if 'text_post_processed' in model_response:
            text_array = model_response['text_post_processed']
            # Handle numpy array or list
            if hasattr(text_array, '__iter__') and len(text_array) > 0:
                return str(text_array[0])
        # Fallback to 'text' field
        if 'text' in model_response:
            text_array = model_response['text']
            if hasattr(text_array, '__iter__') and len(text_array) > 0:
                return str(text_array[0])
    # If all else fails, return string representation
    return str(model_response)


def export_all_wrong_questions(df, question_stats, output_file="all_wrong_questions.csv"):
    """Export questions where all models answered incorrectly to CSV with each model's response."""
    # Filter to questions where all models were wrong
    all_wrong_questions = question_stats[question_stats['all_wrong'] == True]
    
    if len(all_wrong_questions) == 0:
        print("\nNo questions where all models answered incorrectly.")
        return
    
    print(f"\nExporting {len(all_wrong_questions)} questions where all models were wrong...")
    
    # For each question, create a row with model responses as columns
    export_data = []
    
    for _, q_stat in all_wrong_questions.iterrows():
        question_id = q_stat['question_id']
        q_data = df[df['question_id'] == question_id]
        
        # Convert gold_index to letter (0->A, 1->B, 2->C, 3->D, 4->E)
        gold_index = q_data['gold_index'].iloc[0]
        correct_letter = chr(65 + int(gold_index)) if gold_index >= 0 else 'Unknown'
        
        # Base information
        row_data = {
            'question_id': question_id,
            'task_name': q_data['task_name'].iloc[0],
            'question': q_data['full_query'].iloc[0],
            'choices': q_data['choices'].iloc[0],
            'correct': correct_letter
        }
        
        # Add each model's response as a column (extract just the text)
        for _, model_row in q_data.iterrows():
            model_name = model_row['model']
            # Clean model name for column header (replace / with _)
            column_name = f"{model_name.replace('/', '_')}"
            # Extract the actual text from the response
            row_data[column_name] = extract_text_from_response(model_row['model_response'])
        
        export_data.append(row_data)
    
    # Create DataFrame and export to CSV
    export_df = pd.DataFrame(export_data)
    export_df.to_csv(output_file, index=False)
    
    print(f"✅ Exported to: {output_file}")
    print(f"   - {len(export_df)} questions")
    print(f"   - {len([col for col in export_df.columns if col not in ['question_id', 'task_name', 'question', 'choices', 'correct']])} model responses per question")
    
    return export_df


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(
        description="Analyze question performance across all models"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        choices=["gpqa", "lexam"],
        help="Filter by benchmark: 'gpqa' or 'lexam' (default: all benchmarks)"
    )
    parser.add_argument(
        "--question-id",
        type=str,
        default=None,
        help="Show detailed breakdown for specific question ID"
    )
    parser.add_argument(
        "--export-all-wrong",
        type=str,
        default=None,
        metavar="FILENAME",
        help="Export questions where all models were wrong to CSV file"
    )
    
    args = parser.parse_args()
    
    # Map short benchmark names to full names
    benchmark_filter = None
    if args.benchmark:
        benchmark_map = {
            "gpqa": "gpqa-diamond-idk",
            "lexam": "lexam-en-idk"
        }
        benchmark_filter = benchmark_map[args.benchmark]
        print(f"Filtering for benchmark: {benchmark_filter}")
    else:
        print("Analyzing all benchmarks")
    
    print("\nFinding all parquet files...")
    parquet_files = find_all_parquet_files(benchmark_filter=benchmark_filter)
    print(f"Found {len(parquet_files)} parquet files")
    
    if len(parquet_files) == 0:
        print("\nNo parquet files found.")
        return
    
    print("\nLoading data from all files...")
    df = load_all_data(parquet_files)
    print(f"Loaded {len(df)} total evaluations")
    print(f"Unique questions: {df['question_id'].nunique()}")
    print(f"Unique models: {df['model'].nunique()}")
    
    # If specific question requested, show detail and exit
    if args.question_id:
        print_question_detail(df, args.question_id)
        return
    
    # Overall statistics
    print("\n" + "="*100)
    print("OVERALL STATISTICS")
    print("="*100)
    print(f"\nTotal questions: {df['question_id'].nunique()}")
    print(f"Total models: {df['model'].nunique()}")
    print(f"Total evaluations: {len(df)}")
    
    print("\nModels evaluated:")
    for model in sorted(df['model'].unique()):
        count = len(df[df['model'] == model])
        print(f"  - {model}: {count} evaluations")
    
    print("\n" + "-"*100)
    print("Score distribution across ALL evaluations:")
    print("-"*100)
    print(f"IDK Score = 1 (correct):     {len(df[df['idk_score'] == 1]):>5} ({len(df[df['idk_score'] == 1])/len(df)*100:5.1f}%)")
    print(f"IDK Score = -1 (wrong):      {len(df[df['idk_score'] == -1]):>5} ({len(df[df['idk_score'] == -1])/len(df)*100:5.1f}%)")
    print(f"IDK Score = 0 (said IDK):    {len(df[df['idk_score'] == 0]):>5} ({len(df[df['idk_score'] == 0])/len(df)*100:5.1f}%)")
    print(f"Extract Fail = 1:            {len(df[df['extract_fail'] == 1]):>5} ({len(df[df['extract_fail'] == 1])/len(df)*100:5.1f}%)")
    
    # Analyze question diversity
    print("\nAnalyzing question-level statistics...")
    question_stats = analyze_question_diversity(df)
    
    # Print various rankings
    print_top_questions(
        question_stats,
        "🏆 TOP 10 MOST DIVERSE QUESTIONS (appear in most different score categories)",
        'categories_present',
        n=10
    )
    
    print_top_questions(
        question_stats,
        "❌ TOP 10 QUESTIONS WITH MOST WRONG ANSWERS (IDK=-1 across models)",
        'idk_minus1_count',
        n=10
    )
    
    print_top_questions(
        question_stats,
        "🤷 TOP 10 QUESTIONS WHERE MODELS SAID 'I DON'T KNOW' (IDK=0)",
        'idk_0_count',
        n=10
    )
    
    print_top_questions(
        question_stats,
        "⚠️ TOP 10 QUESTIONS WITH MOST EXTRACT FAILURES",
        'extract_fail_count',
        n=10
    )
    
    print_top_questions(
        question_stats,
        "✅ TOP 10 QUESTIONS WITH MOST CORRECT ANSWERS (IDK=1)",
        'idk_1_count',
        n=10
    )
    
    # Summary statistics
    print("\n" + "="*100)
    print("QUESTION-LEVEL SUMMARY")
    print("="*100)
    
    print(f"\nQuestions appearing in all 4 categories: {len(question_stats[question_stats['categories_present'] == 4])}")
    print(f"Questions appearing in 3 categories: {len(question_stats[question_stats['categories_present'] == 3])}")
    print(f"Questions appearing in 2 categories: {len(question_stats[question_stats['categories_present'] == 2])}")
    print(f"Questions appearing in 1 category: {len(question_stats[question_stats['categories_present'] == 1])}")
    
    # Print unanimous results overview at the end
    print_unanimous_overview(question_stats)
    
    # Export all wrong questions if requested
    if args.export_all_wrong:
        export_all_wrong_questions(df, question_stats, args.export_all_wrong)
    
    print("\n" + "="*100)
    print("💡 TIP: Use --question-id <ID> to see detailed breakdown for a specific question")
    print("💡 TIP: Use --export-all-wrong <filename.csv> to export questions where all models were wrong")
    print("="*100)


if __name__ == "__main__":
    main()




