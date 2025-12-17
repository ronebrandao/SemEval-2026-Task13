from datasets import load_dataset
import pandas as pd
import numpy as np
from collections import Counter

# Load dataset
print("=" * 80)
print("DATASET ANALYSIS - SemEval 2026 Task 13")
print("=" * 80)

print("\n>>> Loading dataset...")
ds = load_dataset("DaniilOr/SemEval-2026-Task13", "A")
print(f"Dataset loaded. Available splits: {list(ds.keys())}")

train_ds = ds["train"]
val_ds = ds["validation"]
test_ds = ds["test"]

# Convert to pandas for easier analysis
train_df = pd.DataFrame(train_ds)
val_df = pd.DataFrame(val_ds)
test_df = pd.DataFrame(test_ds)

print("\n" + "=" * 80)
print("1. DATASET SIZES")
print("=" * 80)
print(f"Training set:   {len(train_df):,} examples")
print(f"Validation set: {len(val_df):,} examples")
print(f"Test set:       {len(test_df):,} examples")
print(f"Total:          {len(train_df) + len(val_df) + len(test_df):,} examples")

print("\n" + "=" * 80)
print("2. LABEL DISTRIBUTION")
print("=" * 80)

def print_label_distribution(df, split_name):
    label_counts = Counter(df["label"])
    total = len(df)
    print(f"\n{split_name}:")
    print(f"  Human (0): {label_counts[0]:,} ({label_counts[0]/total*100:.2f}%)")
    print(f"  LLM (1):   {label_counts[1]:,} ({label_counts[1]/total*100:.2f}%)")
    print(f"  Total:     {total:,}")

print_label_distribution(train_df, "Training")
print_label_distribution(val_df, "Validation")
print_label_distribution(test_df, "Test")

# Combined train+val distribution
combined_df = pd.concat([train_df, val_df], ignore_index=True)
print_label_distribution(combined_df, "Training + Validation (combined)")

print("\n" + "=" * 80)
print("3. LANGUAGE DISTRIBUTION")
print("=" * 80)

def print_language_distribution(df, split_name):
    lang_counts = Counter(df["language"])
    total = len(df)
    print(f"\n{split_name}:")
    for lang, count in lang_counts.most_common():
        print(f"  {lang}: {count:,} ({count/total*100:.2f}%)")

if "language" in train_df.columns:
    print_language_distribution(train_df, "Training")
    print_language_distribution(val_df, "Validation")
    print_language_distribution(test_df, "Test")
else:
    print("Language column not found in dataset")

print("\n" + "=" * 80)
print("4. GENERATOR DISTRIBUTION")
print("=" * 80)

def print_generator_distribution(df, split_name):
    gen_counts = Counter(df["generator"])
    total = len(df)
    print(f"\n{split_name}:")
    for gen, count in gen_counts.most_common():
        print(f"  {gen}: {count:,} ({count/total*100:.2f}%)")

if "generator" in train_df.columns:
    print_generator_distribution(train_df, "Training")
    print_generator_distribution(val_df, "Validation")
    print_generator_distribution(test_df, "Test")
else:
    print("Generator column not found in dataset")

print("\n" + "=" * 80)
print("5. CODE LENGTH STATISTICS")
print("=" * 80)

def print_code_stats(df, split_name):
    code_lengths = [len(str(code)) for code in df["code"]]
    print(f"\n{split_name}:")
    print(f"  Mean length:   {np.mean(code_lengths):.1f} characters")
    print(f"  Median length: {np.median(code_lengths):.1f} characters")
    print(f"  Min length:    {np.min(code_lengths):,} characters")
    print(f"  Max length:    {np.max(code_lengths):,} characters")
    print(f"  Std dev:       {np.std(code_lengths):.1f} characters")
    print(f"  25th percentile: {np.percentile(code_lengths, 25):.1f} characters")
    print(f"  75th percentile: {np.percentile(code_lengths, 75):.1f} characters")
    print(f"  90th percentile: {np.percentile(code_lengths, 90):.1f} characters")
    print(f"  95th percentile: {np.percentile(code_lengths, 95):.1f} characters")
    print(f"  99th percentile: {np.percentile(code_lengths, 99):.1f} characters")

print_code_stats(train_df, "Training")
print_code_stats(val_df, "Validation")
print_code_stats(test_df, "Test")

print("\n" + "=" * 80)
print("6. LABEL DISTRIBUTION BY LANGUAGE")
print("=" * 80)

if "language" in train_df.columns:
    def print_label_by_language(df, split_name):
        print(f"\n{split_name}:")
        for lang in sorted(df["language"].unique()):
            lang_df = df[df["language"] == lang]
            label_counts = Counter(lang_df["label"])
            total = len(lang_df)
            print(f"  {lang} (n={total:,}):")
            print(f"    Human (0): {label_counts[0]:,} ({label_counts[0]/total*100:.2f}%)")
            print(f"    LLM (1):   {label_counts[1]:,} ({label_counts[1]/total*100:.2f}%)")
    
    print_label_by_language(train_df, "Training")
    print_label_by_language(val_df, "Validation")

print("\n" + "=" * 80)
print("7. LABEL DISTRIBUTION BY GENERATOR")
print("=" * 80)

if "generator" in train_df.columns:
    def print_label_by_generator(df, split_name):
        print(f"\n{split_name}:")
        for gen in sorted(df["generator"].unique()):
            gen_df = df[df["generator"] == gen]
            label_counts = Counter(gen_df["label"])
            total = len(gen_df)
            print(f"  {gen} (n={total:,}):")
            print(f"    Human (0): {label_counts[0]:,} ({label_counts[0]/total*100:.2f}%)")
            print(f"    LLM (1):   {label_counts[1]:,} ({label_counts[1]/total*100:.2f}%)")
    
    print_label_by_generator(train_df, "Training")
    print_label_by_generator(val_df, "Validation")

print("\n" + "=" * 80)
print("8. CROSS-TABULATION: LANGUAGE x GENERATOR")
print("=" * 80)

if "language" in train_df.columns and "generator" in train_df.columns:
    def print_crosstab(df, split_name):
        print(f"\n{split_name}:")
        crosstab = pd.crosstab(df["language"], df["generator"], margins=True)
        print(crosstab.to_string())
    
    print_crosstab(train_df, "Training")
    print_crosstab(val_df, "Validation")

print("\n" + "=" * 80)
print("9. CODE LENGTH BY LABEL")
print("=" * 80)

def print_length_by_label(df, split_name):
    print(f"\n{split_name}:")
    for label in [0, 1]:
        label_name = "Human" if label == 0 else "LLM"
        label_df = df[df["label"] == label]
        code_lengths = [len(str(code)) for code in label_df["code"]]
        print(f"  {label_name} (label={label}):")
        print(f"    Mean length:   {np.mean(code_lengths):.1f} characters")
        print(f"    Median length: {np.median(code_lengths):.1f} characters")
        print(f"    Min length:    {np.min(code_lengths):,} characters")
        print(f"    Max length:    {np.max(code_lengths):,} characters")

print_length_by_label(train_df, "Training")
print_length_by_label(val_df, "Validation")

print("\n" + "=" * 80)
print("10. MISSING VALUES CHECK")
print("=" * 80)

def check_missing(df, split_name):
    print(f"\n{split_name}:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  No missing values found")
    else:
        for col, count in missing.items():
            if count > 0:
                print(f"  {col}: {count:,} missing ({count/len(df)*100:.2f}%)")

check_missing(train_df, "Training")
check_missing(val_df, "Validation")
check_missing(test_df, "Test")

print("\n" + "=" * 80)
print("11. UNIQUE VALUES")
print("=" * 80)

print(f"\nTraining set unique values:")
for col in train_df.columns:
    if col != "code":  # Skip code column as it's too large
        unique_count = train_df[col].nunique()
        print(f"  {col}: {unique_count:,} unique values")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

