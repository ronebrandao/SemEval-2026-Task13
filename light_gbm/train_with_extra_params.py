from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier
from lightgbm.callback import log_evaluation, early_stopping
import joblib
import time
import json
import numpy as np
from scipy import sparse
from collections import Counter
import os
import argparse
from datetime import datetime

# Start overall timing
start_time = time.time()

# -----------------------------
# CLI / experiment configuration
# -----------------------------
parser = argparse.ArgumentParser(description="LightGBM ablations for SemEval Task 13 (Task A)")
parser.add_argument("--run_name", type=str, default="run", help="Name suffix for the run folder")
parser.add_argument("--features", type=str, default="tfidf+style+meta",
                    choices=["tfidf", "style", "tfidf+style", "tfidf+style+meta"],
                    help="Which feature blocks to use")
parser.add_argument("--tfidf_analyzer", type=str, default="char", choices=["char", "char_wb"],
                    help="TF-IDF analyzer; char_wb often reduces noise")
parser.add_argument("--ablation", action="store_true",
                    help="Run the 4 standard ablations and save each to its own folder")

# Extra LightGBM hyperparameters
parser.add_argument("--n_estimators", type=int, default=3000,
                    help="Max boosting rounds (upper cap; early stopping may stop earlier)")
parser.add_argument("--early_stopping_rounds", type=int, default=200,
                    help="Stop if metric doesn't improve by min_delta for this many rounds")
parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4,
                    help="Minimum improvement to count as progress (prevents endless tiny improvements)")
parser.add_argument(
    "--cache_dir",
    type=str,
    default="",
    help="If set, reuse/save TF-IDF vectorizer joblib in this directory (no matrix caching)",
)
args = parser.parse_args()

RUNS_ROOT = os.path.join("light_gbm", "runs")
os.makedirs(RUNS_ROOT, exist_ok=True)

def make_run_dir(name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_ROOT, f"{ts}_{name}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

# 1) Load Task A from Hugging Face
print(">>> Loading dataset...")
load_start = time.time()
ds = load_dataset("DaniilOr/SemEval-2026-Task13", "A")
load_time = time.time() - load_start
print(f"Dataset loaded in {load_time:.2f} seconds")

print(ds)

train_ds = ds["train"]        # backed by task_a_training_set_1.parquet
val_ds   = ds["validation"]   # backed by task_a_validation_set.parquet
test_ds  = ds["test"]         # backed by task_a_test_set.parquet

# 2) Extract columns
# columns: code, label (0 human / 1 llm), language, generator
X_train_text = train_ds["code"]          # list of strings
y_train      = train_ds["label"]         # list of ints 0/1
X_val_text = val_ds["code"]
y_val      = val_ds["label"]
X_test_text = test_ds["code"]
y_test      = test_ds["label"]

# Optional metadata (present in this dataset): language, generator
# Using them improves robustness when train is dominated by a single language.
# Hugging Face `Dataset` objects don't implement `.get`; check columns explicitly.
lang_train = train_ds["language"] if "language" in train_ds.column_names else None
lang_val   = val_ds["language"]   if "language" in val_ds.column_names else None
lang_test  = test_ds["language"]  if "language" in test_ds.column_names else None

gen_train = train_ds["generator"] if "generator" in train_ds.column_names else None
gen_val   = val_ds["generator"]   if "generator" in val_ds.column_names else None
gen_test  = test_ds["generator"]  if "generator" in test_ds.column_names else None

def extract_style_features(codes):
    feats = []
    for code in codes:
        if code is None:
            code = ""

        lines = code.splitlines() or [""]
        n_lines = len(lines)
        n_chars = len(code)

        n_lines_safe = max(1, n_lines)
        n_chars_safe = max(1, n_chars)

        # Line length stats
        line_lengths = [len(l) for l in lines]
        avg_line_len = float(sum(line_lengths) / n_lines_safe)
        max_line_len = float(max(line_lengths))

        # Comment signals
        # - full comment lines (start)
        # - inline comments (appear later in the line)
        comment_line_prefixes = ("#", "//", "/*")
        comment_lines = 0
        inline_comment_lines = 0
        docstring_like_lines = 0

        for l in lines:
            s = l.strip()
            if s.startswith(comment_line_prefixes):
                comment_lines += 1
            # crude docstring detection (Python-ish) but still useful as a generic signal
            if s.startswith(('"""', "'''")) or s.endswith(('"""', "'''")):
                docstring_like_lines += 1

            # inline comment detection (ignore if it's a pure comment line)
            if not s.startswith(comment_line_prefixes):
                # common inline comment markers
                if " #" in l or "//" in l or "/*" in l:
                    inline_comment_lines += 1

        blank_lines = sum(1 for l in lines if l.strip() == "")

        comment_ratio = comment_lines / n_lines_safe
        inline_comment_ratio = inline_comment_lines / n_lines_safe
        docstring_ratio = docstring_like_lines / n_lines_safe
        blank_ratio = blank_lines / n_lines_safe

        # Indentation stats (spaces-only) + explicit tab/space mixing
        indents_spaces = []
        leading_tabs_lines = 0
        leading_spaces_lines = 0
        mixed_indent_lines = 0

        for l in lines:
            # leading whitespace segment
            stripped = l.lstrip(" \t")
            prefix = l[: len(l) - len(stripped)]

            if prefix:
                has_tab = "\t" in prefix
                has_space = " " in prefix
                if has_tab and has_space:
                    mixed_indent_lines += 1
                elif has_tab:
                    leading_tabs_lines += 1
                elif has_space:
                    leading_spaces_lines += 1

            # keep a simple indentation measure based on spaces only (as before)
            indent_spaces = len(l) - len(l.lstrip(" "))
            indents_spaces.append(indent_spaces)

        indent_mean = float(sum(indents_spaces) / n_lines_safe)
        indent_std = float(np.std(indents_spaces)) if len(indents_spaces) > 1 else 0.0

        leading_tab_ratio = leading_tabs_lines / n_lines_safe
        leading_space_ratio = leading_spaces_lines / n_lines_safe
        mixed_indent_ratio = mixed_indent_lines / n_lines_safe

        # Global whitespace ratios
        n_tabs = code.count("\t")
        tab_ratio = n_tabs / n_chars_safe

        n_digits = sum(ch.isdigit() for ch in code)
        n_alpha = sum(ch.isalpha() for ch in code)
        n_space = sum(ch.isspace() for ch in code)

        digit_ratio = n_digits / n_chars_safe
        alpha_ratio = n_alpha / n_chars_safe
        space_ratio = n_space / n_chars_safe

        feats.append(
            [
                n_chars,
                n_lines,
                avg_line_len,
                max_line_len,
                comment_ratio,
                inline_comment_ratio,
                docstring_ratio,
                blank_ratio,
                indent_mean,
                indent_std,
                tab_ratio,
                leading_tab_ratio,
                mixed_indent_ratio,
                digit_ratio,
                alpha_ratio,
                space_ratio,
            ]
        )

    return np.array(feats, dtype=np.float32)

# -----------------------------------
# Feature caches (computed once)
# -----------------------------------
print(">>> Preparing feature caches...")
cache_start = time.time()

# TF-IDF cache (char n-grams)
TFIDF = {}

def tfidf_joblib_path(analyzer: str):
    """Return the path to the cached TF-IDF vectorizer joblib, if cache_dir was provided."""
    if not args.cache_dir:
        return None
    os.makedirs(args.cache_dir, exist_ok=True)
    # One file per analyzer so char and char_wb don't collide
    return os.path.join(args.cache_dir, f"tfidf_{analyzer}.joblib")


def get_tfidf(analyzer: str):
    key = f"tfidf_{analyzer}"
    if key in TFIDF:
        return TFIDF[key]

    joblib_path = tfidf_joblib_path(analyzer)

    # 1) Load vectorizer if available
    if joblib_path and os.path.exists(joblib_path):
        print(f">>> Loading TF-IDF vectorizer from: {joblib_path}")
        vec = joblib.load(joblib_path)

        Xtr = vec.transform(X_train_text)
        Xva = vec.transform(X_val_text)
        Xte = vec.transform(X_test_text)

        TFIDF[key] = (vec, Xtr, Xva, Xte)
        return TFIDF[key]

    # 2) Otherwise fit a new vectorizer and optionally save it
    print(f">>> Building TF-IDF ({analyzer})...")
    vec = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=(3, 6),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=False,
        max_features=200_000,
    )

    Xtr = vec.fit_transform(X_train_text)
    Xva = vec.transform(X_val_text)
    Xte = vec.transform(X_test_text)

    if joblib_path:
        print(f">>> Saving TF-IDF vectorizer to: {joblib_path}")
        joblib.dump(vec, joblib_path)

    TFIDF[key] = (vec, Xtr, Xva, Xte)
    return TFIDF[key]

# Style cache
STYLE = None

def get_style():
    global STYLE
    if STYLE is not None:
        return STYLE
    print(">>> Building style features...")
    Xtr = sparse.csr_matrix(extract_style_features(X_train_text))
    Xva = sparse.csr_matrix(extract_style_features(X_val_text))
    Xte = sparse.csr_matrix(extract_style_features(X_test_text))
    STYLE = (Xtr, Xva, Xte)
    return STYLE

# Meta cache (language + generator)
META = None

def get_meta():
    global META
    if META is not None:
        return META

    use_meta = (lang_train is not None) and (gen_train is not None)
    if not use_meta:
        META = (None, None, None, None)
        return META

    print(">>> Building meta (language+generator) one-hot...")
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    train_meta = np.column_stack([lang_train, gen_train])
    val_meta   = np.column_stack([lang_val,   gen_val])
    test_meta  = np.column_stack([lang_test,  gen_test])

    Xtr = enc.fit_transform(train_meta)
    Xva = enc.transform(val_meta)
    Xte = enc.transform(test_meta)
    META = (enc, Xtr, Xva, Xte)
    return META

cache_time = time.time() - cache_start
print(f"Feature caches ready in {cache_time:.2f} seconds")

# -----------------------------------
# Experiment runner
# -----------------------------------

def build_matrix(feature_mode: str, tfidf_analyzer: str):
    parts_tr, parts_va, parts_te = [], [], []

    if feature_mode in ("tfidf", "tfidf+style", "tfidf+style+meta"):
        vec, Xtr, Xva, Xte = get_tfidf(tfidf_analyzer)
        parts_tr.append(Xtr)
        parts_va.append(Xva)
        parts_te.append(Xte)
    else:
        vec = None

    if feature_mode in ("style", "tfidf+style", "tfidf+style+meta"):
        Xtr_s, Xva_s, Xte_s = get_style()
        parts_tr.append(Xtr_s)
        parts_va.append(Xva_s)
        parts_te.append(Xte_s)

    if feature_mode == "tfidf+style+meta":
        enc, Xtr_m, Xva_m, Xte_m = get_meta()
        if Xtr_m is not None:
            parts_tr.append(Xtr_m)
            parts_va.append(Xva_m)
            parts_te.append(Xte_m)
    else:
        enc = None

    X_train = sparse.hstack(parts_tr).tocsr() if len(parts_tr) > 1 else parts_tr[0].tocsr()
    X_val   = sparse.hstack(parts_va).tocsr() if len(parts_va) > 1 else parts_va[0].tocsr()
    X_test  = sparse.hstack(parts_te).tocsr() if len(parts_te) > 1 else parts_te[0].tocsr()

    return vec, enc, X_train, X_val, X_test


def compute_lang_weights():
    # Language reweighting to reduce "Python dominates everything" behavior.
    # Keeps most of the data (better than downsampling) while balancing contribution.
    if lang_train is None:
        return None
    counts = Counter(lang_train)
    w = np.array([1.0 / counts[l] for l in lang_train], dtype=np.float32)
    # normalize weights to have mean ~1.0
    w *= (len(w) / float(w.sum()))
    print("Using language-balanced sample weights. #langs:", len(counts))
    return w


def run_experiment(feature_mode: str, tfidf_analyzer: str, run_name: str):
    run_dir = make_run_dir(run_name)

    # Persist config for reproducibility
    cfg = {
        "feature_mode": feature_mode,
        "tfidf_analyzer": tfidf_analyzer,
        "n_estimators": args.n_estimators,
        "early_stopping_rounds": args.early_stopping_rounds,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "lgbm": {
            "n_estimators": args.n_estimators,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 200,
            "subsample": 0.7,
            "subsample_freq": 1,
            "colsample_bytree": 0.7,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
        },
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print("\n" + "=" * 80)
    print(f">>> RUN: {run_name}")
    print(f"    features={feature_mode}  tfidf_analyzer={tfidf_analyzer}")
    print(f"    out={run_dir}")
    print("=" * 80)

    # Build matrices
    vec, enc, X_train, X_val, X_test = build_matrix(feature_mode, tfidf_analyzer)
    print("Shapes:", X_train.shape, X_val.shape, X_test.shape)

    # Save preprocessors
    if vec is not None:
        joblib.dump(vec, os.path.join(run_dir, "tfidf.joblib"))
    if enc is not None:
        joblib.dump(enc, os.path.join(run_dir, "meta_encoder.joblib"))

    # Weights
    sample_weight = compute_lang_weights()

    # Train
    train_start = time.time()
    clf = LGBMClassifier(
        objective="binary",
        n_estimators=args.n_estimators,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=200,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=5.0,
        n_jobs=-1,
    )

    clf.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[
            log_evaluation(50),
            early_stopping(
                stopping_rounds=args.early_stopping_rounds,
                first_metric_only=True,
                min_delta=args.early_stopping_min_delta,
            ),
        ],
    )
    train_time = time.time() - train_start

    joblib.dump(clf, os.path.join(run_dir, "model.joblib"))

    # Threshold tuning on validation for Macro-F1
    val_proba = clf.predict_proba(X_val)[:, 1]
    ths = np.linspace(0.05, 0.95, 181)
    best_th = 0.5
    best_f1 = -1.0
    for t in ths:
        preds = (val_proba >= t).astype(int)
        f1 = f1_score(y_val, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_th = float(t)

    with open(os.path.join(run_dir, "best_threshold.json"), "w") as f:
        json.dump({"best_threshold": best_th, "val_macro_f1": best_f1}, f, indent=2)

    # Evaluate
    y_val_pred = (val_proba >= best_th).astype(int)
    val_macro = f1_score(y_val, y_val_pred, average="macro")

    test_proba = clf.predict_proba(X_test)[:, 1]
    y_test_pred = (test_proba >= best_th).astype(int)
    test_macro = f1_score(y_test, y_test_pred, average="macro")

    # Save reports
    report = {
        "train_time_sec": train_time,
        "best_threshold": best_th,
        "val_macro_f1": float(val_macro),
        "test_macro_f1": float(test_macro),
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    with open(os.path.join(run_dir, "classification_report_val.txt"), "w") as f:
        f.write(classification_report(y_val, y_val_pred, target_names=["human (0)", "llm (1)"]))

    with open(os.path.join(run_dir, "classification_report_test.txt"), "w") as f:
        f.write(classification_report(y_test, y_test_pred, target_names=["human (0)", "llm (1)"]))

    print(f"Train time: {train_time:.2f}s")
    print(f"Best threshold (val Macro-F1): {best_th:.3f} -> {val_macro:.4f}")
    print(f"Macro-F1 (test): {test_macro:.4f}")

    # Per-language breakdown on test
    if lang_test is not None:
        try:
            langs = np.array(lang_test)
            unique_langs = sorted(set(langs))
            lines = []
            lines.append("Per-language Macro-F1 (test):")
            for L in unique_langs:
                idx = np.where(langs == L)[0]
                if len(idx) < 10:
                    continue
                f1L = f1_score(np.array(y_test)[idx], np.array(y_test_pred)[idx], average="macro")
                lines.append(f"  {L:>12s}: n={len(idx):5d}  macro_f1={f1L:.4f}")
            txt = "\n".join(lines) + "\n"
            print("\n" + txt)
            with open(os.path.join(run_dir, "per_language_test.txt"), "w") as f:
                f.write(txt)
        except Exception as e:
            print("Could not compute per-language breakdown:", e)

    return report


# -----------------------------------
# Main: single run or 4-way ablation
# -----------------------------------
if args.ablation:
    # Standard ablation grid
    grid = [
        ("tfidf", args.tfidf_analyzer, f"{args.run_name}_tfidf"),
        ("style", args.tfidf_analyzer, f"{args.run_name}_style"),
        ("tfidf+style", args.tfidf_analyzer, f"{args.run_name}_tfidf_style"),
        ("tfidf+style+meta", args.tfidf_analyzer, f"{args.run_name}_tfidf_style_meta"),
    ]

    for feat_mode, analyzer, name in grid:
        run_experiment(feat_mode, analyzer, name)
else:
    run_experiment(args.features, args.tfidf_analyzer, args.run_name)

# Total execution time
total_time = time.time() - start_time
print("\n" + "="*50)
print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print("="*50)