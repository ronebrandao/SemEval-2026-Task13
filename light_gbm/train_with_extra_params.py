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
import random
import logging
import sys
import hashlib
import re

# Optional Weights & Biases (wandb) integration
try:
    import wandb
except Exception:
    wandb = None


# -----------------------------
# Logging Setup
# -----------------------------
def setup_logging(verbose=True):
    """Setup structured logging with timestamps and levels."""
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)-8s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers = [handler]
    
    return logger

logger = setup_logging(verbose=True)

# Start overall timing
start_time = time.time()
logger.info("=" * 80)
logger.info("Starting SemEval-2026-Task13 LightGBM Training Pipeline")
logger.info("=" * 80)

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
parser.add_argument(
    "--loo_language_out",
    action="store_true",
    help="Run leave-one-language-out evaluation on the TRAIN split (each language held out once as validation)",
)

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
# Weights & Biases (wandb) tracking
parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases experiment tracking")
parser.add_argument("--wandb_project", type=str, default="semeval-task13", help="W&B project name")
parser.add_argument("--wandb_entity", type=str, default="", help="W&B entity/team (optional)")
parser.add_argument("--wandb_tags", type=str, default="", help="Comma-separated W&B tags (optional)")
parser.add_argument("--wandb_log_every", type=int, default=50, help="Log eval metrics to W&B every N boosting rounds")
args = parser.parse_args()

RUNS_ROOT = os.path.join("light_gbm", "runs")
os.makedirs(RUNS_ROOT, exist_ok=True)

def make_run_dir(name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_ROOT, f"{ts}_{name}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

# 1) Load Task A from Hugging Face
logger.info("=" * 80)
logger.info("PHASE 1: Dataset Loading")
logger.info("=" * 80)
load_start = time.time()
logger.info("Loading dataset from Hugging Face: DaniilOr/SemEval-2026-Task13 (Task A)")
ds = load_dataset("DaniilOr/SemEval-2026-Task13", "A")
load_time = time.time() - load_start
logger.info(f"✓ Dataset loaded successfully in {load_time:.2f} seconds")

logger.debug(f"Dataset structure: {ds}")

train_ds = ds["train"]        # backed by task_a_training_set_1.parquet
val_ds   = ds["validation"]   # backed by task_a_validation_set.parquet
test_ds  = ds["test"]         # backed by task_a_test_set.parquet

logger.info(f"Dataset splits loaded:")
logger.info(f"  - Train:      {len(train_ds):,} samples")
logger.info(f"  - Validation: {len(val_ds):,} samples")
logger.info(f"  - Test:       {len(test_ds):,} samples")

# 2) Extract columns
# columns: code, label (0 human / 1 llm), language, generator
logger.info("Extracting columns from datasets...")
X_train_text = train_ds["code"]          # list of strings
y_train      = train_ds["label"]         # list of ints 0/1
X_val_text = val_ds["code"]
y_val      = val_ds["label"]
X_test_text = test_ds["code"]
y_test      = test_ds["label"]

# Log label distribution
train_label_counts = Counter(y_train)
val_label_counts = Counter(y_val)
test_label_counts = Counter(y_test)

logger.info("Label distribution:")
logger.info(f"  Train:      Human={train_label_counts[0]:,} ({train_label_counts[0]/len(y_train)*100:.1f}%), LLM={train_label_counts[1]:,} ({train_label_counts[1]/len(y_train)*100:.1f}%)")
logger.info(f"  Validation: Human={val_label_counts[0]:,} ({val_label_counts[0]/len(y_val)*100:.1f}%), LLM={val_label_counts[1]:,} ({val_label_counts[1]/len(y_val)*100:.1f}%)")
logger.info(f"  Test:       Human={test_label_counts[0]:,} ({test_label_counts[0]/len(y_test)*100:.1f}%), LLM={test_label_counts[1]:,} ({test_label_counts[1]/len(y_test)*100:.1f}%)")

# Optional metadata (present in this dataset): language, generator
# Using them improves robustness when train is dominated by a single language.
# Hugging Face `Dataset` objects don't implement `.get`; check columns explicitly.
lang_train = train_ds["language"] if "language" in train_ds.column_names else None
lang_val   = val_ds["language"]   if "language" in val_ds.column_names else None
lang_test  = test_ds["language"]  if "language" in test_ds.column_names else None

gen_train = train_ds["generator"] if "generator" in train_ds.column_names else None
gen_val   = val_ds["generator"]   if "generator" in val_ds.column_names else None
gen_test  = test_ds["generator"]  if "generator" in test_ds.column_names else None





# Always use the full dataset (no downsampling/balancing).
# Indices are always None, so later code (e.g., TF-IDF cache, LOO) works as expected.
train_indices = None
val_indices = None

# -----------------------------
# TF-IDF cache fingerprint
# -----------------------------
# IMPORTANT:
# We cache BOTH the vectorizer and the transformed matrices. Since the full dataset is always used,
# the fingerprint is based on the current dataset split and TF-IDF params.

def _hash_indices(idxs):
    if idxs is None:
        return "none"
    arr = np.asarray(idxs, dtype=np.int32)
    # 8-byte digest is enough to avoid collisions in practice for this use-case
    return hashlib.blake2b(arr.tobytes(), digest_size=8).hexdigest()


def compute_tfidf_cache_fingerprint():
    payload = {
        "train_n": len(X_train_text),
        "val_n": len(X_val_text),
        "test_n": len(X_test_text),
        "train_indices_hash": _hash_indices(train_indices),
        "val_indices_hash": _hash_indices(val_indices),
        # TF-IDF params that affect the learned vocab / matrices
        "tfidf": {
            "ngram_range": (3, 6),
            "min_df": 3,
            "max_df": 0.95,
            "sublinear_tf": True,
            "lowercase": False,
            "max_features": 200_000,
        },
    }
    s = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.blake2b(s, digest_size=8).hexdigest()


TFIDF_CACHE_FP = compute_tfidf_cache_fingerprint()
logger.info(f"TF-IDF cache fingerprint: {TFIDF_CACHE_FP}")

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
                leading_space_ratio,    
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
logger.info("")
logger.info("=" * 80)
logger.info("PHASE 3: Feature Engineering")
logger.info("=" * 80)
cache_start = time.time()

# TF-IDF cache (char n-grams)
TFIDF = {}

def tfidf_joblib_path(analyzer: str):
    """Return the path to the cached TF-IDF vectorizer joblib, if cache_dir was provided."""
    if not args.cache_dir:
        return None
    os.makedirs(args.cache_dir, exist_ok=True)
    # One file per analyzer so char and char_wb don't collide
    return os.path.join(args.cache_dir, f"tfidf_{analyzer}_{TFIDF_CACHE_FP}.joblib")

def tfidf_matrix_path(analyzer: str, split: str):
    """Return the path to the cached TF-IDF transformed matrix, if cache_dir was provided."""
    if not args.cache_dir:
        return None
    os.makedirs(args.cache_dir, exist_ok=True)
    return os.path.join(args.cache_dir, f"tfidf_{analyzer}_{TFIDF_CACHE_FP}_{split}.npz")




def get_tfidf(analyzer: str):
    key = f"tfidf_{analyzer}"
    if key in TFIDF:
        return TFIDF[key]

    joblib_path = tfidf_joblib_path(analyzer)

    # 1) Load vectorizer if available
    if joblib_path and os.path.exists(joblib_path):
        logger.info(f"Loading cached TF-IDF vectorizer from: {joblib_path}")
        load_start = time.time()
        vec = joblib.load(joblib_path)
        load_time = time.time() - load_start
        logger.info(f"  ✓ Vectorizer loaded in {load_time:.2f} seconds")

        # Try to load cached matrices first
        matrix_paths = {
            "train": tfidf_matrix_path(analyzer, "train"),
            "val": tfidf_matrix_path(analyzer, "val"),
            "test": tfidf_matrix_path(analyzer, "test"),
        }

        if all(p and os.path.exists(p) for p in matrix_paths.values()):
            logger.info("Loading cached TF-IDF matrices...")
            t0 = time.time()
            Xtr = sparse.load_npz(matrix_paths["train"])
            Xva = sparse.load_npz(matrix_paths["val"])
            Xte = sparse.load_npz(matrix_paths["test"])
            logger.info(f"  ✓ Matrices loaded in {time.time() - t0:.2f} seconds")
            logger.info(f"    Shapes: train={Xtr.shape}, val={Xva.shape}, test={Xte.shape}")
        else:
            logger.info("Transforming data with cached vectorizer (this may take a while)...")
            t0 = time.time()
            logger.info("  Transforming training set...")
            Xtr = vec.transform(X_train_text)
            logger.info(f"    ✓ Train transform: {time.time() - t0:.2f} seconds, shape={Xtr.shape}")

            t1 = time.time()
            logger.info("  Transforming validation set...")
            Xva = vec.transform(X_val_text)
            logger.info(f"    ✓ Val transform: {time.time() - t1:.2f} seconds, shape={Xva.shape}")

            t2 = time.time()
            logger.info("  Transforming test set...")
            Xte = vec.transform(X_test_text)
            logger.info(f"    ✓ Test transform: {time.time() - t2:.2f} seconds, shape={Xte.shape}")

            if args.cache_dir:
                logger.info("Caching transformed matrices...")
                sparse.save_npz(matrix_paths["train"], Xtr)
                sparse.save_npz(matrix_paths["val"], Xva)
                sparse.save_npz(matrix_paths["test"], Xte)
                logger.info("  ✓ Matrices cached")

            logger.info(f"  ✓ All transforms completed in {time.time() - t0:.2f} seconds")

        TFIDF[key] = (vec, Xtr, Xva, Xte)
        return TFIDF[key]

    # 2) Otherwise fit a new vectorizer and optionally save it
    logger.info(f"Building new TF-IDF vectorizer (analyzer={analyzer})...")
    logger.info("  Parameters: ngram_range=(3,6), min_df=3, max_df=0.95, max_features=200,000")
    vec = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=(3, 6),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=False,
        max_features=200_000,
    )

    fit_start = time.time()
    logger.info("  Fitting vectorizer on training set...")
    Xtr = vec.fit_transform(X_train_text)
    fit_time = time.time() - fit_start
    logger.info(f"  ✓ Fit+transform train: {fit_time:.2f} seconds, shape={Xtr.shape}")

    t1 = time.time()
    logger.info("  Transforming validation set...")
    Xva = vec.transform(X_val_text)
    logger.info(f"  ✓ Transform val: {time.time() - t1:.2f} seconds, shape={Xva.shape}")

    t2 = time.time()
    logger.info("  Transforming test set...")
    Xte = vec.transform(X_test_text)
    logger.info(f"  ✓ Transform test: {time.time() - t2:.2f} seconds, shape={Xte.shape}")

    logger.info(f"  ✓ Total fit+transform: {time.time() - fit_start:.2f} seconds")

    if joblib_path:
        logger.info(f"Saving TF-IDF vectorizer to: {joblib_path}")
        joblib.dump(vec, joblib_path)
        logger.info("  ✓ Vectorizer saved")

        # Also cache the transformed matrices
        logger.info("Caching transformed matrices...")
        matrix_paths = {
            "train": tfidf_matrix_path(analyzer, "train"),
            "val": tfidf_matrix_path(analyzer, "val"),
            "test": tfidf_matrix_path(analyzer, "test"),
        }
        sparse.save_npz(matrix_paths["train"], Xtr)
        sparse.save_npz(matrix_paths["val"], Xva)
        sparse.save_npz(matrix_paths["test"], Xte)
        logger.info("  ✓ Matrices cached")

    TFIDF[key] = (vec, Xtr, Xva, Xte)
    return TFIDF[key]

# Style cache
STYLE = None

def get_style():
    global STYLE
    if STYLE is not None:
        return STYLE
    logger.info("Building style features...")
    logger.info("  Extracting from training set...")
    t0 = time.time()
    Xtr = sparse.csr_matrix(extract_style_features(X_train_text))
    logger.info(f"    ✓ Training: {time.time() - t0:.2f} seconds, shape={Xtr.shape}")
    
    t1 = time.time()
    logger.info("  Extracting from validation set...")
    Xva = sparse.csr_matrix(extract_style_features(X_val_text))
    logger.info(f"    ✓ Validation: {time.time() - t1:.2f} seconds, shape={Xva.shape}")
    
    t2 = time.time()
    logger.info("  Extracting from test set...")
    Xte = sparse.csr_matrix(extract_style_features(X_test_text))
    logger.info(f"    ✓ Test: {time.time() - t2:.2f} seconds, shape={Xte.shape}")
    
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
        logger.warning("  Meta features not available (missing language or generator columns)")
        META = (None, None, None, None)
        return META

    logger.info("Building meta features (language+generator one-hot encoding)...")
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    train_meta = np.column_stack([lang_train, gen_train])
    val_meta   = np.column_stack([lang_val,   gen_val])
    test_meta  = np.column_stack([lang_test,  gen_test])

    logger.info("  Fitting encoder on training set...")
    t0 = time.time()
    Xtr = enc.fit_transform(train_meta)
    logger.info(f"    ✓ Training: {time.time() - t0:.2f} seconds, shape={Xtr.shape}")
    
    t1 = time.time()
    logger.info("  Transforming validation set...")
    Xva = enc.transform(val_meta)
    logger.info(f"    ✓ Validation: {time.time() - t1:.2f} seconds, shape={Xva.shape}")
    
    t2 = time.time()
    logger.info("  Transforming test set...")
    Xte = enc.transform(test_meta)
    logger.info(f"    ✓ Test: {time.time() - t2:.2f} seconds, shape={Xte.shape}")
    
    META = (enc, Xtr, Xva, Xte)
    return META

cache_time = time.time() - cache_start
logger.info("")
logger.info(f"✓ Feature engineering completed in {cache_time:.2f} seconds ({cache_time/60:.2f} minutes)")

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
        logger.info("No language information available, skipping sample weighting")
        return None
    counts = Counter(lang_train)
    w = np.array([1.0 / counts[l] for l in lang_train], dtype=np.float32)
    # normalize weights to have mean ~1.0
    w *= (len(w) / float(w.sum()))
    logger.info(f"Using language-balanced sample weights for {len(counts)} languages")
    logger.debug(f"  Language distribution: {dict(counts)}")
    return w


# ---------------------------
# W&B helpers
# ---------------------------

def sanitize_wandb_name(name: str) -> str:
    """Sanitize a string for W&B artifact/run names.

    W&B artifact names may only contain alphanumerics, dashes, underscores, and dots.
    """
    # Replace any invalid char sequence with a single underscore
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(name))
    # Avoid leading/trailing underscores/dots from odd inputs
    safe = safe.strip("_.-")
    return safe or "unnamed"
def start_wandb_if_enabled(run_name: str, run_dir: str, cfg: dict):
    if not getattr(args, "wandb", False):
        return None
    if wandb is None:
        raise RuntimeError(
            "wandb is not installed but --wandb was provided. Install with: pip install wandb"
        )

    tags = [t.strip() for t in (args.wandb_tags or "").split(",") if t.strip()]
    init_kwargs = {
        "project": args.wandb_project,
        "name": run_name,
        "config": cfg,
        "dir": run_dir,
        "tags": tags,
        "settings": wandb.Settings(console="wrap"),
    }
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity

    return wandb.init(**init_kwargs)



def log_lgbm_evals_to_wandb(clf, log_every: int = 50):
    if wandb is None or wandb.run is None:
        return
    # LightGBM sklearn API exposes eval curves here when eval_set is provided
    evals = getattr(clf, "evals_result_", None)
    if not evals:
        return

    # evals is like: {"training": {"binary_logloss": [...]}, "valid_0": {...}}
    for dataset_name, metrics in evals.items():
        for metric_name, values in metrics.items():
            for i, v in enumerate(values):
                step = i + 1
                if step % log_every == 0 or step == len(values):
                    wandb.log({f"{dataset_name}/{metric_name}": float(v)}, step=step)

    # Also log best iteration / best scores if available
    if getattr(clf, "best_iteration_", None):
        wandb.log({"best_iteration": int(clf.best_iteration_)})
    if getattr(clf, "best_score_", None):
        # best_score_ is a dict; keep it in summary for easy comparison
        wandb.run.summary["best_score"] = clf.best_score_


# Stream LightGBM eval metrics to W&B during training
def wandb_lgbm_callback(log_every: int = 50):
    """LightGBM callback that streams eval metrics to W&B during training."""

    def _cb(env):
        if wandb is None or wandb.run is None:
            return

        step = int(env.iteration) + 1
        end_it = getattr(env, "end_iteration", None)

        # Log every N rounds and also on the final iteration
        if step % int(log_every) != 0 and (end_it is None or step != int(end_it)):
            return

        payload = {}
        # evaluation_result_list items are tuples:
        # (data_name, eval_name, result, is_higher_better, stdv)
        for item in env.evaluation_result_list or []:
            data_name, eval_name, result = item[0], item[1], item[2]
            payload[f"{data_name}/{eval_name}"] = float(result)

        if payload:
            wandb.log(payload, step=step)

    # LightGBM expects callbacks to have an 'order' attribute
    _cb.order = 10
    return _cb


# -----------------------------------------------------------
# Leave-One-Language-Out evaluation over TRAIN split
# -----------------------------------------------------------
def run_leave_one_language_out(feature_mode: str, tfidf_analyzer: str, base_run_name: str):
    """Train on all but one language from TRAIN split; validate on the held-out language.

    This is meant to simulate generalization to unseen languages.

    Note: This reuses `run_experiment` by temporarily overwriting the global split variables
    (X_train_text/y_train/lang_train/gen_train and X_val_text/y_val/lang_val/gen_val) per fold.
    """
    global X_train_text, y_train, lang_train, gen_train
    global X_val_text, y_val, lang_val, gen_val
    global TFIDF, STYLE, META
    global TFIDF_CACHE_FP
    global train_indices, val_indices

    if lang_train is None:
        raise RuntimeError("Cannot run --loo_language_out because 'language' column is missing in the TRAIN split")

    # Snapshot originals
    X_train_text_orig = list(X_train_text)
    y_train_orig = list(y_train)
    lang_train_orig = list(lang_train) if lang_train is not None else None
    gen_train_orig = list(gen_train) if gen_train is not None else None

    X_val_text_orig = list(X_val_text)
    y_val_orig = list(y_val)
    lang_val_orig = list(lang_val) if lang_val is not None else None
    gen_val_orig = list(gen_val) if gen_val is not None else None

    # Use TRAIN split only for fold generation
    unique_langs = sorted(set(lang_train_orig))

    logger.info("")
    logger.info("=" * 80)
    logger.info("LEAVE-ONE-LANGUAGE-OUT EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Languages in TRAIN: {unique_langs}")
    logger.info(f"Total folds: {len(unique_langs)}")

    fold_reports = []

    try:
        for fold_i, hold_lang in enumerate(unique_langs, 1):
            logger.info("")
            logger.info("-" * 80)
            logger.info(f"Fold {fold_i}/{len(unique_langs)} — holding out language: {hold_lang}")
            logger.info("-" * 80)

            # Indices relative to the ORIGINAL train split
            idx_hold = [i for i, l in enumerate(lang_train_orig) if l == hold_lang]
            idx_train = [i for i, l in enumerate(lang_train_orig) if l != hold_lang]

            # Overwrite globals so `run_experiment` uses the fold splits
            X_train_text = [X_train_text_orig[i] for i in idx_train]
            y_train = [y_train_orig[i] for i in idx_train]
            lang_train = [lang_train_orig[i] for i in idx_train] if lang_train_orig is not None else None
            gen_train = [gen_train_orig[i] for i in idx_train] if gen_train_orig is not None else None

            X_val_text = [X_train_text_orig[i] for i in idx_hold]
            y_val = [y_train_orig[i] for i in idx_hold]
            lang_val = [lang_train_orig[i] for i in idx_hold] if lang_train_orig is not None else None
            gen_val = [gen_train_orig[i] for i in idx_hold] if gen_train_orig is not None else None

            logger.info(f"  Fold sizes: train={len(X_train_text):,}  val(heldout)={len(X_val_text):,}")

            # For TF-IDF caching fingerprint uniqueness (indices refer to original TRAIN ordering)
            train_indices = idx_train
            val_indices = idx_hold

            # Clear feature caches so each fold recomputes on the fold-specific data
            TFIDF = {}
            STYLE = None
            META = None

            # Recompute cache fingerprint for this fold (affects tfidf cache file names)
            TFIDF_CACHE_FP = compute_tfidf_cache_fingerprint()
            logger.info(f"  TF-IDF cache fingerprint (fold): {TFIDF_CACHE_FP}")

            # Run a normal experiment using the fold splits
            fold_name = f"{base_run_name}_loo_holdout_{hold_lang}"
            rep = run_experiment(feature_mode, tfidf_analyzer, fold_name)
            rep["heldout_language"] = hold_lang
            fold_reports.append(rep)

        # Summarize folds
        logger.info("")
        logger.info("=" * 80)
        logger.info("LOO SUMMARY")
        logger.info("=" * 80)
        vals = [r.get("val_macro_f1", 0.0) for r in fold_reports]
        tests = [r.get("test_macro_f1", 0.0) for r in fold_reports]
        if vals:
            logger.info(f"Val Macro-F1 across folds:  mean={float(np.mean(vals)):.4f}  std={float(np.std(vals)):.4f}")
        if tests:
            logger.info(f"Test Macro-F1 across folds: mean={float(np.mean(tests)):.4f}  std={float(np.std(tests)):.4f}")

        # Save a summary json in the runs root
        summary_path = os.path.join(RUNS_ROOT, f"loo_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{base_run_name}.json")
        with open(summary_path, "w") as f:
            json.dump({"folds": fold_reports}, f, indent=2)
        logger.info(f"Saved LOO summary to: {summary_path}")

    finally:
        # Restore original globals
        X_train_text = X_train_text_orig
        y_train = y_train_orig
        lang_train = lang_train_orig
        gen_train = gen_train_orig

        X_val_text = X_val_text_orig
        y_val = y_val_orig
        lang_val = lang_val_orig
        gen_val = gen_val_orig

        # Reset fold indices
        train_indices = None
        val_indices = None

        # Clear caches to avoid accidental reuse after LOO
        TFIDF = {}
        STYLE = None
        META = None


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

    cfg["run_dir"] = run_dir
    wb_run = start_wandb_if_enabled(run_name=run_name, run_dir=run_dir, cfg=cfg)
    if wb_run:
        logger.info(f"✓ Weights & Biases tracking enabled: {wb_run.url}")

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"EXPERIMENT: {run_name}")
    logger.info("=" * 80)
    logger.info(f"  Features:        {feature_mode}")
    logger.info(f"  TF-IDF Analyzer: {tfidf_analyzer}")
    logger.info(f"  Output Directory: {run_dir}")
    logger.info(f"  Max Estimators:  {args.n_estimators}")
    logger.info(f"  Early Stopping:  {args.early_stopping_rounds} rounds, min_delta={args.early_stopping_min_delta}")

    # Build matrices
    logger.info("")
    logger.info("Building feature matrices...")
    vec, enc, X_train, X_val, X_test = build_matrix(feature_mode, tfidf_analyzer)
    logger.info(f"  ✓ Feature matrices built:")
    logger.info(f"    Train:      {X_train.shape}")
    logger.info(f"    Validation: {X_val.shape}")
    logger.info(f"    Test:       {X_test.shape}")

    # Save preprocessors
    logger.info("Saving preprocessors...")
    if vec is not None:
        joblib.dump(vec, os.path.join(run_dir, "tfidf.joblib"))
        logger.info("  ✓ TF-IDF vectorizer saved")
    if enc is not None:
        joblib.dump(enc, os.path.join(run_dir, "meta_encoder.joblib"))
        logger.info("  ✓ Meta encoder saved")

    # Weights
    logger.info("")
    logger.info("Computing sample weights...")
    sample_weight = compute_lang_weights()
    if sample_weight is not None:
        logger.info("  ✓ Sample weights computed")

    # Train
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 4: Model Training")
    logger.info("=" * 80)
    logger.info("Initializing LightGBM classifier...")
    train_start = time.time()
    clf = LGBMClassifier(
        objective="binary",
        force_col_wise=True,
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
    logger.info("  Hyperparameters:")
    logger.info(f"    learning_rate={0.03}, num_leaves={31}, min_child_samples={200}")
    logger.info(f"    subsample={0.7}, colsample_bytree={0.7}")
    logger.info(f"    reg_alpha={1.0}, reg_lambda={5.0}")
    logger.info("")
    logger.info("Starting training (this may take a while)...")
    logger.info("  Evaluation metrics: binary_logloss, auc")
    logger.info("  Logging every 50 iterations")

    callbacks = [
        log_evaluation(50),
        early_stopping(
            stopping_rounds=args.early_stopping_rounds,
            first_metric_only=True,
            min_delta=args.early_stopping_min_delta,
        ),
    ]
    if wandb is not None and wandb.run is not None:
        callbacks.append(wandb_lgbm_callback(args.wandb_log_every))

    clf.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_metric=["binary_logloss", "auc"],
        callbacks=callbacks,
    )
    # Log LightGBM learning curves (loss / AUC) to W&B, if enabled
    log_lgbm_evals_to_wandb(clf, log_every=args.wandb_log_every)
    train_time = time.time() - train_start
    
    best_iter = getattr(clf, "best_iteration_", None)
    if best_iter:
        logger.info(f"  ✓ Training completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
        logger.info(f"  ✓ Best iteration: {best_iter} (early stopping triggered)")
    else:
        logger.info(f"  ✓ Training completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
        logger.info(f"  ✓ Used all {args.n_estimators} iterations")

    logger.info("Saving trained model...")
    joblib.dump(clf, os.path.join(run_dir, "model.joblib"))
    logger.info("  ✓ Model saved")

    # Threshold tuning on validation for Macro-F1
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 5: Model Evaluation")
    logger.info("=" * 80)
    logger.info("Tuning classification threshold on validation set...")
    logger.info("  Searching threshold range: [0.05, 0.95] with 181 candidates")
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
    logger.info(f"  ✓ Best threshold: {best_th:.4f} (val Macro-F1: {best_f1:.4f})")

    with open(os.path.join(run_dir, "best_threshold.json"), "w") as f:
        json.dump({"best_threshold": best_th, "val_macro_f1": best_f1}, f, indent=2)

    # Evaluate
    logger.info("")
    logger.info("Evaluating on validation and test sets...")
    y_val_pred = (val_proba >= best_th).astype(int)
    val_macro = f1_score(y_val, y_val_pred, average="macro")
    logger.info(f"  Validation Macro-F1: {val_macro:.4f}")

    test_proba = clf.predict_proba(X_test)[:, 1]
    y_test_pred = (test_proba >= best_th).astype(int)
    test_macro = f1_score(y_test, y_test_pred, average="macro")
    logger.info(f"  Test Macro-F1:      {test_macro:.4f}")

    # Save reports
    report = {
        "train_time_sec": train_time,
        "best_threshold": best_th,
        "val_macro_f1": float(val_macro),
        "test_macro_f1": float(test_macro),
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    logger.info("")
    logger.info("Saving classification reports...")
    with open(os.path.join(run_dir, "classification_report_val.txt"), "w") as f:
        f.write(classification_report(y_val, y_val_pred, target_names=["human (0)", "llm (1)"]))
    logger.info("  ✓ Validation report saved")
    
    with open(os.path.join(run_dir, "classification_report_test.txt"), "w") as f:
        f.write(classification_report(y_test, y_test_pred, target_names=["human (0)", "llm (1)"]))
    logger.info("  ✓ Test report saved")

    logger.info("")
    logger.info("=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)
    logger.info(f"  Training time:        {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
    logger.info(f"  Best threshold:      {best_th:.4f}")
    logger.info(f"  Validation Macro-F1: {val_macro:.4f}")
    logger.info(f"  Test Macro-F1:       {test_macro:.4f}")

    # Per-language breakdown on test
    if lang_test is not None:
        logger.info("")
        logger.info("Per-language performance (test set):")
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
                logger.info(f"    {L:>12s}: n={len(idx):5d}  macro_f1={f1L:.4f}")
            txt = "\n".join(lines) + "\n"
            with open(os.path.join(run_dir, "per_language_test.txt"), "w") as f:
                f.write(txt)
        except Exception as e:
            logger.warning(f"Could not compute per-language breakdown: {e}")

    # Log final metrics and model to W&B, if enabled
    if wandb is not None and wandb.run is not None:
        logger.info("")
        logger.info("Logging to Weights & Biases...")
        wandb.log({
            "train_time_sec": float(train_time),
            "best_threshold": float(best_th),
            "val_macro_f1": float(val_macro),
            "test_macro_f1": float(test_macro),
        })

        # Save the trained model + key outputs as an artifact
        artifact_name = sanitize_wandb_name(f"lgbm_{run_name}")
        art = wandb.Artifact(name=artifact_name, type="model")
        art.add_file(os.path.join(run_dir, "model.joblib"))
        art.add_file(os.path.join(run_dir, "config.json"))
        art.add_file(os.path.join(run_dir, "metrics.json"))
        art.add_file(os.path.join(run_dir, "best_threshold.json"))
        wandb.log_artifact(art)
        wandb.finish()
        logger.info("  ✓ W&B logging completed")

    return report


# -----------------------------------
# Main: single run or 4-way ablation
# -----------------------------------
logger.info("")
logger.info("=" * 80)
logger.info("STARTING EXPERIMENTS")
logger.info("=" * 80)

if args.loo_language_out:
    logger.info("Running leave-one-language-out evaluation...")
    run_leave_one_language_out(args.features, args.tfidf_analyzer, args.run_name)
elif args.ablation:
    # Standard ablation grid
    logger.info("Running 4-way ablation study...")
    grid = [
        ("tfidf", args.tfidf_analyzer, f"{args.run_name}_tfidf"),
        ("style", args.tfidf_analyzer, f"{args.run_name}_style"),
        ("tfidf+style", args.tfidf_analyzer, f"{args.run_name}_tfidf_style"),
        ("tfidf+style+meta", args.tfidf_analyzer, f"{args.run_name}_tfidf_style_meta"),
    ]
    logger.info(f"  Total experiments: {len(grid)}")
    for i, (feat_mode, analyzer, name) in enumerate(grid, 1):
        logger.info("")
        logger.info(f"Experiment {i}/{len(grid)}: {name}")
        run_experiment(feat_mode, analyzer, name)
else:
    logger.info("Running single experiment...")
    run_experiment(args.features, args.tfidf_analyzer, args.run_name)

# Total execution time
total_time = time.time() - start_time
logger.info("")
logger.info("=" * 80)
logger.info("PIPELINE COMPLETED")
logger.info("=" * 80)
logger.info(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
logger.info("=" * 80)