from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from lightgbm import LGBMClassifier
from lightgbm.callback import log_evaluation
import joblib
import time
import numpy as np
from scipy import sparse

# Start overall timing
start_time = time.time()

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

        line_lengths = [len(l) for l in lines]
        avg_line_len = sum(line_lengths) / n_lines_safe
        max_line_len = max(line_lengths)

        comment_lines = sum(
            1 for l in lines if l.strip().startswith(("#", "//", "/*"))
        )
        blank_lines = sum(1 for l in lines if l.strip() == "")

        comment_ratio = comment_lines / n_lines_safe
        blank_ratio = blank_lines / n_lines_safe

        indents = []
        for l in lines:
            indent = len(l) - len(l.lstrip(" "))
            indents.append(indent)
        indent_mean = sum(indents) / n_lines_safe
        indent_std = float(np.std(indents)) if len(indents) > 1 else 0.0

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
                blank_ratio,
                indent_mean,
                indent_std,
                tab_ratio,
                digit_ratio,
                alpha_ratio,
                space_ratio,
            ]
        )
    return np.array(feats, dtype=np.float32)

print(">>> Building TF-IDF...")
tfidf_start = time.time()
# Character n-grams work very well for style detection in code
tfidf = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 6),      # 3–6 char n-grams
    min_df=3,                # ignore super-rare n-grams
    max_features=150_000     # adjust if you hit RAM issues
)

X_train_tfidf = tfidf.fit_transform(X_train_text)
X_val_tfidf   = tfidf.transform(X_val_text)
X_test_tfidf  = tfidf.transform(X_test_text)

# Style features (dense)
X_train_style = extract_style_features(X_train_text)
X_val_style   = extract_style_features(X_val_text)
X_test_style  = extract_style_features(X_test_text)

# Convert style features to sparse and concatenate with TF-IDF
X_train_style_sp = sparse.csr_matrix(X_train_style)
X_val_style_sp   = sparse.csr_matrix(X_val_style)
X_test_style_sp  = sparse.csr_matrix(X_test_style)

X_train = sparse.hstack([X_train_tfidf, X_train_style_sp]).tocsr()
X_val   = sparse.hstack([X_val_tfidf,   X_val_style_sp]).tocsr()
X_test  = sparse.hstack([X_test_tfidf,  X_test_style_sp]).tocsr()

joblib.dump(tfidf, "light_gbm/tfidf_taskA_lightgbm_with_style.joblib")
tfidf_time = time.time() - tfidf_start
print("TF-IDF + style feature shapes:", X_train.shape, X_val.shape, X_test.shape)
print(f"Feature building completed in {tfidf_time:.2f} seconds")

print(">>> Training LightGBM...")
train_start = time.time()
clf = LGBMClassifier(
    objective="binary",
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
)

clf.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="binary_logloss",
    callbacks=[log_evaluation(50)],
)

joblib.dump(clf, "light_gbm/lightgbm_taskA_model_with_style.joblib")
train_time = time.time() - train_start
print(f"Training completed in {train_time:.2f} seconds")
print(">>> Finished training, evaluating...")

# Predictions and macro-F1
eval_start = time.time()
y_val_pred = clf.predict(X_val)
f1_macro = f1_score(y_val, y_val_pred, average="macro")
print("Macro-F1 (validation):", f1_macro)

print(classification_report(
    y_val,
    y_val_pred,
    target_names=["human (0)", "llm (1)"]
))

print(">>> Finished evaluating, predicting...")
y_test_pred = clf.predict(X_test)
f1_macro = f1_score(y_test, y_test_pred, average="macro")
print("Macro-F1 (test):", f1_macro)

print(classification_report(
    y_test,
    y_test_pred,
    target_names=["human (0)", "llm (1)"]
))
eval_time = time.time() - eval_start
print(f"Evaluation completed in {eval_time:.2f} seconds")

# Total execution time
total_time = time.time() - start_time
print("\n" + "="*50)
print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print("="*50)