from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from lightgbm import LGBMClassifier
from lightgbm.callback import log_evaluation
import joblib
import time

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

print(">>> Building TF-IDF...")
tfidf_start = time.time()
# Character n-grams work very well for style detection in code
tfidf = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 6),      # 3–6 char n-grams
    min_df=3,                # ignore super-rare n-grams
    max_features=150_000     # adjust if you hit RAM issues
)

X_train = tfidf.fit_transform(X_train_text)
X_val   = tfidf.transform(X_val_text)
X_test  = tfidf.transform(X_test_text)

joblib.dump(tfidf, "light_gbm/tfidf_taskA_lightgbm.joblib")
tfidf_time = time.time() - tfidf_start
print("TF-IDF shapes:", X_train.shape, X_val.shape, X_test.shape)
print(f"TF-IDF building completed in {tfidf_time:.2f} seconds")

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

joblib.dump(clf, "light_gbm/lightgbm_taskA_model.joblib")
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