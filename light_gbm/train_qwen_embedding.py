from datasets import load_dataset
from sklearn.metrics import f1_score, classification_report
from lightgbm import LGBMClassifier
from lightgbm.callback import log_evaluation
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
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

print(">>> Loading Qwen embedding model...")
embedding_start = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
model.to(device)
model.eval()
print(f"Model loaded in {time.time() - embedding_start:.2f} seconds")

def get_embeddings(texts, batch_size=32):
    """Generate embeddings for a list of texts in batches."""
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling over token embeddings
            # Shape: (batch_size, seq_len, hidden_size)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(batch_embeddings.cpu().numpy())
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {i + len(batch_texts)}/{len(texts)} texts...")
    
    return np.vstack(embeddings)

print(">>> Generating embeddings for training set...")
embedding_start = time.time()
X_train = get_embeddings(X_train_text)
print(f"Training embeddings shape: {X_train.shape}")
print(f"Training embeddings generated in {time.time() - embedding_start:.2f} seconds")

print(">>> Generating embeddings for validation set...")
embedding_start = time.time()
X_val = get_embeddings(X_val_text)
print(f"Validation embeddings shape: {X_val.shape}")
print(f"Validation embeddings generated in {time.time() - embedding_start:.2f} seconds")

print(">>> Generating embeddings for test set...")
embedding_start = time.time()
X_test = get_embeddings(X_test_text)
print(f"Test embeddings shape: {X_test.shape}")
print(f"Test embeddings generated in {time.time() - embedding_start:.2f} seconds")

# Save the tokenizer and model info (optional, for reproducibility)
# Note: The model itself is large, so we just save the reference
embedding_info = {
    "model_name": "Qwen/Qwen3-Embedding-0.6B",
    "embedding_dim": X_train.shape[1]
}
joblib.dump(embedding_info, "light_gbm/qwen_embedding_info.joblib")
print(f"Embedding generation completed. Total time: {time.time() - embedding_start:.2f} seconds")

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

joblib.dump(clf, "light_gbm/lightgbm_taskA_model_qwen_embedding.joblib")
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

