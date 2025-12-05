import time
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel


def log(msg: str, start_time: float) -> None:
    elapsed = time.time() - start_time
    print(f"[{elapsed:6.1f}s] {msg}", flush=True)


def get_embeddings(texts, tokenizer, model, device, batch_size=32):
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


def main() -> None:
    start_time = time.time()
    base_dir = Path(__file__).resolve().parent

    # ------------------------------------------------------------------
    # 1. Load Qwen embedding model and LightGBM classifier
    # ------------------------------------------------------------------
    log("Loading Qwen embedding model...", start_time)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    model.to(device)
    model.eval()
    log("Loaded Qwen embedding model.", start_time)
    
    log("Loading LightGBM model from disk...", start_time)
    model_path = base_dir / "lightgbm_taskA_model_qwen_embedding.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    clf = joblib.load(model_path)
    log("Loaded LightGBM model.", start_time)

    # ------------------------------------------------------------------
    # 2. Load dataset (expects a 'test' split)
    # ------------------------------------------------------------------
    log("Loading SemEval Task A dataset...", start_time)
    ds = load_dataset("DaniilOr/SemEval-2026-Task13", "A")

    if "test" not in ds:
        raise ValueError(
            f"No 'test' split found in dataset. Available splits: {list(ds.keys())}"
        )

    test_ds = ds["test"]
    log(f"Loaded test split with {len(test_ds)} examples.", start_time)

    # ------------------------------------------------------------------
    # 3. Generate embeddings for test set
    # ------------------------------------------------------------------
    log("Generating embeddings for test code...", start_time)
    X_test_text = list(test_ds["code"])
    X_test = get_embeddings(X_test_text, tokenizer, model, device)
    log(f"Test embeddings shape: {X_test.shape}", start_time)

    # ------------------------------------------------------------------
    # 4. Predict
    # ------------------------------------------------------------------
    log("Running predictions on test set...", start_time)
    y_test_pred = clf.predict(X_test)
    log("Finished predictions.", start_time)

    # ------------------------------------------------------------------
    # 5. Optional: evaluate if labels are present (for local testing)
    # ------------------------------------------------------------------
    if "label" in test_ds.column_names:
        y_test_true = list(test_ds["label"])
        f1_macro = f1_score(y_test_true, y_test_pred, average="macro")
        log(f"Macro-F1 on test split (local labels): {f1_macro:.4f}", start_time)
    else:
        log("No 'label' column found in test split; skipping F1 evaluation.", start_time)

    # ------------------------------------------------------------------
    # 6. Build a submission / predictions CSV
    # ------------------------------------------------------------------
    log("Building predictions DataFrame...", start_time)
    # Use 'id' column if available, otherwise fall back to the row index
    if "id" in test_ds.column_names:
        ids = test_ds["id"]
    else:
        ids = list(range(len(test_ds)))

    pred_df = pd.DataFrame(
        {
            "id": ids,
            "label": y_test_pred,
        }
    )

    out_path = base_dir / "predictions_lightgbm_taskA_test_qwen_embedding.csv"
    pred_df.to_csv(out_path, index=False)
    log(f"Saved predictions to {out_path}", start_time)


if __name__ == "__main__":
    main()

