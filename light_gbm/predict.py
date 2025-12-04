import time
from pathlib import Path

import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import f1_score


def log(msg: str, start_time: float) -> None:
    elapsed = time.time() - start_time
    print(f"[{elapsed:6.1f}s] {msg}", flush=True)


def main() -> None:
    start_time = time.time()
    base_dir = Path(__file__).resolve().parent

    # ------------------------------------------------------------------
    # 1. Load saved vectorizer and model
    # ------------------------------------------------------------------
    log("Loading TF-IDF vectorizer and LightGBM model from disk...", start_time)
    tfidf_path = base_dir / "tfidf_taskA_lightgbm.joblib"
    model_path = base_dir / "lightgbm_taskA_model.joblib"

    if not tfidf_path.exists():
        raise FileNotFoundError(f"TF-IDF file not found: {tfidf_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    tfidf = joblib.load(tfidf_path)
    clf = joblib.load(model_path)
    log("Loaded TF-IDF and model.", start_time)

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
    # 3. Build features for test set
    # ------------------------------------------------------------------
    log("Vectorizing test code with TF-IDF...", start_time)
    X_test_text = list(test_ds["code"])
    X_test = tfidf.transform(X_test_text)
    log(f"TF-IDF test matrix shape: {X_test.shape}", start_time)

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

    out_path = base_dir / "predictions_lightgbm_taskA_test.csv"
    pred_df.to_csv(out_path, index=False)
    log(f"Saved predictions to {out_path}", start_time)


if __name__ == "__main__":
    main()
