import argparse
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def evaluate(pred_path, gold_path):
    pred_df = pd.read_csv(pred_path)
    gold_df = pd.read_parquet(gold_path)

    merged = pd.merge(gold_df, pred_df, on="ID")
    if merged.empty:
        raise ValueError("No matching IDs between prediction and gold files.")

    y_true = merged["label"]
    y_pred = merged["prediction"]


    macro_f1 = f1_score(y_true, y_pred, average="macro")
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    print("Evaluation Results")
    print("--------------------------")
    print(f"Main Metric - Macro F1:     {macro_f1:.4f}")
    print(f"Accuracy:                  {accuracy:.4f}")
    print(f"Macro Precision:           {macro_precision:.4f}")
    print(f"Macro Recall:              {macro_recall:.4f}")
    print("--------------------------")

    return macro_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to submission .csv")
    parser.add_argument("--gold", required=True, help="Path to gold labels .parquet (with columns: id,label)")
    args = parser.parse_args()

    evaluate(args.predictions, args.gold)
