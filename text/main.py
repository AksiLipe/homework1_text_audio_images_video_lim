import argparse
import json
import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score
from transformers import pipeline


def normalize_label(s: str) -> str:
    return str(s).strip().lower()


def load_local_text_data(text_data_dir: Path):
    """
    Expected format:
      text/data/<label>/*.txt
    Each .txt file contains a single text example.
    """
    texts = []
    labels = []
    if not text_data_dir.exists():
        return texts, labels

    for label_dir in sorted([p for p in text_data_dir.iterdir() if p.is_dir()]):
        label = label_dir.name
        for txt_path in sorted(label_dir.glob("*.txt")):
            try:
                content = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                content = ""
            if content:
                texts.append(content)
                labels.append(label)
    return texts, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(Path("text") / "data"))
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument(
        "--model",
        default="tabularisai/multilingual-sentiment-analysis",
        help="HF sentiment model (multilingual, predicts positive/negative/neutral).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("text-classification", model=args.model, device=device)

    texts, labels = load_local_text_data(data_dir)
    use_local = len(texts) > 0

    results = {"mode": "local" if use_local else "fallback", "model": args.model, "device": device}
    t_total = time.time()

    if use_local:
        if args.max_samples > 0:
            texts = texts[: args.max_samples]
            labels = labels[: args.max_samples]

        y_true = [normalize_label(lab) for lab in labels]
        y_pred = []
        preds_raw = []
        for t in texts:
            pred = clf(t)[0]
            pred_label = pred.get("label", "")
            pred_score = pred.get("score", None)
            y_pred.append(normalize_label(pred_label))
            preds_raw.append({"pred_label": pred_label, "score": pred_score})

        results["accuracy"] = float(accuracy_score(y_true, y_pred)) if y_true else None
        results["y_true"] = y_true
        results["y_pred"] = y_pred
        results["predictions"] = preds_raw
    else:
        if args.no_fallback:
            results["warning"] = "No local data found and fallback disabled."
        else:
            # Fallback dataset: GLUE SST-2 (binary sentiment).
            from datasets import load_dataset

            ds = load_dataset("glue", "sst2", split="validation")
            n = min(len(ds), args.max_samples)
            texts = [ds[i]["sentence"] for i in range(n)]
            y_true = [int(ds[i]["label"]) for i in range(n)]  # 0=negative, 1=positive

            y_pred = []
            valid_mask = []
            for t in texts:
                pred = clf(t)[0]
                pred_label = normalize_label(pred.get("label", ""))
                pred_score = pred.get("score", None)

                if "positive" in pred_label:
                    y_pred.append(1)
                    valid_mask.append(True)
                elif "negative" in pred_label:
                    y_pred.append(0)
                    valid_mask.append(True)
                else:
                    # If model predicts neutral, it doesn't exist in SST-2.
                    y_pred.append(None)
                    valid_mask.append(False)

            valid_idx = [i for i, ok in enumerate(valid_mask) if ok]
            if valid_idx:
                y_true_valid = [y_true[i] for i in valid_idx]
                y_pred_valid = [y_pred[i] for i in valid_idx]
                results["accuracy"] = float(accuracy_score(y_true_valid, y_pred_valid))
            results["y_true"] = y_true
            results["y_pred"] = y_pred

    results["total_seconds"] = time.time() - t_total
    out_path = out_dir / f"text_sentiment_{int(time.time())}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

