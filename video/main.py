import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score
from transformers import AutoImageProcessor, SiglipForImageClassification


def normalize_label(s: str) -> str:
    return str(s).strip().lower()


def iter_videos(video_data_dir: Path):
    exts = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    items = []
    if not video_data_dir.exists():
        return items
    for label_dir in sorted([p for p in video_data_dir.iterdir() if p.is_dir()]):
        label = label_dir.name
        for p in sorted(label_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts:
                items.append((p, label))
    return items


def download_file(url: str, out_path: Path):
    import urllib.request

    out_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, out_path)


def sample_frame_indices(frame_count: int, num_frames: int):
    if frame_count <= 0:
        return []
    if frame_count <= num_frames:
        return list(range(frame_count))
    return list(np.linspace(0, frame_count - 1, num_frames, dtype=int))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(Path("video") / "data"))
    parser.add_argument("--max-samples", type=int, default=10, help="max number of videos")
    parser.add_argument("--num-frames", type=int, default=16, help="frames sampled per video")
    parser.add_argument(
        "--model",
        default="prithivMLmods/Hand-Gesture-19",
        help="Hand-gesture classification model (SigLIP).",
    )
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--download-sample", action="store_true")
    parser.add_argument(
        "--sample-url",
        default="https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
        help="Sample video if local dataset is empty.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SiglipForImageClassification.from_pretrained(args.model).to(device)
    model.eval()

    local_items = iter_videos(data_dir)
    use_local = len(local_items) > 0

    results = {"mode": "local" if use_local else "fallback", "model": args.model, "device": device}
    per_video = []
    y_true = []
    y_pred = []

    videos = []
    if use_local:
        videos = [(p, lbl) for p, lbl in local_items]
        if args.max_samples > 0:
            videos = videos[: args.max_samples]
    else:
        if args.no_fallback:
            results["warning"] = "No local videos found and fallback disabled."
        else:
            sample_path = out_dir / "video_sample.mp4"
            if args.download_sample and not sample_path.exists():
                try:
                    download_file(args.sample_url, sample_path)
                except Exception as e:
                    results["download_error"] = str(e)
            if not sample_path.exists():
                results["warning"] = "No local videos found and sample video not downloaded."
            else:
                videos = [(sample_path, None)]

    t_total = time.time()

    for video_path, true_label in videos:
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = sample_frame_indices(frame_count, args.num_frames)
        if not idxs:
            cap.release()
            continue

        frame_labels = []
        frame_confs = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame_bgr = cap.read()
            if not ok:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_img = Image.fromarray(frame_rgb)

            inputs = processor(images=frame_img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=1)
                pred_idx = int(torch.argmax(probs, dim=1).item())
                conf = float(probs[0, pred_idx].item())

            frame_labels.append(model.config.id2label[pred_idx])
            frame_confs.append(conf)

        cap.release()
        if not frame_labels:
            continue

        vote_counts = Counter(frame_labels)
        pred_label = vote_counts.most_common(1)[0][0]
        avg_conf = float(np.mean(frame_confs)) if frame_confs else None

        if true_label is not None:
            y_true.append(normalize_label(true_label))
            y_pred.append(normalize_label(pred_label))

        per_video.append(
            {
                "video": str(video_path),
                "true_label": true_label,
                "pred_label": pred_label,
                "avg_frame_confidence": avg_conf,
                "frame_votes": dict(vote_counts),
            }
        )

    results["total_seconds"] = time.time() - t_total
    results["per_video"] = per_video

    if y_true:
        results["accuracy"] = float(accuracy_score(y_true, y_pred))
        results["y_true"] = y_true
        results["y_pred"] = y_pred

    out_path = out_dir / f"video_gesture_{int(time.time())}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

