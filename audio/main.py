import argparse
import csv
import json
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def normalize_label(s: str) -> str:
    return str(s).strip().lower()


def iter_audio_files(audio_data_dir: Path):
    exts = {".wav"}
    if not audio_data_dir.exists():
        return []
    files = []
    for p in sorted(audio_data_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def load_sidecar_label(audio_path: Path):
    """
    Optional sidecar:
      audio/data/<...>/name.wav -> audio/data/<...>/name.txt
    Sidecar contains ground-truth YAMNet display_name.
    """
    gt_path = audio_path.with_suffix(".txt")
    if not gt_path.exists() or not gt_path.is_file():
        return None
    try:
        text = gt_path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return None
    return text if text else None


def download_file(url: str, out_path: Path):
    import urllib.request

    out_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, out_path)


def generate_sine_wave(out_path: Path, duration_s: float = 2.0, sr: int = 16000, freq_hz: float = 440.0):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    sf.write(str(out_path), audio, sr)


def load_yamnet(model_url: str):
    import tensorflow as tf
    import tensorflow_hub as hub

    model = hub.load(model_url)

    # Class map is stored as a file inside the module.
    class_map_path = model.class_map_path().numpy()
    class_names = []
    with tf.io.gfile.GFile(class_map_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_names.append(row["display_name"])

    return model, class_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(Path("audio") / "data"))
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--download-sample", action="store_true")
    parser.add_argument(
        "--model-url",
        default="https://tfhub.dev/google/yamnet/1",
        help="TensorFlow Hub model URL for YAMNet.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--sample-url",
        default="https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav",
        help="Small sample audio file",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    yamnet, class_names = load_yamnet(args.model_url)

    audio_files = iter_audio_files(data_dir)
    use_local = len(audio_files) > 0
    results = {"mode": "local" if use_local else "fallback", "model_url": args.model_url, "top_k": args.top_k}

    t_total = time.time()

    if use_local:
        if args.max_samples > 0:
            audio_files = audio_files[: args.max_samples]

        preds = []
        y_true = []
        y_pred = []
        for audio_path in audio_files:
            t0 = time.time()

            y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
            # YAMNet expects float waveform at 16kHz.
            scores, embeddings, spectrogram = yamnet(y)
            scores_np = scores.numpy()  # [frames, 521]
            mean_scores = scores_np.mean(axis=0)  # [521]

            top_idx = np.argsort(mean_scores)[::-1][: max(1, args.top_k)]
            top_preds = []
            for idx in top_idx:
                label = class_names[idx] if idx < len(class_names) else f"class_{idx}"
                conf = float(mean_scores[idx])
                top_preds.append({"label": label, "confidence": conf})

            top1 = top_preds[0]["label"] if top_preds else None
            gt_label = load_sidecar_label(audio_path)

            if gt_label is not None and top1 is not None:
                y_true.append(normalize_label(gt_label))
                y_pred.append(normalize_label(top1))

            preds.append(
                {
                    "audio": str(audio_path),
                    "top_preds": top_preds,
                    "top1": top1,
                    "gt_present": gt_label is not None,
                    "gt_label": gt_label,
                    "inference_seconds": time.time() - t0,
                }
            )

        results["predictions"] = preds
        results["total_seconds"] = time.time() - t_total
        if y_true:
            from sklearn.metrics import accuracy_score

            results["accuracy"] = float(accuracy_score(y_true, y_pred))
            results["y_true"] = y_true
            results["y_pred"] = y_pred
    else:
        if args.no_fallback:
            results["warning"] = "No local audio found and fallback disabled."
        else:
            sample_path = out_dir / "audio_sample.wav"
            sample_generated = False

            if args.download_sample and not sample_path.exists():
                try:
                    download_file(args.sample_url, sample_path)
                except Exception as e:
                    results["download_error"] = str(e)

            if not sample_path.exists():
                generate_sine_wave(sample_path)
                sample_generated = True

            y, sr = librosa.load(str(sample_path), sr=16000, mono=True)
            scores, embeddings, spectrogram = yamnet(y)
            scores_np = scores.numpy()
            mean_scores = scores_np.mean(axis=0)

            top_idx = np.argsort(mean_scores)[::-1][: max(1, args.top_k)]
            top_preds = []
            for idx in top_idx:
                label = class_names[idx] if idx < len(class_names) else f"class_{idx}"
                conf = float(mean_scores[idx])
                top_preds.append({"label": label, "confidence": conf})

            results["sample_audio"] = str(sample_path)
            results["sample_generated"] = sample_generated
            results["top_preds"] = top_preds
            results["total_seconds"] = time.time() - t_total

    out_path = out_dir / f"audio_yamnet_{int(time.time())}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

