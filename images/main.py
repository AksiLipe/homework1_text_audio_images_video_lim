import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image


def normalize_label(s: str) -> str:
    return str(s).strip().lower()


def load_local_images(images_data_dir: Path):
    """
    Expected format:
      images/data/<label>/*.jpg|*.jpeg|*.png

    Metric:
      We take the highest-confidence detected object label from YOLOv5 and
      compare it with <label>.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = []
    labels = []
    if not images_data_dir.exists():
        return images, labels

    for label_dir in sorted([p for p in images_data_dir.iterdir() if p.is_dir()]):
        label = label_dir.name
        for p in sorted(label_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts:
                images.append(p)
                labels.append(label)
    return images, labels


def download_file(url: str, out_path: Path):
    import urllib.request

    out_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(Path("images") / "data"))
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--download-sample", action="store_true")
    parser.add_argument("--model-name", default="yolov5s", help="YOLOv5 model variant (e.g., yolov5s).")
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument(
        "--sample-url",
        default="https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg",
        help="Sample image if local dataset is empty.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLOv5 from PyTorch Hub.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # trust_repo=True чтобы не зависеть от интерактивного подтверждения при загрузке кода YOLOv5
    model = torch.hub.load("ultralytics/yolov5", args.model_name, pretrained=True, trust_repo=True)
    if device == "cuda":
        model = model.to("cuda")
    model.eval()

    local_images, local_labels = load_local_images(data_dir)
    use_local = len(local_images) > 0

    results = {
        "mode": "local" if use_local else "fallback",
        "model": f"ultralytics/yolov5:{args.model_name}",
        "device": device,
        "conf_thres": args.conf_thres,
    }

    t_total = time.time()

    if use_local:
        if args.max_samples > 0:
            local_images = local_images[: args.max_samples]
            local_labels = local_labels[: args.max_samples]

        y_true = []
        y_pred = []
        preds = []
        for img_path, true_label in zip(local_images, local_labels):
            t0 = time.time()
            img = Image.open(img_path).convert("RGB")
            out = model(img)
            det = out.xyxy[0]  # Nx6: x1,y1,x2,y2,conf,cls
            if det is None or len(det) == 0:
                best_label = None
                best_conf = None
                top_dets = []
            else:
                det_np = det.detach().cpu().numpy()
                # Filter by confidence threshold
                det_np = det_np[det_np[:, 4] >= args.conf_thres]
                if len(det_np) == 0:
                    best_label = None
                    best_conf = None
                    top_dets = []
                else:
                    # Sort by confidence desc
                    det_np = det_np[np.argsort(det_np[:, 4])[::-1]]
                    best = det_np[0]
                    cls_idx = int(best[5])
                    best_label = model.names[cls_idx]
                    best_conf = float(best[4])

                    top_dets = []
                    for row in det_np[:5]:
                        cls_idx = int(row[5])
                        top_dets.append(
                            {
                                "label": model.names[cls_idx],
                                "confidence": float(row[4]),
                            }
                        )

            preds.append(
                {
                    "image": str(img_path),
                    "true_label": true_label,
                    "pred_top1": best_label,
                    "pred_top1_conf": best_conf,
                    "top5": top_dets,
                    "inference_seconds": time.time() - t0,
                }
            )

            y_true.append(normalize_label(true_label))
            y_pred.append(normalize_label(best_label) if best_label is not None else "__none__")

        from sklearn.metrics import accuracy_score

        results["accuracy"] = float(accuracy_score(y_true, y_pred)) if y_true else None
        results["y_true"] = y_true
        results["y_pred"] = y_pred
        results["predictions"] = preds
    else:
        if args.no_fallback:
            results["warning"] = "No local images found and fallback disabled."
        else:
            sample_path = out_dir / "images_sample.jpg"
            if args.download_sample and not sample_path.exists():
                try:
                    download_file(args.sample_url, sample_path)
                except Exception as e:
                    results["download_error"] = str(e)

            if not sample_path.exists():
                results["warning"] = "No local images found and sample image not downloaded."
            else:
                img = Image.open(sample_path).convert("RGB")
                out = model(img)
                det = out.xyxy[0]
                top_dets = []
                if det is not None and len(det) > 0:
                    det_np = det.detach().cpu().numpy()
                    det_np = det_np[det_np[:, 4] >= args.conf_thres]
                    if len(det_np) > 0:
                        det_np = det_np[np.argsort(det_np[:, 4])[::-1]]
                        for row in det_np[:5]:
                            cls_idx = int(row[5])
                            top_dets.append({"label": model.names[cls_idx], "confidence": float(row[4])})
                results["sample_image"] = str(sample_path)
                results["top5"] = top_dets
                results["predictions"] = [{"sample_image": str(sample_path), "top5": top_dets}]

    results["total_seconds"] = time.time() - t_total
    out_path = out_dir / f"images_yolov5_{int(time.time())}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

