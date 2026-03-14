"""
Evaluation pipeline for YOLO-World (Option A — pretrained weights).
Computes: AP, AP50, AP75, APr (rare), APs/APm/APl, FPS
against COCO val2017 (or the mini subset from download_data.py).

Usage:
    python evaluate.py --ann data/coco_mini500.json --img-dir data/val2017
    python evaluate.py --ann data/annotations/instances_val2017.json --img-dir data/val2017 --model yolov8m-worldv2.pt
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLOWorld

# ── COCO 80-class names (in category_id order) ──────────────────────────────
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# COCO category IDs are not contiguous (e.g., skip 12, 26, …)
# This maps class index (0-79) → COCO category_id
COCO_CAT_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]


def load_model(weights: str) -> YOLOWorld:
    print(f"Loading model: {weights}")
    model = YOLOWorld(weights)
    model.set_classes(COCO_CLASSES)
    return model


def run_inference(model: YOLOWorld, img_dir: Path, image_infos: list,
                  conf_thresh: float = 0.001, iou_thresh: float = 0.7
                  ) -> tuple[list[dict], float]:
    """
    Run inference on all images and return (predictions_list, fps).
    predictions_list: COCO-format [{"image_id", "category_id", "bbox", "score"}, ...]
    """
    predictions = []
    total_time = 0.0
    missing = 0

    for info in tqdm(image_infos, desc="Inference"):
        img_path = img_dir / info["file_name"]
        if not img_path.exists():
            missing += 1
            continue

        t0 = time.perf_counter()
        results = model.predict(str(img_path), conf=conf_thresh, iou=iou_thresh,
                                verbose=False)
        total_time += time.perf_counter() - t0

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes  = result.boxes.xyxy.cpu().numpy()   # [N, 4] x1y1x2y2
        scores = result.boxes.conf.cpu().numpy()   # [N]
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)  # [N]

        for box, score, cls_idx in zip(boxes, scores, cls_ids):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            predictions.append({
                "image_id":   info["id"],
                "category_id": COCO_CAT_IDS[cls_idx],
                "bbox":       [round(float(x1), 2), round(float(y1), 2),
                               round(float(w), 2),  round(float(h), 2)],
                "score":      round(float(score), 4),
            })

    num_images = len(image_infos) - missing
    fps = num_images / total_time if total_time > 0 else 0.0
    if missing:
        print(f"[warn] {missing} images not found in {img_dir} — skipped")
    return predictions, fps


def run_coco_eval(coco_gt: COCO, predictions: list[dict],
                  iou_type: str = "bbox") -> dict:
    """Run COCOeval and return a results dict."""
    if not predictions:
        print("[warn] No predictions — returning zero metrics")
        return {}

    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    # stats indices (standard COCOeval order):
    # 0:AP, 1:AP50, 2:AP75, 3:APs, 4:APm, 5:APl
    # 6:AR@1, 7:AR@10, 8:AR@100, 9:ARs, 10:ARm, 11:ARl
    stats = evaluator.stats
    return {
        "AP":    round(float(stats[0]), 4),
        "AP50":  round(float(stats[1]), 4),
        "AP75":  round(float(stats[2]), 4),
        "APs":   round(float(stats[3]), 4),
        "APm":   round(float(stats[4]), 4),
        "APl":   round(float(stats[5]), 4),
    }


def compute_APr(coco_gt: COCO, predictions: list[dict],
                rare_max_instances: int = 10) -> float:
    """
    Compute APr: AP restricted to 'rare' categories.
    A category is 'rare' if it has fewer than `rare_max_instances` annotations
    in the evaluation set (mirrors the LVIS definition applied to COCO).
    """
    cat_counts = {}
    for ann in coco_gt.dataset["annotations"]:
        cat_counts[ann["category_id"]] = cat_counts.get(ann["category_id"], 0) + 1

    rare_cat_ids = {cid for cid, cnt in cat_counts.items() if cnt < rare_max_instances}
    if not rare_cat_ids:
        print("[info] No rare categories found with current threshold")
        return -1.0

    # Filter ground truth and predictions to rare categories only
    import copy
    gt_mini = copy.deepcopy(coco_gt.dataset)
    gt_mini["annotations"] = [a for a in gt_mini["annotations"]
                               if a["category_id"] in rare_cat_ids]
    gt_mini["categories"]   = [c for c in gt_mini["categories"]
                                if c["id"] in rare_cat_ids]

    coco_gt_rare = COCO()
    coco_gt_rare.dataset = gt_mini
    coco_gt_rare.createIndex()

    preds_rare = [p for p in predictions if p["category_id"] in rare_cat_ids]
    if not preds_rare:
        return 0.0

    coco_dt_rare = coco_gt_rare.loadRes(preds_rare)
    ev = COCOeval(coco_gt_rare, coco_dt_rare, iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return round(float(ev.stats[0]), 4)


def compute_fixed_ap(coco_gt: COCO, predictions: list[dict],
                     max_dets: int = 300) -> float:
    """
    Fixed AP: same as AP but with max detections per image set to 300
    (COCO default is 100; LVIS uses 300 to avoid penalizing large-vocabulary detectors).
    """
    if not predictions:
        return 0.0
    coco_dt = coco_gt.loadRes(predictions)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.params.maxDets = [1, 10, max_dets]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return round(float(ev.stats[0]), 4)


def print_results(metrics: dict, fps: float, APr: float, fixed_AP: float):
    sep = "─" * 45
    print(f"\n{sep}")
    print("  EVALUATION RESULTS")
    print(sep)
    print(f"  AP         (IoU 0.50:0.95) : {metrics.get('AP', -1):.4f}")
    print(f"  AP50       (IoU 0.50)      : {metrics.get('AP50', -1):.4f}")
    print(f"  AP75       (IoU 0.75)      : {metrics.get('AP75', -1):.4f}")
    print(f"  APs        (small objects) : {metrics.get('APs', -1):.4f}")
    print(f"  APm        (medium)        : {metrics.get('APm', -1):.4f}")
    print(f"  APl        (large)         : {metrics.get('APl', -1):.4f}")
    print(f"  APr        (rare cats)     : {APr:.4f}")
    print(f"  Fixed AP   (maxDets=300)   : {fixed_AP:.4f}")
    print(f"  FPS        (inference only): {fps:.1f}")
    print(sep)


def save_results(metrics: dict, fps: float, APr: float, fixed_AP: float,
                 out_path: str = "results.json"):
    output = {**metrics, "APr": APr, "FixedAP": fixed_AP, "FPS": round(fps, 2)}
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[ok] Results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLO-World COCO Evaluation")
    parser.add_argument("--ann",       default="data/coco_mini500.json",
                        help="Path to COCO annotation JSON")
    parser.add_argument("--img-dir",   default="data/val2017",
                        help="Directory containing val2017 images")
    parser.add_argument("--model",     default="yolov8s-worldv2.pt",
                        help="YOLO-World weights (auto-downloaded if not present)")
    parser.add_argument("--conf",      type=float, default=0.001,
                        help="Confidence threshold (low = more recall for AP)")
    parser.add_argument("--iou",       type=float, default=0.7,
                        help="NMS IoU threshold")
    parser.add_argument("--out",       default="results.json",
                        help="Output JSON for metric results")
    args = parser.parse_args()

    ann_path = Path(args.ann)
    img_dir  = Path(args.img_dir)
    assert ann_path.exists(), f"Annotation file not found: {ann_path}\nRun: python download_data.py"

    # ── Load GT ────────────────────────────────────────────────────────────
    print(f"Loading ground truth: {ann_path}")
    coco_gt = COCO(str(ann_path))
    image_infos = coco_gt.dataset["images"]
    print(f"  {len(image_infos)} images, "
          f"{len(coco_gt.dataset['annotations'])} annotations, "
          f"{len(coco_gt.dataset['categories'])} categories")

    # ── Load Model ─────────────────────────────────────────────────────────
    model = load_model(args.model)

    # ── Inference ──────────────────────────────────────────────────────────
    predictions, fps = run_inference(model, img_dir, image_infos,
                                     conf_thresh=args.conf, iou_thresh=args.iou)
    print(f"\nGenerated {len(predictions)} predictions at {fps:.1f} FPS")

    # ── Metrics ────────────────────────────────────────────────────────────
    print("\n── Standard AP ──")
    metrics  = run_coco_eval(coco_gt, predictions, iou_type="bbox")
    APr      = compute_APr(coco_gt, predictions)
    fixed_AP = compute_fixed_ap(coco_gt, predictions)

    print_results(metrics, fps, APr, fixed_AP)
    save_results(metrics, fps, APr, fixed_AP, out_path=args.out)


if __name__ == "__main__":
    main()
