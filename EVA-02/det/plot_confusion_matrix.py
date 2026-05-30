"""
plot_confusion_matrix.py
========================
Build and plot a confusion matrix for EVA-02 / ARMED detection results.

HOW IT WORKS
------------
Object detection confusion matrices differ from classification ones.
For each ground-truth (GT) box we find the highest-IoU predicted box:
  - If IoU >= iou_thresh AND predicted class == GT class  → True Positive (diagonal)
  - If IoU >= iou_thresh AND predicted class != GT class  → Confusion (off-diagonal)
  - If no pred box has IoU >= iou_thresh                  → False Negative (→ Background column)
For each predicted box that was NOT matched to any GT box:
  - score >= score_thresh → False Positive (Background → predicted class row)

This produces a (N+1) × (N+1) matrix where the last row/column is "Background".

INPUTS REQUIRED
---------------
1. --gt-json   : COCO-format annotations JSON  (armed_val.json)
2. --pred-json : Predictions JSON saved by Detectron2 after --eval-only
                 Default path: <output_dir>/inference/coco_instances_results.json

Usage:
    python plot_confusion_matrix.py \
        --gt-json   /kaggle/input/.../annotations/armed_val.json \
        --pred-json /kaggle/working/output_v3_eval/inference/coco_instances_results.json \
        --out       /kaggle/working/plots \
        --iou-thresh  0.50 \
        --score-thresh 0.3 \
        --normalize
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── Dark theme (matches plot_metrics.py style) ────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d2e",
    "axes.edgecolor":   "#3a3f5c",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "font.family":      "DejaVu Sans",
    "font.size":        12,
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
})

CLASS_NAMES = ["Armed", "Unarmed", "Gun"]   # contiguous ids 0, 1, 2
BACKGROUND  = "Background"


# ── IoU helper ────────────────────────────────────────────────────────────────
def box_iou(boxA, boxB):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)


def xywh_to_xyxy(box):
    """Convert COCO [x,y,w,h] → [x1,y1,x2,y2]."""
    x, y, w, h = box
    return [x, y, x + w, y + h]


# ── Load data ─────────────────────────────────────────────────────────────────
def load_gt(json_path):
    """
    Returns:
        id_to_contiguous : dict  {dataset_category_id → contiguous 0-based id}
        gt_by_image      : dict  {image_id → list of {cat_id_contiguous, bbox_xyxy}}
    """
    with open(json_path) as f:
        data = json.load(f)

    # Build category id → contiguous id mapping (sorted by dataset id)
    cats = sorted(data["categories"], key=lambda c: c["id"])
    id_to_contiguous = {c["id"]: i for i, c in enumerate(cats)}
    class_names = [c["name"] for c in cats]

    gt_by_image = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        gt_by_image.setdefault(img_id, []).append({
            "cat": id_to_contiguous[ann["category_id"]],
            "box": xywh_to_xyxy(ann["bbox"]),
            "matched": False,
        })
    return id_to_contiguous, class_names, gt_by_image


def load_preds(json_path, score_thresh):
    """
    Returns:
        pred_by_image : dict  {image_id → list of {cat_id_contiguous, bbox_xyxy, score}}
    Note: Detectron2 prediction JSON uses contiguous (0-based) category_id directly.
    """
    with open(json_path) as f:
        preds = json.load(f)

    pred_by_image = {}
    for p in preds:
        if p["score"] < score_thresh:
            continue
        img_id = p["image_id"]
        pred_by_image.setdefault(img_id, []).append({
            "cat": p["category_id"],       # already contiguous in Detectron2 output
            "box": xywh_to_xyxy(p["bbox"]),
            "score": p["score"],
            "matched": False,
        })
    return pred_by_image


# ── Build confusion matrix ────────────────────────────────────────────────────
def build_confusion_matrix(gt_by_image, pred_by_image, n_classes, iou_thresh):
    """
    Returns a (n_classes+1) × (n_classes+1) confusion matrix.
    Rows = Ground Truth class (last row = Background / FP source).
    Cols = Predicted class   (last col = Background / FN sink).
    """
    n = n_classes + 1   # +1 for Background
    cm = np.zeros((n, n), dtype=np.int64)

    all_image_ids = set(gt_by_image.keys()) | set(pred_by_image.keys())

    for img_id in all_image_ids:
        gts   = gt_by_image.get(img_id, [])
        preds = pred_by_image.get(img_id, [])

        # Sort predictions by score descending (greedy matching)
        preds_sorted = sorted(preds, key=lambda p: p["score"], reverse=True)

        # For each GT box, find the best matching prediction
        for gt in gts:
            best_iou  = 0.0
            best_pred = None

            for pred in preds_sorted:
                if pred["matched"]:
                    continue
                iou = box_iou(gt["box"], pred["box"])
                if iou > best_iou:
                    best_iou  = iou
                    best_pred = pred

            if best_iou >= iou_thresh and best_pred is not None:
                # Matched — record GT class vs predicted class
                cm[gt["cat"]][best_pred["cat"]] += 1
                best_pred["matched"] = True
            else:
                # No match → False Negative (GT → Background column)
                cm[gt["cat"]][n_classes] += 1

        # Any unmatched prediction → False Positive (Background row → pred class)
        for pred in preds_sorted:
            if not pred["matched"]:
                cm[n_classes][pred["cat"]] += 1

    return cm


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_cm(cm, class_names, normalize, out_path, iou_thresh, score_thresh):
    labels = class_names + [BACKGROUND]
    n = len(labels)

    if normalize:
        # Normalize by row (per GT class)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1          # avoid division by zero
        cm_plot = cm.astype(float) / row_sums
        fmt_int  = False
        cbar_label = "Fraction of GT instances"
        title_suffix = " (Row-Normalized)"
    else:
        cm_plot = cm.astype(float)
        fmt_int  = True
        cbar_label = "Count"
        title_suffix = " (Counts)"

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor("#0f1117")

    # Custom colormap: dark → vivid blue
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "dark_blue", ["#0f1117", "#1a4a8a", "#58a6ff", "#cae8ff"], N=256
    )

    im = ax.imshow(cm_plot, interpolation="nearest", cmap=cmap,
                   vmin=0, vmax=(1.0 if normalize else cm_plot.max()))

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, color="#c9d1d9")
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8b949e")

    # Tick labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)

    # Cell annotations
    thresh = cm_plot.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = cm_plot[i, j]
            if normalize:
                text = f"{val:.2f}"
            else:
                text = f"{int(val)}"
            color = "#0f1117" if val > thresh else "#c9d1d9"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=11, color=color, fontweight="bold")

    ax.set_xlabel("Predicted Class", fontsize=12, labelpad=10)
    ax.set_ylabel("Ground Truth Class", fontsize=12, labelpad=10)
    ax.set_title(
        f"FIDASS Detection Confusion Matrix{title_suffix}\n"
        f"IoU≥{iou_thresh}  |  Score≥{score_thresh}",
        fontsize=13, pad=14
    )

    # Highlight diagonal with a white border
    for k in range(min(n - 1, n)):   # skip Background diagonal
        rect = plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                              linewidth=2, edgecolor="#3fb950", facecolor="none")
        ax.add_patch(rect)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Summary stats ─────────────────────────────────────────────────────────────
def print_summary(cm, class_names):
    n_cls = len(class_names)
    labels = class_names + [BACKGROUND]

    print("\n── Confusion Matrix (raw counts) ──")
    header = f"{'GT \\ Pred':<12}" + "".join(f"{l:>12}" for l in labels)
    print(header)
    print("-" * len(header))
    for i, row_label in enumerate(labels):
        row = "".join(f"{cm[i,j]:>12}" for j in range(len(labels)))
        print(f"{row_label:<12}{row}")

    print("\n── Per-class metrics ──")
    print(f"{'Class':<12}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print("-" * 46)
    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[n_cls, i]               # BG predicted as this class
        fn = cm[i, n_cls]               # this class predicted as BG
        # Also count cross-class confusions
        fp += cm[:n_cls, i].sum() - tp  # other GT classes predicted as this
        fn += cm[i, :n_cls].sum() - tp  # this GT class predicted as others

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        print(f"{cls:<12}  {precision:>10.3f}  {recall:>8.3f}  {f1:>8.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Confusion matrix for EVA-02 ARMED detection."
    )
    parser.add_argument(
        "--gt-json", required=True,
        help="COCO-format ground-truth annotation JSON (e.g. armed_val.json)"
    )
    parser.add_argument(
        "--pred-json", required=True,
        help="Detectron2 prediction JSON (coco_instances_results.json)"
    )
    parser.add_argument(
        "--out", default="./plots",
        help="Output directory for the confusion matrix PNG"
    )
    parser.add_argument(
        "--iou-thresh", type=float, default=0.50,
        help="IoU threshold to consider a detection as matched (default: 0.50)"
    )
    parser.add_argument(
        "--score-thresh", type=float, default=0.3,
        help="Minimum prediction confidence score (default: 0.3)"
    )
    parser.add_argument(
        "--normalize", action="store_true",
        help="Row-normalize the matrix (show fractions instead of counts)"
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading GT annotations : {args.gt_json}")
    id_to_contiguous, class_names, gt_by_image = load_gt(args.gt_json)
    n_gt_boxes = sum(len(v) for v in gt_by_image.values())
    print(f"  Classes     : {class_names}")
    print(f"  Images      : {len(gt_by_image)}")
    print(f"  GT boxes    : {n_gt_boxes}")

    print(f"\nLoading predictions    : {args.pred_json}")
    pred_by_image = load_preds(args.pred_json, args.score_thresh)
    n_pred_boxes = sum(len(v) for v in pred_by_image.values())
    print(f"  Pred boxes (score>={args.score_thresh:.2f}) : {n_pred_boxes}")

    print(f"\nBuilding confusion matrix (IoU>={args.iou_thresh}) ...")
    cm = build_confusion_matrix(
        gt_by_image, pred_by_image, len(class_names), args.iou_thresh
    )

    print_summary(cm, class_names)

    # Save counts version
    out_counts = out_dir / "confusion_matrix_counts.png"
    plot_cm(cm, class_names, normalize=False,
            out_path=out_counts,
            iou_thresh=args.iou_thresh, score_thresh=args.score_thresh)

    # Save normalized version
    out_norm = out_dir / "confusion_matrix_normalized.png"
    plot_cm(cm, class_names, normalize=True,
            out_path=out_norm,
            iou_thresh=args.iou_thresh, score_thresh=args.score_thresh)

    print("\nDone!")


if __name__ == "__main__":
    main()
