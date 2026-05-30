"""
generate_predictions_json.py
=============================
Run EVA-02 inference on the ARMED validation set and save predictions
as a COCO-format JSON file ( coco_instances_results.json ).

This is the alternative to --eval-only when the evaluator is not saving
the JSON automatically.  The output JSON can then be passed to:
  - plot_confusion_matrix.py  (confusion matrix)
  - tools/visualize_json_results.py  (side-by-side pred vs GT images)

Outputs (in --out directory):
  coco_instances_results.json  ← COCO predictions file for confusion matrix
  eval_summary.txt             ← mAP / AP50 / per-class AP text summary

Usage (on Kaggle):
    python generate_predictions_json.py \\
        --config-file projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_l_4attn_1024_gun_ap50_v3.py \\
        --checkpoint  output_v3/model_0008000.pth \\
        --gt-json     /kaggle/input/datasets/ziaawan132/weaponsgunsdataset/OriginalDataset/OriginalDataset/annotations/armed_val.json \\
        --image-root  /kaggle/input/datasets/ziaawan132/weaponsgunsdataset/OriginalDataset/OriginalDataset \\
        --out         /kaggle/working/eval_output \\
        --use-ema \\
        --score-thresh 0.05
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import cv2

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data.detection_utils import read_image
from detectron2.modeling import ema
from detectron2.utils.logger import setup_logger


# ── Parse args ────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate coco_instances_results.json from EVA-02 checkpoint."
    )
    p.add_argument("--config-file",   required=True,
                   help="LazyConfig .py file used during training")
    p.add_argument("--checkpoint",    required=True,
                   help="Path to model checkpoint (.pth)")
    p.add_argument("--gt-json",       required=True,
                   help="COCO annotation JSON for the val/test split (armed_val.json)")
    p.add_argument("--image-root",    required=True,
                   help="Root folder containing the images listed in gt-json")
    p.add_argument("--out",           default="./eval_output",
                   help="Output folder (default: ./eval_output)")
    p.add_argument("--score-thresh",  type=float, default=0.05,
                   help="Keep predictions above this score for the JSON (default: 0.05). "
                        "Keep low here; you can filter later in the confusion matrix.")
    p.add_argument("--use-ema",       action="store_true",
                   help="Apply EMA weights (recommended – your model used EMA)")
    p.add_argument("--no-gpu",        action="store_true",
                   help="Force CPU inference")
    return p.parse_args()


# ── Model builder ─────────────────────────────────────────────────────────────
def build_model(cfg, checkpoint_path, use_ema, device):
    model = instantiate(cfg.model)
    model.to(device)
    model.eval()

    ema.may_build_model_ema(cfg, model)
    checkpointer = DetectionCheckpointer(model, **ema.may_get_ema_checkpointer(cfg, model))
    checkpointer.load(checkpoint_path)

    if use_ema and cfg.train.model_ema.enabled:
        ema.apply_model_ema(model)
        print("[INFO] EMA weights applied.")
    else:
        print("[INFO] Standard (non-EMA) weights used.")
    return model


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    logger = setup_logger(name="gen_pred_json")

    device = "cpu" if args.no_gpu else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load GT JSON ──────────────────────────────────────────────────────────
    logger.info(f"Loading GT JSON: {args.gt_json}")
    with open(args.gt_json) as f:
        gt_data = json.load(f)

    # Build image-id → file_name and image-id → (height, width)
    id_to_info = {
        img["id"]: {
            "file_name": img["file_name"],
            "height":    img["height"],
            "width":     img["width"],
        }
        for img in gt_data["images"]
    }

    # COCO dataset_category_id → contiguous 0-based id
    cats_sorted = sorted(gt_data["categories"], key=lambda c: c["id"])
    ds_id_to_cont = {c["id"]: i for i, c in enumerate(cats_sorted)}
    cont_to_ds_id = {v: k for k, v in ds_id_to_cont.items()}
    class_names   = [c["name"] for c in cats_sorted]
    logger.info(f"Classes: {class_names}   (dataset IDs: {list(ds_id_to_cont.keys())})")

    # ── Build model ───────────────────────────────────────────────────────────
    logger.info(f"Loading config : {args.config_file}")
    cfg = LazyConfig.load(args.config_file)
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = build_model(cfg, args.checkpoint, args.use_ema, device)

    # ── Inference over all val images ─────────────────────────────────────────
    coco_predictions = []   # list of COCO-format dicts
    n_images = len(id_to_info)
    logger.info(f"Running inference on {n_images} images ...")

    for idx, (img_id, info) in enumerate(id_to_info.items()):
        # Build full image path
        fname = info["file_name"]
        # file_name in COCO JSON can be relative (e.g. "images/img.jpg") or basename
        img_path = os.path.join(args.image_root, fname)
        if not os.path.isfile(img_path):
            # Try treating file_name as a basename only
            img_path = os.path.join(args.image_root, os.path.basename(fname))
        if not os.path.isfile(img_path):
            logger.warning(f"  [{idx+1}/{n_images}] Image not found, skipping: {fname}")
            continue

        img_bgr = read_image(img_path, format="BGR")
        h, w    = img_bgr.shape[:2]

        inp = [{
            "image":  torch.as_tensor(img_bgr.transpose(2, 0, 1).astype("float32")).to(device),
            "height": h,
            "width":  w,
        }]

        t0 = time.time()
        with torch.no_grad():
            outputs = model(inp)[0]
        elapsed = time.time() - t0

        instances = outputs["instances"].to("cpu")
        keep      = instances.scores >= args.score_thresh
        instances = instances[keep]
        n_det     = len(instances)

        if idx % 50 == 0 or idx == n_images - 1:
            logger.info(
                f"  [{idx+1}/{n_images}] {os.path.basename(fname)} "
                f"→ {n_det} detections  ({elapsed:.3f}s)"
            )

        if n_det == 0:
            continue

        boxes   = instances.pred_boxes.tensor.numpy()   # [N,4] xyxy
        scores  = instances.scores.numpy()
        classes = instances.pred_classes.numpy()         # contiguous 0-based

        for box, score, cls_cont in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box.tolist()
            # Convert back to COCO [x, y, w, h]
            coco_box = [x1, y1, x2 - x1, y2 - y1]
            # Convert contiguous id back to dataset category_id
            cat_id = cont_to_ds_id.get(int(cls_cont), int(cls_cont) + 1)

            coco_predictions.append({
                "image_id":   img_id,
                "category_id": cat_id,
                "bbox":       [round(v, 2) for v in coco_box],
                "score":      round(float(score), 6),
            })

    # ── Save predictions JSON ─────────────────────────────────────────────────
    pred_json_path = out_dir / "coco_instances_results.json"
    with open(pred_json_path, "w") as f:
        json.dump(coco_predictions, f)
    logger.info(f"\nSaved {len(coco_predictions)} predictions → {pred_json_path}")

    # ── Run COCO eval ─────────────────────────────────────────────────────────
    logger.info("\nRunning COCO evaluation ...")
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_gt   = COCO(args.gt_json)
        coco_dt   = coco_gt.loadRes(str(pred_json_path))
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Per-class AP
        summary_lines = []
        summary_lines.append("=" * 60)
        summary_lines.append("COCO Evaluation Results")
        summary_lines.append("=" * 60)
        metric_names = [
            "AP@[.5:.95]", "AP@50", "AP@75",
            "AP-small", "AP-medium", "AP-large",
            "AR@1",  "AR@10", "AR@100",
            "AR-small", "AR-medium", "AR-large",
        ]
        for name, val in zip(metric_names, coco_eval.stats):
            summary_lines.append(f"  {name:<18}: {val:.4f}")

        summary_lines.append("\nPer-category AP@50 / AP@[.5:.95]:")
        import numpy as np
        # coco_eval.eval["precision"] shape: [T, R, K, A, M]
        #   T = IoU thresholds (default: 0.50:0.05:0.95 → 10 values)
        #   R = recall thresholds (101 values)
        #   K = categories (in the order of params.catIds)
        #   A = area ranges (all, small, medium, large)
        #   M = max detections
        precisions = coco_eval.eval["precision"]   # shape [T, R, K, A, M]
        cat_ids    = coco_eval.params.catIds        # dataset-level category ids

        for i, cls_name in enumerate(class_names):
            # Find index of this class in the catIds list
            ds_cat_id = cats_sorted[i]["id"]
            if ds_cat_id not in cat_ids:
                summary_lines.append(f"  {cls_name:<12}: AP@50 = N/A  (not in eval catIds)")
                continue
            k_idx = cat_ids.index(ds_cat_id)

            # IoU threshold index for 0.50:  iouThrs = [.50, .55, ..., .95]
            iou_idx_50  = 0   # index 0 = IoU 0.50
            # area index 0 = 'all', max-det index 2 = 100 detections
            prec_ap50   = precisions[iou_idx_50, :, k_idx, 0, 2]
            prec_all    = precisions[:, :, k_idx, 0, 2]

            ap50 = float(np.mean(prec_ap50[prec_ap50 > -1])) if np.any(prec_ap50 > -1) else float("nan")
            ap   = float(np.mean(prec_all[prec_all > -1]))   if np.any(prec_all  > -1) else float("nan")
            summary_lines.append(
                f"  {cls_name:<12}: AP@50 = {ap50:.4f}   AP@[.5:.95] = {ap:.4f}"
            )


        summary_text = "\n".join(summary_lines)
        print("\n" + summary_text)

        summary_path = out_dir / "eval_summary.txt"
        with open(summary_path, "w") as f:
            f.write(summary_text + "\n")
        logger.info(f"Eval summary saved → {summary_path}")

    except ImportError:
        logger.warning("pycocotools not found — skipping mAP evaluation.")
        logger.warning("Install with: pip install pycocotools")

    # ── Final instructions ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("NEXT STEP — Generate Confusion Matrix:")
    logger.info(f"  python plot_confusion_matrix.py \\")
    logger.info(f"      --gt-json   {args.gt_json} \\")
    logger.info(f"      --pred-json {pred_json_path} \\")
    logger.info(f"      --out       {out_dir / 'plots'} \\")
    logger.info(f"      --iou-thresh 0.50 \\")
    logger.info(f"      --score-thresh 0.3 \\")
    logger.info(f"      --normalize")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
