"""
infer_test_images.py
====================
Run EVA-02 Cascade Mask R-CNN inference on a folder of test images.
No annotation JSON required — pure prediction mode.

Outputs:
  - Visualized images with bounding boxes, class labels, and confidence scores
  - A summary CSV listing every detected box per image

Usage (on Kaggle / Linux):
    python infer_test_images.py \
        --config-file projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_l_4attn_1024_gun_ap50_v3.py \
        --checkpoint output_v3/model_0008000.pth \
        --input /kaggle/input/datasets/naumanlatif1234/testing-dataset/TestingImages \
        --output /kaggle/working/predictions_out \
        --score-thresh 0.3

Arguments:
    --config-file   : Path to the LazyConfig .py file used during training.
    --checkpoint    : Path to the model checkpoint (.pth).
    --input         : Folder containing test images (jpg/jpeg/png/bmp).
    --output        : Folder where visualized images will be saved.
    --score-thresh  : Minimum confidence score to show a box (default: 0.3).
    --use-ema       : Flag. Pass this to apply EMA weights (recommended if
                      your model was trained with EMA enabled).
    --no-gpu        : Flag. Force CPU inference (slow but works without GPU).
"""

import argparse
import csv
import glob
import os
import sys
import time

import cv2
import torch

# ── Detectron2 imports ────────────────────────────────────────────────────────
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data.detection_utils import read_image
from detectron2.modeling import ema
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog

# ── Supported image extensions ────────────────────────────────────────────────
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

# ── ARMED class names (fallback if metadata is not loaded from JSON) ──────────
ARMED_CLASSES = ["Armed", "Unarmed", "Gun"]

# ── Box colors per class (BGR) ────────────────────────────────────────────────
BOX_COLORS = {
    0: (0, 140, 255),    # Armed  → orange
    1: (0, 200, 0),      # Unarmed → green
    2: (0, 0, 255),      # Gun    → red
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="EVA-02 inference on test images (no annotations needed)."
    )
    parser.add_argument(
        "--config-file", required=True,
        help="Path to the LazyConfig .py file (e.g. projects/ViTDet/configs/.../gun_ap50_v3.py)"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the model checkpoint .pth file"
    )
    parser.add_argument(
        "--input", required=True,
        help="Folder containing test images"
    )
    parser.add_argument(
        "--output", required=True,
        help="Folder to save visualized output images"
    )
    parser.add_argument(
        "--score-thresh", type=float, default=0.3,
        help="Minimum confidence score to display a bounding box (default: 0.3)"
    )
    parser.add_argument(
        "--use-ema", action="store_true",
        help="Apply EMA weights for inference (recommended if trained with EMA)"
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="Force CPU inference"
    )
    return parser.parse_args()


def collect_images(input_dir):
    """Collect all image paths from the input directory."""
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(input_dir, ext)))
        paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    paths = sorted(set(paths))
    return paths


def build_model(cfg, checkpoint_path, use_ema, device):
    """Instantiate model, load checkpoint, optionally apply EMA."""
    model = instantiate(cfg.model)
    model.to(device)
    model.eval()

    # Build EMA state if enabled in config
    ema.may_build_model_ema(cfg, model)

    # Load checkpoint weights
    checkpointer = DetectionCheckpointer(
        model,
        **ema.may_get_ema_checkpointer(cfg, model)
    )
    checkpointer.load(checkpoint_path)

    # Apply EMA weights if requested
    if use_ema and cfg.train.model_ema.enabled:
        ema.apply_model_ema(model)
        print("[INFO] EMA weights applied.")
    else:
        print("[INFO] Using standard (non-EMA) weights.")

    return model


def preprocess_image(img_bgr, device):
    """Convert BGR numpy image to model input dict."""
    h, w = img_bgr.shape[:2]
    img_tensor = torch.as_tensor(
        img_bgr.transpose(2, 0, 1).astype("float32")
    ).to(device)
    return [{"image": img_tensor, "height": h, "width": w}]


def draw_boxes_cv2(img_bgr, instances, class_names, score_thresh):
    """
    Draw bounding boxes directly with OpenCV.
    Falls back to this if Detectron2 Visualizer metadata is incomplete.
    """
    out = img_bgr.copy()
    boxes   = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
    scores  = instances.scores.numpy()            if instances.has("scores")    else []
    classes = instances.pred_classes.numpy()      if instances.has("pred_classes") else []

    for box, score, cls in zip(boxes, scores, classes):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = map(int, box)
        color = BOX_COLORS.get(int(cls), (255, 255, 255))
        label = f"{class_names[int(cls)]}: {score:.2f}"

        # Draw rectangle
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    args = parse_args()
    logger = setup_logger(name="infer_test_images")

    # ── Device ────────────────────────────────────────────────────────────────
    device = "cpu" if args.no_gpu else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── Load config ───────────────────────────────────────────────────────────
    logger.info(f"Loading config: {args.config_file}")
    cfg = LazyConfig.load(args.config_file)

    # ── Build & load model ────────────────────────────────────────────────────
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = build_model(cfg, args.checkpoint, args.use_ema, device)

    # ── Class names ───────────────────────────────────────────────────────────
    # Try to get class names from registered metadata (armed_val), else use defaults
    try:
        meta = MetadataCatalog.get("armed_val")
        class_names = meta.thing_classes
        logger.info(f"Classes from metadata: {class_names}")
    except Exception:
        class_names = ARMED_CLASSES
        logger.info(f"Using default class names: {class_names}")

    # ── Collect images ────────────────────────────────────────────────────────
    image_paths = collect_images(args.input)
    if not image_paths:
        logger.error(f"No images found in: {args.input}")
        sys.exit(1)
    logger.info(f"Found {len(image_paths)} images in: {args.input}")

    # ── Output folder ─────────────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    csv_path = os.path.join(args.output, "predictions.csv")

    # ── Inference loop ────────────────────────────────────────────────────────
    total_detections = 0
    csv_rows = []

    for i, img_path in enumerate(image_paths):
        img_bgr = read_image(img_path, format="BGR")
        inputs  = preprocess_image(img_bgr, device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model(inputs)[0]
        elapsed = time.time() - t0

        # Filter by score threshold
        instances = outputs["instances"].to("cpu")
        keep      = instances.scores >= args.score_thresh
        instances = instances[keep]

        n_det = len(instances)
        total_detections += n_det
        logger.info(
            f"[{i+1}/{len(image_paths)}] {os.path.basename(img_path)} "
            f"→ {n_det} detections  ({elapsed:.3f}s)"
        )

        # ── Draw boxes ────────────────────────────────────────────────────────
        vis_img = draw_boxes_cv2(img_bgr, instances, class_names, score_thresh=0.0)

        # Save visualized image
        out_filename = os.path.join(args.output, os.path.basename(img_path))
        cv2.imwrite(out_filename, vis_img)

        # ── Collect CSV rows ──────────────────────────────────────────────────
        if n_det == 0:
            csv_rows.append({
                "image": os.path.basename(img_path),
                "class": "none", "score": "", "x1": "", "y1": "", "x2": "", "y2": ""
            })
        else:
            boxes   = instances.pred_boxes.tensor.numpy()
            scores  = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            for box, score, cls in zip(boxes, scores, classes):
                csv_rows.append({
                    "image": os.path.basename(img_path),
                    "class": class_names[int(cls)],
                    "score": f"{score:.4f}",
                    "x1": int(box[0]), "y1": int(box[1]),
                    "x2": int(box[2]), "y2": int(box[3]),
                })

    # ── Save CSV ──────────────────────────────────────────────────────────────
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image","class","score","x1","y1","x2","y2"])
        writer.writeheader()
        writer.writerows(csv_rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Done. {len(image_paths)} images processed.")
    logger.info(f"Total detections (score >= {args.score_thresh}): {total_detections}")
    logger.info(f"Visualized images saved to : {args.output}")
    logger.info(f"Predictions CSV saved to   : {csv_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
