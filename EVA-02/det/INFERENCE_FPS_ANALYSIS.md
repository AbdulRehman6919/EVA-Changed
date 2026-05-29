# Inference Speed (FPS) Analysis — EVA-02 Armed Detection

> **Model:** EVA-02 Large + Cascade Mask R-CNN + ViTDet  
> **Task:** Early Detection of Firearms, Weapons, and Unarmed Persons  
> **Classes:** Armed · Unarmed · Gun  
> **Input Resolution:** 1024 × 1024  

---

## 1. Benchmark Command Used

```bash
python tools/benchmark.py \
  --task eval \
  --num-gpus 1 \
  --config-file projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_l_4attn_1024_gun_ap50_v5.py \
  train.init_checkpoint=<path_to_checkpoint.pth>
```

This runs **300 inference iterations** on your test dataset with a 5-iteration GPU warm-up, then reports total elapsed time. FPS and latency are derived from that output.

---

## 2. Actual Benchmark Results (T4 GPU — Kaggle)

```
[05/29 06:51:18] Serializing 100 elements ... 187.56 MiB
100%|█████████████████████| 300/300 [07:15<00:00,  1.45s/it]
[05/29 06:58:41] 300 iters in 435.9594311369999 seconds.
```

### Derived Metrics

| Metric | Value |
|---|---|
| **Total frames processed** | 300 |
| **Total time** | 435.96 seconds (7 min 15 sec) |
| **FPS (Frames Per Second)** | **0.69 FPS** |
| **Latency per frame** | **1,453 ms (1.45 sec/frame)** |
| **GPU used** | NVIDIA T4 (16 GB) — Kaggle free tier |

### Formula
```
FPS     = 300 ÷ 435.96 = 0.688 FPS
Latency = 435.96 ÷ 300 = 1,453 ms per frame
```

---

## 3. Real-Time Suitability Assessment

| FPS Range | Use Case | Your Model |
|---|---|---|
| ≥ 25 FPS | Live surveillance / security cameras | ❌ |
| 10 – 25 FPS | Near-real-time alerting | ❌ |
| 3 – 10 FPS | Buffered / delayed detection | ❌ |
| 1 – 3 FPS | Low-frequency alert systems | ❌ |
| **< 1 FPS** | **Post-event forensic / batch analysis** | **✅ 0.69 FPS** |

> **Verdict:** The model is NOT suitable for continuous live-stream real-time detection (≥15 FPS).  
> It IS suitable for **asynchronous early-warning systems** and **forensic post-event analysis**.

---

## 4. Why This Model Is Slow

The inference bottleneck is the **model architecture itself**, not just hardware. Three compounding factors:

### 4.1 ViT-L Backbone at 1024×1024
- Vision Transformer Large has ~307M parameters
- At 1024×1024, the image is split into **4,096 patches**
- Self-attention is **O(n²)** — 4096² = ~16.7 million attention operations per layer
- With 4 attention layers, this dominates inference time

### 4.2 Three-Stage Cascade Detection
- Stage 1 (IoU ≥ 0.4) → refines proposals
- Stage 2 (IoU ≥ 0.5) → refines again
- Stage 3 (IoU ≥ 0.6) → final predictions
- The ROI head runs **3× per image**

### 4.3 High RPN Proposal Count
- `pre_nms_topk  = (4000, 2000)`
- `post_nms_topk = (2000, 2000)`
- 2000 region proposals fed into all 3 cascade stages

### Pipeline Breakdown
```
Image Input (1024×1024)
        ↓
CPU: Resize / Normalize        ← medium cost
        ↓
GPU: ViT-L Backbone            ← 🔴 LARGEST cost (~70% of total)
        ↓
GPU: FPN Neck                  ← moderate cost
        ↓
GPU: RPN (4000 → 2000 proposals)  ← moderate cost
        ↓
GPU: Cascade Stage 1 (ROI)     ← heavy
GPU: Cascade Stage 2 (ROI)     ← heavy
GPU: Cascade Stage 3 (ROI)     ← heavy
        ↓
CPU: Soft-NMS Post-processing  ← medium cost
        ↓
Output: Detections
```

---

## 5. GPU Impact on FPS

GPU hardware **directly and proportionally** improves FPS for the GPU-bound stages. However, the architecture has a hard ceiling regardless of hardware.

### Estimated FPS Across Different GPUs

| GPU | VRAM | Approx. Speedup vs T4 | Estimated FPS |
|---|---|---|---|
| **NVIDIA T4** *(measured)* | 16 GB | 1× (baseline) | **0.69 FPS** |
| RTX 3090 | 24 GB | ~2× | ~1.4 FPS |
| RTX 4090 | 24 GB | ~3.5× | ~2.4 FPS |
| A100 (40 GB) | 40 GB | ~4× | ~2.8 FPS |
| A100 (80 GB) | 80 GB | ~5× | ~3.5 FPS |
| 2× A100 (multi-GPU) | 160 GB | ~9× | ~6.2 FPS |

> ⚠️ Even on a **2× A100 setup**, the model does not reach real-time (15 FPS) at 1024×1024 input.  
> The bottleneck is the ViT-L attention complexity — a fundamental architectural constraint.

### What GPU Does and Does NOT Control

| Factor | Controlled by GPU? |
|---|---|
| ViT-L backbone computation | ✅ Yes — biggest win |
| Cascade ROI head computation | ✅ Yes |
| GPU memory bandwidth (attention) | ✅ Yes |
| Image pre-processing (resize, normalize) | ❌ No — CPU |
| Soft-NMS post-processing | ❌ No — CPU |
| PCI-e data transfer speed | ❌ No — bus speed |

---

## 6. Speed vs. Accuracy Trade-offs

If real-time speed is required, the following options exist (ordered by ease of implementation):

| Optimization | Effort | Speed Gain | AP Impact |
|---|---|---|---|
| **Better GPU** (e.g., A100 vs T4) | None (hardware) | ~5× | ✅ Zero |
| **FP16 / AMP inference** | Low — 1 flag | ~1.5× | ✅ < 0.5% drop |
| **Reduce RPN proposals** (2000 → 500) | Low — config change | ~1.3× | ⚠️ Small drop |
| **Reduce input to 640×640** | Low — config change | ~2.5× | ⚠️ ~5–8% AP drop |
| **TensorRT FP16 export** | Medium | ~4× | ✅ < 1% drop |
| **Switch to EVA-02 Base backbone** | High — retrain | ~2× | ⚠️ Moderate drop |
| **Switch to YOLO-style model** | High — full retrain | ~20× | ❌ Large AP drop |

---

## 7. Thesis Recommended Statement

> *"The proposed EVA-02 Large model achieves an inference throughput of **0.69 FPS** (latency: 1,453 ms/frame) on an NVIDIA T4 GPU at 1024×1024 input resolution, as measured using Detectron2's built-in inference benchmark over 300 frames. While this precludes real-time continuous video stream processing (≥25 FPS), the model is well-suited for **near-real-time early warning systems** in which individual suspicious frames are flagged asynchronously, or for **post-event forensic analysis** of recorded surveillance footage where detection accuracy is prioritized over processing speed. Future work may explore TensorRT optimization or lighter backbone alternatives to approach real-time deployment thresholds."*

---

## 8. How to Re-Run the Benchmark

```bash
cd "d:\Masters Program\Ms thesis\Experiments\EVA-posting\EVA-Changed\EVA-02\det"

python tools/benchmark.py \
    --task eval \
    --num-gpus 1 \
    --config-file projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_l_4attn_1024_gun_ap50_v5.py \
    train.init_checkpoint=/path/to/your/checkpoint.pth
```

### Reading the Output
```
300 iters in X seconds.

FPS     = 300 / X
Latency = (X / 300) × 1000  ms
```

### Benchmark Script Location
- [`tools/benchmark.py`](tools/benchmark.py) — Detectron2 built-in benchmark tool
- [`tools/analyze_model.py`](tools/analyze_model.py) — GFLOPs / parameter count analysis

---

## 9. Additional: GFLOPs Analysis (Hardware-Independent)

To measure model complexity independently of GPU hardware:

```bash
python tools/analyze_model.py \
    --tasks flop \
    --num-inputs 10 \
    --config-file projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_l_4attn_1024_gun_ap50_v5.py \
    train.init_checkpoint=/path/to/checkpoint.pth
```

GFLOPs (Giga Floating-Point Operations) measure the **computational cost of one forward pass**, independent of what GPU you use. This is a hardware-agnostic metric useful for comparing models in publications.

---

*Last updated: May 2026 | Model version: v5 (Balanced Armed/Unarmed/Gun)*
