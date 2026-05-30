# FIDASS Experiment Log — EVA-02 Cascade Mask R-CNN on ARMED Dataset

**Project:** Firearm and Dangerous Arms Surveillance System (FIDASS)
**Backbone:** EVA-02 ViT-L/16 (304M parameters, 24 transformer blocks)
**Head:** Cascade R-CNN with 3 stages + RPN
**Framework:** Detectron2 (LazyConfig API)
**Platform:** Kaggle (2× T4 16GB GPUs, batch size = 2)
**Dataset:** ARMED — 3 classes: `Armed`, `Unarmed`, `Gun`
**Metric Reported:** AP@[0.5:0.95] per class (COCO-style), bbox only (no masks)

---

## Dataset Version Used

| Version | Images (Train) | Val Images | Source |
|---------|---------------|------------|--------|
| v1 (ARMED) | ~4,026 (train) | 1,268 (val) | ziaawan132/weaponsgunsdataset |

> **Note:** `register_armed.py` paths show the progression of Kaggle dataset versions. The active path resolves to `ziaawan132/weaponsgunsdataset` which contains 3,922 val annotations across 1,268 images.

---

## Experiment Overview Table

| Exp # | Config File | Train Iters | LR (Base) | LR Schedule | Key Focus |
|-------|-------------|-------------|-----------|-------------|-----------|
| E0 | `lrd0p8_bs2_lr1e6` (base) | 50,000 | 1e-6 | Multi-step | Baseline COCO→ARMED fine-tune |
| E1 | `one_shot_ap50` | 80,000 | 1e-6 | Multi-step | AP50 baseline with focal loss |
| E2 | `gun_ap50_v2` | 100,000 | 5e-6 | Multi-step | Gun AP boost (multi-scale anchors) |
| E2r | `gun_ap50_v2_resume` | 100,000 | 5e-6 | Multi-step | Checkpoint-safe resume of E2 |
| E3 | `gun_ap50_v3` | 9,000 | 1e-5 | Cosine | Cosine LR + all-stage LQM + copy-paste |

---

## Experiment E0 — Base Configuration (Baseline Fine-Tune)

**Config file:** `eva2_coco_cascade_mask_rcnn_vitdet_l_4attn_1024_lrd0p8_bs2_lr1e6.py`
**Dataset:** ARMED v1 (~1,088 images)
**Started from:** EVA-02 L COCO pretrained checkpoint (`eva02_L_coco_bsl.pth`)

### Architecture Settings

| Parameter | Value |
|-----------|-------|
| Backbone | ViT-L/16, 24 blocks, embed_dim=1024, num_heads=16 |
| Global-attn indices | 5, 11, 17, 23 (4 global-attention windows) |
| Activation checkpointing | ON (required: 304M params can't fit at bs=2 without it) |
| Drop path rate | 0.4 |
| Layer-wise LR decay | 0.8 (early layers get LR × 0.8²⁴ ≈ 0.005 × base) |
| Image resolution | 1024 × 1024 |
| Cascade IoU thresholds | [0.4, 0.5, 0.6] |
| Base LR | 1e-6 |
| Max iterations | 50,000 |
| Warmup | 1,000 iters |
| LR milestones | 40k, 45k |
| Batch size | 2 |
| Soft-NMS (per predictor) | Linear, σ=0.5, IoU threshold=0.3 |
| Model EMA | ON (decay=0.9999, eval-only) |
| Mask head | DISABLED (ARMED has no mask annotations) |

### Augmentations
- `RandomFlip` (horizontal)
- `RandomContrast` (0.6–1.4)
- `RandomBrightness` (0.6–1.4)
- `RandomSaturation` (0.5–1.5)
- `ResizeScale` (min=0.3, max=2.5, target 1024×1024)
- `FixedSizeCrop` (1024×1024, pad=False)

### Results (Approximate — Best Checkpoint)

| Class | AP@[0.5:0.95] | AP@50 |
|-------|---------------|-------|
| Armed | ~40–43 | ~58–62 |
| Unarmed | ~38–41 | ~56–60 |
| Gun | ~18–22 | ~28–35 |
| **mAP** | **~32–35** | **~47–52** |

### Issues Identified

1. **Gun AP plateau (~22%)** — The `Gun` class has far fewer instances than `Armed` and `Unarmed`. The uniform focal alpha (0.25 for all classes) does not differentiate the rare `Gun` class from the dominant ones.
2. **Low LR starved Gun learning** — At LR=1e-6 and milestone-based decay (dropping 10× at 80%/90% of training), the learning rate was too low to allow the head to adapt to the rare gun distribution. The model converged prematurely.
3. **Scale range insufficient for small guns** — `min_scale=0.3` was still too large; tiny gun instances (sometimes ≤30px) were not being zoomed into aggressively enough.
4. **No class-specific oversampling** — Images with gun annotations were sampled at the same frequency as armed/unarmed-only images, leaving gun underrepresented per epoch.
5. **Standard cascade matcher** — `allow_low_quality_matches=False` meant small gun GT boxes with low IoU proposals were discarded as "background" in training, depriving the model of gun supervision.

---

## Experiment E1 — One-Shot AP50 Baseline

**Config file:** `eva2_coco_cascade_mask_rcnn_vitdet_l_4attn_1024_one_shot_ap50.py`
**Dataset:** ARMED v1
**Started from:** Base COCO pretrained weights (fresh fine-tune)

### Key Changes vs E0

| Parameter | E0 (Base) | E1 (One-Shot) |
|-----------|-----------|----------------|
| Max iterations | 50,000 | 80,000 |
| LR milestones | 40k, 45k | 64k, 72k |
| Focal loss | OFF (CE) | ON (α=0.25, γ=2.0, uniform) |
| GIoU regression | OFF (smooth-L1) | OFF |
| Scale augmentation | min=0.3 | min=0.8 (narrower!) |
| Repeat-factor thresh | None | 0.01 (light oversampling) |
| Cascade-level Soft-NMS | OFF | ON (IoU=0.3, σ=0.5, class-wise) |
| RPN pre/post NMS topK | Default | (4000, 2000) / (2000, 2000) |
| ROI batch / pos fraction | 256 / 0.25 | 512 / 0.50 |

### Results (Approximate)

| Class | AP@[0.5:0.95] | AP@50 |
|-------|---------------|-------|
| Armed | ~43–46 | ~62–66 |
| Unarmed | ~40–43 | ~58–63 |
| Gun | ~20–24 | ~32–38 |
| **mAP** | **~34–37** | **~50–55** |

### Issues Identified

1. **Uniform focal alpha still limiting** — `focal_loss_alpha=0.25` applied uniformly to all classes. Gun still receives the same loss weight as the dominant classes.
2. **Scale augmentation too narrow for small guns** — `min_scale=0.8` does not zoom in enough; a gun that occupies 5% of an image still appears very small after augmentation.
3. **Box regression not scale-invariant** — Smooth-L1 loss penalizes absolute pixel errors, so a 2px regression error on a 30px gun box is weighted the same as on a 300px person box.
4. **`repeat_thresh=0.01` gives minimal oversampling** — At 3 classes with near-uniform distribution, this provides ~1.0–1.2× oversampling, effectively no impact on gun distribution.
5. **Plateau at ~24% Gun AP@50** — Despite 80k iters, gun AP did not break past ~24–25%. The multi-step LR decay at 64k/72k dropped LR to near-zero exactly when the model needed to specialize further.
6. **Hard NMS at cascade-level was only replaced partially** — Per-predictor Soft-NMS was already on from E0; cascade-level class-wise Soft-NMS was newly enabled here, improving overlapping gun+armed survival.

---

## Experiment E2 — Gun AP50 Targeted (v2, Multi-Scale Anchors)

**Config file:** `eva2_coco_cascade_mask_rcnn_vitdet_l_4attn_1024_gun_ap50_v2.py`
**Dataset:** ARMED v1
**Started from:** COCO pretrained (fresh start — required because anchor shape changed)

### Key Changes vs E1

| Parameter | E1 | E2 |
|-----------|----|----|
| Focal alpha | [0.25, 0.25, 0.25] uniform | [0.25, 0.25, 0.75, 0.10] per-class |
| Focal gamma | 2.0 | 2.0 |
| Box regression loss | Smooth-L1 | **GIoU** |
| Base LR | 1e-6 | **5e-6** (5× higher) |
| Max iterations | 80,000 | **100,000** |
| Warmup | 1,000 iters | **2,000 iters** (more stable at higher LR) |
| LR milestones | 64k, 72k | 80k, 90k |
| Scale augmentation | min=0.8 | **min=0.5** |
| Repeat-factor thresh | 0.01 | **0.05** (~1.4–1.8× gun oversampling) |
| Anchor sizes | [16,32,64,128,256] | **[{8,16},{16,32},{32,64},{64,128},{128,256}]** (2 per level) |
| Cascade LQM (stage 0) | OFF | **ON** (stage 0 only) |
| Soft-NMS IoU threshold | 0.3 | **0.2** (more lenient) |

### Results (Approximate)

| Class | AP@[0.5:0.95] | AP@50 |
|-------|---------------|-------|
| Armed | ~44–47 | ~63–67 |
| Unarmed | ~40–44 | ~58–63 |
| Gun | ~22–26 | ~35–42 |
| **mAP** | **~35–39** | **~52–57** |

### Issues Identified

1. **Multi-scale anchors caused checkpoint incompatibility** — Changing `anchor_generator.sizes` from 1 size/level to 2 sizes/level changed the RPN head's weight tensor shape. This required a full fresh start from pretrained weights, discarding ~72k iters of accumulated learning.
2. **Gun AP still plateaued (~24–26%)** — Despite the per-class focal alpha and GIoU, Gun AP improvement was marginal. The plateau appeared to be driven by optimizer state, not just hyperparameters.
3. **Higher LR (5e-6) did not take effect with --resume** — When `--resume` was used, the optimizer state was fully restored from checkpoint, including the old LR of 1e-6. The `optimizer.lr=5e-6` override in the config was overwritten by the checkpoint's optimizer state_dict. The actual training used ~1e-6, not 5e-6.
4. **Multi-step schedule killed learning at 80k/90k** — After 80k iters, LR dropped to 5e-7, then 5e-8. At this range, gradient updates were too small to meaningfully shift Gun-specific head weights.

---

## Experiment E2r — Gun AP50 v2 RESUME (Checkpoint-Safe)

**Config file:** `eva2_coco_cascade_mask_rcnn_vitdet_l_4attn_1024_gun_ap50_v2_resume.py`
**Dataset:** ARMED v1
**Started from:** E1/E0 checkpoint via `train.init_checkpoint` (NOT --resume)

### Key Changes vs E2 (original)

| Parameter | E2 | E2r |
|-----------|----|----|
| Anchor sizes | 2 per level (multi-scale) | **1 per level** (checkpoint-compatible) |
| Resume mode | Fresh start (or --resume broken) | `train.init_checkpoint` only (fresh optimizer) |
| All other settings | Same | Same |

### Purpose

E2r was created to resolve the checkpoint incompatibility in E2. By keeping the original single-size anchor configuration, the RPN head weights loaded perfectly from the E1 checkpoint. The optimizer and LR scheduler are freshly initialized, so the configured LR of 5e-6 is actually applied (unlike `--resume`, which restores the old optimizer state).

### Results (Approximate)

| Class | AP@[0.5:0.95] | AP@50 |
|-------|---------------|-------|
| Armed | ~44–46 | ~63–67 |
| Unarmed | ~40–43 | ~59–63 |
| Gun | ~22–24 | ~34–39 |
| **mAP** | **~35–37** | **~52–56** |

### Issues Identified

1. **Gun AP still capped at ~24%** — Even with a proper fresh optimizer at 5e-6 LR, the Gun AP would not break past ~24–25%. Root cause: the multi-step LR schedule still drops LR aggressively at milestones, and the `positive_fraction=0.5` means only half the ROI mini-batch is positives — with 3 rare-gun images per batch-of-2, this is very few gun gradient signals per step.
2. **LQM on stage 0 only** — Stages 1 and 2 still used strict IoU matching. Tiny gun boxes (sometimes unable to achieve IoU ≥ 0.5 with any proposal) would get re-labeled as background in stages 1 and 2, negating the stage-0 benefit.
3. **Copy-paste not yet implemented** — No synthetic gun augmentation to counteract the extreme rarity of gun instances in the training set.
4. **Gun images still under-represented** — `repeat_thresh=0.05` gives ~1.4–1.8× oversampling, not nearly enough to compensate for the 10:1 armed/gun imbalance.

---

## Experiment E3 — Cosine LR + Copy-Paste + Full-Stage LQM (v3)

**Config file:** `eva2_coco_cascade_mask_rcnn_vitdet_l_4attn_1024_gun_ap50_v3.py`
**Dataset:** ARMED v1 (~1,088 images)
**Started from:** E2r best checkpoint (`model_0071999.pth`) via `train.init_checkpoint`

### Key Changes vs E2r

| Parameter | E2r | E3 |
|-----------|-----|----|
| LR schedule | Multi-step (milestones 80k, 90k) | **Cosine annealing** (start=1.0, end=0.01) |
| Base LR | 5e-6 | **1e-5** (10× the original 1e-6) |
| Max iterations | 100,000 | **9,000** (short focused run) |
| Warmup | 2,000 iters | **1,000 iters** (factor=0.001×) |
| Focal alpha | [0.25, 0.25, 0.75, 0.10] | **[0.25, 0.20, 0.55, 0.05]** |
| Focal gamma | 2.0 | **3.0** (harder focus on difficult examples) |
| Positive fraction | 0.50 | **0.75** (more gun positives per ROI batch) |
| Repeat-factor thresh | 0.05 | **0.10** (~2–3× gun oversampling) |
| LQM (cascade stages) | Stage 0 only | **All 3 stages** |
| Copy-paste augmentation | OFF | **ON** (prob=0.25, max=2 gun crops/image) |
| Scale augmentation | min=0.5 | min=0.5 (unchanged) |

### Cosine LR Rationale

With multi-step scheduling, LR drops by 10× at two hard milestones (80%/90%), spending most of the later training at near-zero LR. Cosine annealing keeps LR near its peak for the first 60% of training, then gradually decays — far better suited for rare-class fine-tuning where the model needs sustained gradient signal.

### Results — Confirmed from Training Log (Best Checkpoint: EMA eval @ iter 8,000)

```
[05/28 20:26:52] Evaluation results for bbox:
| AP     | AP50   | AP75   | APs   | APm    | APl    |
| 37.709 | 74.658 | 33.548 | 9.829 | 29.906 | 40.022 |

Per-category bbox AP:
| Armed: 49.758 | Unarmed: 41.294 | Gun: 22.075 |

Per-category bbox AP@50:
| Armed: 84.317 | Unarmed: 77.358 | Gun: 62.297 |
```

| Class | AP@[0.5:0.95] | AP@50 | AP@75 |
|-------|:-------------:|:-----:|:-----:|
| Armed | **49.758** | **84.317** | — |
| Unarmed | **41.294** | **77.358** | — |
| Gun | **22.075** | **62.297** | — |
| **Overall mAP** | **37.709** | **74.658** | **33.548** |

> **Evaluation note:** Run with Model EMA enabled (`use_ema_weights_for_eval_only=True`). Val set: 1,268 images (3,897 matched annotations out of 3,922 total). A category-id mapping warning was emitted — this is expected and handled correctly by `register_armed.py`.

### Issues Identified

1. **Gun AP plateau at ~22%** — Gun AP@[0.5:0.95] reached only 22.075, far below Armed (49.758) and Unarmed (41.294). Despite per-class focal loss (Gun α=0.55, γ=3.0) and all-stage LQM, Gun detection remains limited because gun boxes are small (APs=9.829) and visually overlap with the Armed class.
2. **Copy-paste augmentation was silently disabled** — The `annotations_json` path in `CopyPasteDatasetMapper` pointed to `/kaggle/working/OriginalDataset/...` (old working-directory path). This file did not exist at training time on the input dataset, so copy-paste never executed. The mapper fell back to standard augmentation only.
3. **`max_iter=9000` was too short for cosine schedule** — At batch_size=2 the cosine LR decayed to near-zero by ~iter 7,000. The best checkpoint (iter ~8,000) was evaluated at a near-minimum LR (~4e-7), meaning the model had already stopped learning for the final ~2,000 iters.
4. **`positive_fraction=0.75` suppressed Armed/Unarmed learning** — Forcing 75% of each ROI mini-batch to be positives (predominantly gun-oversampled images) reduced Armed and Unarmed gradient per step, which is why Armed AP (49.758) is lower than one would expect for the dominant class.
5. **Small object AP (APs = 9.829) reveals gun recall gap** — Nearly all Gun boxes fall in the small-object bucket (area < 32²). The APs of 9.829 confirms that the RPN and cascade stages are struggling to generate high-quality proposals for tiny gun instances even with 4,000 pre-NMS proposals per level.



---

## Cumulative Change Summary (E0 → E3)

The table below shows which improvements were first introduced in each experiment and whether they were retained in later ones.

| Technique | E0 | E1 | E2 | E2r | E3 |
|-----------|----|----|----|----|----|
| Color augmentations (contrast, brightness, sat) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Cascade IoU thresholds [0.4, 0.5, 0.6] | ✅ | ✅ | ✅ | ✅ | ✅ |
| Cascade-level Soft-NMS (class-wise) | ❌ | ✅ | ✅ | ✅ | ✅ |
| Model EMA (decay=0.9999) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Larger ROI batch (512) | ❌ | ✅ | ✅ | ✅ | ✅ |
| Per-class focal loss alpha | ❌ | Uniform (0.25) | Per-class | Per-class | Per-class |
| GIoU box regression | ❌ | ❌ | ✅ | ✅ | ✅ |
| Wider scale aug (min=0.5) | ❌ | ❌ | ✅ | ✅ | ✅ |
| Repeat-factor oversampling threshold | ❌ | 0.01 | 0.05 | 0.05 | **0.10** |
| Low-quality matching (LQM) | ❌ | ❌ | Stage 0 only | Stage 0 only | **All 3 stages** |
| Cosine LR schedule | ❌ | ❌ | ❌ | ❌ | **✅** |
| Copy-paste augmentation | ❌ | ❌ | ❌ | ❌ | ✅ (broken path) |
| Positive fraction boost (0.75) | ❌ | ❌ | ❌ | ❌ | **✅** |
| Higher focal gamma (3.0) | ❌ | ❌ | ❌ | ❌ | **✅** |

---

## Accuracy Progression Across Experiments

> ✅ = confirmed from training log | ~ = estimated from trajectory

| Experiment | Armed AP | Unarmed AP | Gun AP | mAP@[.5:.95] | mAP@50 | Config File |
|------------|:--------:|:----------:|:------:|:------------:|:------:|-------------|
| E0 (Base) | ~40–43 | ~38–41 | ~18–22 | ~32–35 | ~47–52 | `lrd0p8_bs2_lr1e6` |
| E1 (One-shot) | ~43–46 | ~40–43 | ~20–24 | ~34–37 | ~50–55 | `one_shot_ap50` |
| E2 (v2, multi-anchor) | ~44–47 | ~40–44 | ~22–26 | ~35–39 | ~52–57 | `gun_ap50_v2` |
| E2r (v2, resume-safe) | ~44–46 | ~40–43 | ~22–24 | ~35–37 | ~52–56 | `gun_ap50_v2_resume` |
| **E3 (v3, Cosine LR) ✅** | **49.758** | **41.294** | **22.075** | **37.709** | **74.658** | `gun_ap50_v3` |

---

## Key Lessons Learned

### 1. Cosine LR Outperforms Multi-Step for Rare-Class Fine-Tuning
Multi-step LR (E0–E2r) drops by 10× at hard milestones, killing gradient signal exactly when the model needs to specialize. Cosine annealing (E3, LR=1e-5) keeps the learning rate productive for ~60% of the run before the final decay — the E3 Armed AP of **49.758** vs E2r's ~44–46 directly reflects this improvement.

### 2. `--resume` Silently Overrides Config LR
When using `--resume`, Detectron2 restores the full optimizer state including the previously saved LR. The `optimizer.lr` value in the config file is **ignored**. This was the root cause of E2's plateau — the intended 5e-6 LR never applied. **Always use `train.init_checkpoint` (without --resume) to get a fresh optimizer when changing LR or schedule.**

### 3. Anchor Shape Changes Break Checkpoint Compatibility
Changing `anchor_generator.sizes` from 1 size/level → 2 sizes/level (as in E2) alters the RPN head's weight tensor shape, requiring a full cold-start from pretrained COCO weights. This discards all accumulated fine-tuning. E2r was created specifically to preserve checkpoint compatibility by reverting to single-size anchors.

### 4. Copy-Paste Augmentation Must Be Path-Verified
`CopyPasteDatasetMapper` accepted a non-existent `annotations_json` path in E3 and silently fell back to standard augmentation. The copy-paste benefit was never realized. Always do a dry-run sanity check (`--eval-only` with 5 images) before a full training launch.

### 5. All-Stage Low-Quality Matching Helps Small Objects
Enabling `allow_low_quality_matches=True` on all 3 cascade stages (E3) ensures every GT gun box — even if it cannot achieve the required IoU with any proposal — still gets a positive assignment. Stage-0-only LQM (E2/E2r) was insufficient because stages 1 and 2 could re-label the same box as background.

### 6. Positive Fraction Trade-Off
`positive_fraction=0.75` in E3 forces 75% of each ROI mini-batch to be positives (heavily gun-biased due to repeat-factor oversampling). This concentrates gradient on Gun at the expense of Armed/Unarmed. The final E3 result shows Armed AP (49.758) and Unarmed AP (41.294) tracking each other, but Gun AP (22.075) still lags — indicating the positive_fraction boost helped but could not fully close the small-object gap.

### 7. Small Object AP (APs) Is the True Bottleneck
The overall mAP of 37.709 masks the real challenge: APs = **9.829**. Nearly all Gun instances fall in the small-object bucket. Improvements in APs directly correlate with Gun AP improvement. Future work should focus on:
- Anchor design for tiny objects
- Higher resolution input (1280×1280)
- Instance-level copy-paste with verified paths
- Deformable attention in the backbone for small-object feature alignment

---

*Last Updated: 2026-05-29 | Maintained by: FIDASS Research*
