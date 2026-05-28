"""EVA-02 Large + ARMED: v5 — Balanced Training for Armed & Unarmed.

Goal: IMPROVE Armed and Unarmed AP while PRESERVING Gun AP.

The v3 config was Gun-focused (alpha=0.85, positive_fraction=0.75,
repeat_thresh=0.10). That worked for Gun but starved Armed & Unarmed
of learning signal. v5 rebalances the training to treat all 3 classes
more equally, now that we have a larger dataset (5035 images).

KEY CHANGES from v3:
 1. BALANCED focal alpha: [0.35, 0.30, 0.30, 0.05]
    - Armed gets 0.35 (was 0.25) → more loss weight
    - Unarmed gets 0.30 (was 0.20) → more loss weight
    - Gun gets 0.30 (was 0.55) → reduced but still strong
    - BG stays 0.05

 2. BALANCED positive_fraction: 0.50 (was 0.75)
    - 0.75 crammed positives for rare classes (Gun).
    - 0.50 gives Armed/Unarmed fair representation in ROI batches.

 3. LOWER repeat_thresh: 0.05 (was 0.10)
    - With 5035 images, extreme Gun oversampling is no longer needed.
    - Less Gun oversampling = more Armed/Unarmed training per epoch.

 4. PROPER max_iter: 25000 (~10 epochs at batch_size=2 with 5035 images)
    - v3 had max_iter=9000, so LR died after ~3.5 epochs.
    - Cosine schedule now has room to keep LR high for ~6 epochs.

 5. REMOVED copy-paste augmentation.
    - With 5035 images, copy-paste is no longer needed.
    - It was silently disabled anyway (wrong path in v3).

 6. LOWER focal gamma: 2.0 (was 3.0)
    - gamma=3.0 focuses extremely hard on difficult examples (mostly Gun).
    - gamma=2.0 is the standard COCO default, better for balanced training.

PRESERVED from v3 (these help ALL classes):
    - Cosine LR schedule at 1e-5
    - GIoU box regression
    - Soft-NMS (class-wise)
    - Low-quality matching on all cascade stages
    - Small-object augmentations (flip, color, resize, crop)

Launch command:
    python tools/lazyconfig_train_net.py \\
        --config-file projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_l_4attn_1024_gun_ap50_v5.py \\
        --num-gpus 2 \\
        train.init_checkpoint=<path_to_best_checkpoint>.pth \\
        train.output_dir=output_v5 \\
        train.eval_period=2000 \\
        train.checkpointer.period=2000

    WARNING: Do NOT use --resume. Fresh optimizer is essential so the
    new cosine LR schedule starts from the beginning.
"""

from detectron2.config import LazyCall as L
import detectron2.data.transforms as T
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.modeling.matcher import Matcher
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import CosineParamScheduler

from .eva2_coco_cascade_mask_rcnn_vitdet_l_4attn_1024_lrd0p8_bs2_lr1e6 import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
)

# ---------------------------------------------------------------------------
# 1) Small-object augmentation — same as v3.
# ---------------------------------------------------------------------------
dataloader.train.mapper.augmentations = [
    L(T.RandomFlip)(horizontal=True),
    L(T.RandomContrast)(intensity_min=0.6, intensity_max=1.4),
    L(T.RandomBrightness)(intensity_min=0.6, intensity_max=1.4),
    L(T.RandomSaturation)(intensity_min=0.5, intensity_max=1.5),
    L(T.ResizeScale)(
        min_scale=0.5,
        max_scale=2.5,
        target_height=1024,
        target_width=1024,
    ),
    L(T.FixedSizeCrop)(crop_size=(1024, 1024), pad=False),
]

# ---------------------------------------------------------------------------
# 2) Mild repeat-factor oversampling: thresh=0.05 (was 0.10 in v3).
#    With 5035 images, Gun is less rare → lighter oversampling is enough.
#    This lets Armed & Unarmed get more training time per epoch.
# ---------------------------------------------------------------------------
dataloader.train.sampler = L(RepeatFactorTrainingSampler)(
    repeat_factors=L(RepeatFactorTrainingSampler.repeat_factors_from_category_frequency)(
        dataset_dicts="${dataloader.train.dataset}", repeat_thresh=0.05
    )
)

# ---------------------------------------------------------------------------
# 3) Anchor sizes — checkpoint-compatible, same as v3.
# ---------------------------------------------------------------------------
model.proposal_generator.anchor_generator.sizes = [
    [16],
    [32],
    [64],
    [128],
    [256],
]

# ---------------------------------------------------------------------------
# 4) Cascade matchers: allow_low_quality_matches on ALL 3 stages.
#    Helps all classes, especially small objects. Same as v3.
# ---------------------------------------------------------------------------
model.roi_heads.proposal_matchers = [
    L(Matcher)(thresholds=[0.4], labels=[0, 1], allow_low_quality_matches=True),
    L(Matcher)(thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=True),
    L(Matcher)(thresholds=[0.6], labels=[0, 1], allow_low_quality_matches=True),
]

# ---------------------------------------------------------------------------
# 5) Soft-NMS — same as v3.
# ---------------------------------------------------------------------------
model.roi_heads.use_soft_nms = True
model.roi_heads.method = "linear"
model.roi_heads.iou_threshold = 0.2
model.roi_heads.sigma = 0.5
model.roi_heads.class_wise = True

# ---------------------------------------------------------------------------
# 6) RPN proposal capacity — same as v3.
# ---------------------------------------------------------------------------
model.proposal_generator.pre_nms_topk = (4000, 2000)
model.proposal_generator.post_nms_topk = (2000, 2000)

# ---------------------------------------------------------------------------
# 7) ROI sampling: BALANCED positive_fraction (0.50 vs 0.75 in v3).
#    0.75 was designed to force more Gun positives into each batch.
#    With a larger dataset and balanced focus, 0.50 gives Armed & Unarmed
#    fair representation in the ROI mini-batch.
# ---------------------------------------------------------------------------
model.roi_heads.batch_size_per_image = 512
model.roi_heads.positive_fraction = 0.50

# ---------------------------------------------------------------------------
# 8) BALANCED per-class focal loss.
#
#    Alpha weights (indexed by class):
#      class 0 = Armed   → 0.35 (was 0.25 — boosted to improve Armed AP)
#      class 1 = Unarmed → 0.30 (was 0.20 — boosted to improve Unarmed AP)
#      class 2 = Gun     → 0.30 (was 0.55 — still strong, but not dominant)
#      class 3 = BG      → 0.05 (unchanged — de-weight easy background)
#
#    Gamma 2.0 (was 3.0) — standard COCO default. gamma=3.0 was too
#    aggressive and over-focused on hard examples (mostly Gun).
#    gamma=2.0 gives a more balanced gradient across all classes.
#
#    GIoU box regression — same as v3 (scale-invariant, helps all classes).
# ---------------------------------------------------------------------------
for _bp in model.roi_heads.box_predictors:
    _bp.use_focal_loss = True
    _bp.focal_loss_alpha = [0.35, 0.30, 0.30, 0.05]
    _bp.focal_loss_gamma = 2.0
    _bp.box_reg_loss_type = "giou"

# ---------------------------------------------------------------------------
# 9) COSINE LR schedule — HIGHER LR + LONGER training.
#
#    Base LR = 2e-5 (2× v3's 1e-5). With 5035 images (5× more data),
#    a higher LR is safe and drives stronger gradients for Armed/Unarmed.
#
#    max_iter = 25000 (~10 epochs at batch_size=2 with 5035 images).
#
#    v3 had max_iter=9000, so the cosine schedule decayed the LR to near
#    zero after just ~3.5 epochs — the model barely trained for most of
#    the run. With 25000 iters, the LR stays productive for ~6-7 epochs
#    before the final decay phase.
#
#    Warmup: 1000 iters from 0.001× base → full LR.
# ---------------------------------------------------------------------------
optimizer.lr = 2e-5

train.max_iter = 6000
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(CosineParamScheduler)(
        start_value=1.0,
        end_value=0.01,
    ),
    warmup_length=1000 / train.max_iter,  # 1000 iters warmup
    warmup_factor=0.001,
)

# ---------------------------------------------------------------------------
# 10) NO copy-paste augmentation.
#
#     With 5035 images (5× the original), copy-paste is no longer needed.
#     In v3, the copy-paste mapper path was wrong and it was silently
#     disabled anyway. Removing it avoids confusion.
#
#     The default DatasetMapper from the base config is used instead.
# ---------------------------------------------------------------------------
# (No copy-paste — using default mapper with augmentations defined above)
