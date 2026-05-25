"""EVA-02 Large + ARMED: Gun AP50 v4 — Frozen Backbone + Head-Only Training.

This config addresses the core problem identified in v3: the 304M-param
ViT-L backbone is OVERFITTING on the tiny 1088-image dataset, causing
noisy Gun AP and a ceiling at ~25%.

KEY CHANGES from v3:
 1. NEAR-FROZEN BACKBONE via lr_decay_rate=0.5 (was 0.8).
    - Early backbone layers get LR × 0.5^25 ≈ 0 (effectively frozen)
    - Middle layers get LR × 0.5^13 ≈ LR × 0.0001 (nearly frozen)
    - Last backbone layer gets LR × 0.5 (minimal updates)
    - Detection heads get FULL LR (all learning happens here)
 2. HIGHER HEAD LR: 5e-5 (was 1e-5). Since only heads are learning,
    they need a stronger signal to compensate.
 3. LOWER test_score_thresh: 0.01 (was 0.05) to recover more Gun
    detections during evaluation.
 4. SOFTER NMS: sigma=0.7 (was 0.5), iou_threshold=0.15 (was 0.2)
    to reduce suppression of valid Gun detections.
 5. Updated data paths to match Kaggle input dataset location.

All other v3 settings are PRESERVED:
    - Cosine LR schedule, copy-paste augmentation, focal loss,
      repeat-factor oversampling, low-quality matching, etc.

Launch command:
    python tools/lazyconfig_train_net.py \\
        --config-file projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_l_4attn_1024_gun_ap50_v4.py \\
        --num-gpus 2 \\
        train.init_checkpoint=output/model_0013999.pth \\
        train.output_dir=output_v4 \\
        train.eval_period=2000 \\
        train.checkpointer.period=2000

    WARNING: Do NOT use --resume. Fresh optimizer is essential so the
    new lr_decay_rate=0.5 takes effect from scratch.
"""

from functools import partial

from detectron2.config import LazyCall as L
import detectron2.data.transforms as T
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import CosineParamScheduler

from .eva2_coco_cascade_mask_rcnn_vitdet_l_4attn_1024_lrd0p8_bs2_lr1e6 import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
)

# ═══════════════════════════════════════════════════════════════════════════
# ██  BACKBONE FREEZE — THE CORE v4 CHANGE
# ═══════════════════════════════════════════════════════════════════════════
#
# With lr_decay_rate=0.5 and 24 layers, the effective LR per layer is:
#
#   Layer  0 (patch_embed):  5e-5 × 0.5^25 = 1.5e-12  (frozen)
#   Layer  6:                5e-5 × 0.5^19 = 9.5e-11   (frozen)
#   Layer 12:                5e-5 × 0.5^13 = 6.1e-09   (near-frozen)
#   Layer 18:                5e-5 × 0.5^7  = 3.9e-07   (minimal updates)
#   Layer 23 (last block):   5e-5 × 0.5^2  = 1.25e-05  (light fine-tuning)
#   Detection heads (FPN):   5e-5 × 1.0    = 5.0e-05   (FULL learning)
#
# This preserves the powerful pretrained EVA-02 features while focusing
# ALL learning capacity on the detection heads → less overfitting,
# more stable Gun AP.
# ═══════════════════════════════════════════════════════════════════════════

optimizer.lr = 5e-5
optimizer.params.lr_factor_func = partial(
    get_vit_lr_decay_rate, lr_decay_rate=0.5, num_layers=24
)
optimizer.params.overrides = {}
optimizer.params.weight_decay_norm = None

# ---------------------------------------------------------------------------
# 1) Aggressive small-object augmentation — same as v3.
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
# 2) Repeat-factor oversampling — same as v3 (thresh=0.10).
# ---------------------------------------------------------------------------
dataloader.train.sampler = L(RepeatFactorTrainingSampler)(
    repeat_factors=L(RepeatFactorTrainingSampler.repeat_factors_from_category_frequency)(
        dataset_dicts="${dataloader.train.dataset}", repeat_thresh=0.10
    )
)

# ---------------------------------------------------------------------------
# 3) Anchor sizes — same as v1/v2/v3 (checkpoint-compatible).
# ---------------------------------------------------------------------------
model.proposal_generator.anchor_generator.sizes = [
    [16],
    [32],
    [64],
    [128],
    [256],
]

# ---------------------------------------------------------------------------
# 4) Cascade matchers: allow_low_quality_matches on ALL 3 stages — same as v3.
# ---------------------------------------------------------------------------
model.roi_heads.proposal_matchers = [
    L(Matcher)(thresholds=[0.4], labels=[0, 1], allow_low_quality_matches=True),
    L(Matcher)(thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=True),
    L(Matcher)(thresholds=[0.6], labels=[0, 1], allow_low_quality_matches=True),
]

# ---------------------------------------------------------------------------
# 5) Soft-NMS — MORE LENIENT than v3 to reduce Gun suppression.
#    Lower iou_threshold (0.15 vs 0.2) = less aggressive suppression.
#    Higher sigma (0.7 vs 0.5) = softer score decay for overlapping boxes.
# ---------------------------------------------------------------------------
model.roi_heads.use_soft_nms = True
model.roi_heads.method = "linear"
model.roi_heads.iou_threshold = 0.15
model.roi_heads.sigma = 0.7
model.roi_heads.class_wise = True

# ---------------------------------------------------------------------------
# 6) RPN proposal capacity — same as v3.
# ---------------------------------------------------------------------------
model.proposal_generator.pre_nms_topk = (4000, 2000)
model.proposal_generator.post_nms_topk = (2000, 2000)

# ---------------------------------------------------------------------------
# 7) ROI sampling — same as v3 (positive_fraction=0.75).
# ---------------------------------------------------------------------------
model.roi_heads.batch_size_per_image = 512
model.roi_heads.positive_fraction = 0.75

# ---------------------------------------------------------------------------
# 8) Focal loss + GIoU — same as v3 BUT lower test_score_thresh.
#
#    test_score_thresh: 0.01 (was 0.05) — recover more low-confidence
#    Gun detections. AP rewards recall, and many Gun predictions are
#    borderline. This alone can give +0.5–1.5 AP.
# ---------------------------------------------------------------------------
for _bp in model.roi_heads.box_predictors:
    _bp.use_focal_loss = True
    _bp.focal_loss_alpha = [0.15, 0.15, 0.85, 0.05]
    _bp.focal_loss_gamma = 3.0
    _bp.box_reg_loss_type = "giou"
    _bp.test_score_thresh = 0.01  # was 0.05 — recover more Gun detections

# ---------------------------------------------------------------------------
# 9) Cosine LR schedule — same structure as v3 but at higher base LR.
#    Base LR = 5e-5 (5× v3), but backbone layers see almost zero
#    thanks to lr_decay_rate=0.5. Only detection heads get the full 5e-5.
#
#    Training for 30k iters (shorter than v3's 50k because heads converge
#    faster when backbone is frozen).
# ---------------------------------------------------------------------------
train.max_iter = 30000
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(CosineParamScheduler)(
        start_value=1.0,
        end_value=0.01,
    ),
    warmup_length=500 / train.max_iter,  # 500 iters warmup (faster since heads only)
    warmup_factor=0.001,
)

# ---------------------------------------------------------------------------
# 10) Copy-paste augmentation — same as v3 (reduced to 25% prob, max 2).
#     Updated paths to match Kaggle input dataset location.
# ---------------------------------------------------------------------------
from detectron2.data.copy_paste_mapper import CopyPasteDatasetMapper

_augmentations = dataloader.train.mapper.augmentations

dataloader.train.mapper = L(CopyPasteDatasetMapper)(
    is_train=True,
    augmentations=_augmentations,
    image_format="BGR",
    use_instance_mask=False,
    recompute_boxes=False,
    # Copy-paste specific settings
    copy_paste_prob=0.25,
    max_paste_instances=2,
    gun_category_id=2,
    annotations_json="/kaggle/input/datasets/alirehman6666/weaponsdatasetoriginal/OriginalDataset/annotations/armed_train.json",
    image_root="/kaggle/input/datasets/alirehman6666/weaponsdatasetoriginal/OriginalDataset",
)
