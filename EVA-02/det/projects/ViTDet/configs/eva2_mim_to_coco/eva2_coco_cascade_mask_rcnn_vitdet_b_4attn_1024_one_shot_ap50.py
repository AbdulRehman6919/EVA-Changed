from detectron2.config import LazyCall as L
import detectron2.data.transforms as T
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.modeling.matcher import Matcher

from .eva2_coco_cascade_mask_rcnn_vitdet_b_4attn_1024_lrd0p7_10k_bs2_lr5e7 import (
    dataloader,
    lr_multiplier,
    model,
    train,
)

# One-shot AP50 experiment: prioritize small-object visibility and gun recall.

# 1) Small-object focused augmentation policy.
dataloader.train.mapper.augmentations = [
    L(T.RandomFlip)(horizontal=True),
    L(T.RandomContrast)(intensity_min=0.6, intensity_max=1.4),
    L(T.RandomBrightness)(intensity_min=0.6, intensity_max=1.4),
    L(T.RandomSaturation)(intensity_min=0.5, intensity_max=1.5),
    L(T.ResizeScale)(
        min_scale=0.8,
        max_scale=2.0,
        target_height=1024,
        target_width=1024,
    ),
    L(T.FixedSizeCrop)(crop_size=(1024, 1024), pad=False),
]

# 2) Class-imbalance mitigation for rare gun instances.
dataloader.train.sampler = L(RepeatFactorTrainingSampler)(
    repeat_factors=L(RepeatFactorTrainingSampler.repeat_factors_from_category_frequency)(
        dataset_dicts="${dataloader.train.dataset}", repeat_thresh=0.001
    )
)

# 3) Recover stricter localization quality in cascade matching.
model.roi_heads.proposal_matchers = [
    L(Matcher)(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False)
    for th in [0.5, 0.6, 0.7]
]

# 4) Moderate schedule extension for convergence.
train.max_iter = 80000
lr_multiplier.scheduler.milestones = [
    train.max_iter * 8 // 10,
    train.max_iter * 9 // 10,
]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 1000 / train.max_iter
