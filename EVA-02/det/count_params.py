import torch
import sys
import os

# Add the det directory to path so mmdet custom modules can be found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmdet.apis import init_detector

config_file = 'projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_l_4attn_1024_gun_ap50_v3.py'
checkpoint_file = r'/kaggle/input/datasets/ziaawan132/weaponsgunsdataset/model_0005999_new.pth'   # <-- UPDATE THIS PATH

print("Loading model... (this may take a minute)")
model = init_detector(config_file, checkpoint_file, device='cpu')

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
backbone_params  = sum(p.numel() for p in model.backbone.parameters())
head_params      = total_params - backbone_params

print("=" * 45)
print(f"  Total Parameters:     {total_params / 1e6:.2f}M")
print(f"  Trainable Parameters: {trainable_params / 1e6:.2f}M")
print(f"  Backbone (ViT-L):     {backbone_params / 1e6:.2f}M")
print(f"  Detection Head:       {head_params / 1e6:.2f}M")
print("=" * 45)
