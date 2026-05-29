import torch
from collections import defaultdict

# ── Configuration ──────────────────────────────────────────────────────────────
checkpoint_file = '/kaggle/input/datasets/ziaawan132/weaponsgunsdataset/model_0005999_new.pth'

# ── Load checkpoint (no mmdet needed) ─────────────────────────────────────────
print("Loading checkpoint...")
ckpt = torch.load(checkpoint_file, map_location='cpu')

# Handle different checkpoint formats
if 'model' in ckpt:
    state_dict = ckpt['model']
elif 'state_dict' in ckpt:
    state_dict = ckpt['state_dict']
else:
    state_dict = ckpt

print(f"Checkpoint keys format: '{list(state_dict.keys())[0].split('.')[0]}...'")

# ── Count parameters per component ────────────────────────────────────────────
component_params = defaultdict(int)
total_params = 0

for key, tensor in state_dict.items():
    n = tensor.numel()
    total_params += n
    top_key = key.split('.')[0]   # e.g. 'backbone', 'neck', 'roi_head', etc.
    component_params[top_key] += n

# ── Print results ──────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print(f"  {'Component':<25} {'Parameters':>10}")
print("-" * 50)
for comp, count in sorted(component_params.items(), key=lambda x: -x[1]):
    print(f"  {comp:<25} {count/1e6:>8.2f}M")
print("-" * 50)
print(f"  {'TOTAL':<25} {total_params/1e6:>8.2f}M")
print("=" * 50)
