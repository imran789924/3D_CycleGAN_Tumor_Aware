"""
Save middle slices of 3D volumes (real_A, fake_B, real_B, fake_A, etc.) as PNG
for training visibility. Tensors are (B, C, D, H, W) with layout (B, C, X, Y, Z)
e.g. (B, 1, 128, 128, 64). We take the middle slice along the Z dimension so
the saved image is the XY plane (128x128). Assumes values in [-1, 1] (no min-max scaling).
"""
import os
import numpy as np
import torch

# After selecting batch index we have 4D (C, D, H, W). Slice along last dim (Z, index 3) -> XY plane 128x128
SLICE_DIM_4D = 3


def _tensor_to_slice_uint8(tensor, sample_idx=0, slice_dim=SLICE_DIM_4D):
    """
    Extract middle slice from a 5D tensor (B, C, D, H, W), denormalize from [-1, 1]
    to [0, 255], and return as uint8 numpy (H, W) for grayscale.
    After taking batch index, x is 4D (C, D, H, W); we slice along dim 3 (Z) for XY plane.
    """
    if tensor.dim() != 5:
        raise ValueError("Expected 5D tensor (B, C, D, H, W), got %dD" % tensor.dim())
    x = tensor.detach().float().cpu().numpy()
    x = x[sample_idx]  # (C, D, H, W) — now 4D, so Z is at index 3
    mid = x.shape[slice_dim] // 2
    if slice_dim == 0:
        sl = x[mid, ...]
    elif slice_dim == 1:
        sl = x[:, mid, ...]
    elif slice_dim == 2:
        sl = x[:, mid, :, :]
    else:
        # slice_dim == 3: slice along Z -> (C, D, H) = (C, 128, 128)
        sl = x[:, :, :, mid]
    if sl.ndim == 3:
        sl = sl[0]  # (H, W) grayscale
    # Denormalize from [-1, 1] to [0, 255]
    sl = (sl + 1.0) * 127.5
    sl = np.clip(sl, 0, 255).astype(np.uint8)
    return sl


def save_epoch_slices(visuals_dict, save_dir, epoch):
    """
    Save middle slices of all visuals (real_A, fake_B, real_B, fake_A, rec_A, rec_B, etc.)
    as PNG in save_dir/epoch_slices/epoch_XXX/. Tensors assumed in [-1, 1].
    Args:
        visuals_dict: dict name -> tensor (B, C, D, H, W)
        save_dir: base checkpoint dir (e.g. checkpoints/exp_name)
        epoch: current epoch number
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required to save PNG slices. Install with: pip install pillow")

    epoch_slices_dir = os.path.join(save_dir, "epoch_slices", "epoch_%d" % epoch)
    os.makedirs(epoch_slices_dir, exist_ok=True)

    for name, tensor in visuals_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        try:
            sl = _tensor_to_slice_uint8(tensor)
            png_path = os.path.join(epoch_slices_dir, "%s.png" % name)
            Image.fromarray(sl).save(png_path)
        except Exception as e:
            # Skip if tensor shape is wrong or missing
            print("Warning: could not save slice for %s: %s" % (name, e))
