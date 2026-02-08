"""
Segmentation metrics: Dice, IoU, HD95, etc.
"""
import numpy as np

try:
    from medpy.metric.binary import hd, hd95, dc
    _has_medpy = True
except ImportError:
    _has_medpy = False


def dice_coef(y_true, y_pred, smooth=1e-5):
    """Dice (F1) per sample; inputs (B, 1, D, H, W) or (D, H, W)."""
    if y_true.ndim == 5:
        y_true = y_true.view(y_true.size(0), -1)
        y_pred = y_pred.view(y_pred.size(0), -1)
    elif y_true.ndim == 4:
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
    intersection = (y_true * y_pred).sum()
    return (2.0 * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)


def iou_coef(y_true, y_pred, smooth=1e-5):
    """IoU (Jaccard) per sample."""
    if y_true.ndim == 5:
        y_true = y_true.view(y_true.size(0), -1)
        y_pred = y_pred.view(y_pred.size(0), -1)
    elif y_true.ndim == 4:
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def numpy_dice(y_true, y_pred, smooth=1e-5):
    """Dice on numpy arrays; y_true, y_pred binary (0/1)."""
    y_true = y_true.astype(np.float32).ravel()
    y_pred = y_pred.astype(np.float32).ravel()
    intersection = (y_true * y_pred).sum()
    return float((2.0 * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth))


def numpy_iou(y_true, y_pred, smooth=1e-5):
    """IoU on numpy arrays."""
    y_true = y_true.astype(np.float32).ravel()
    y_pred = y_pred.astype(np.float32).ravel()
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return float((intersection + smooth) / (union + smooth))


def numpy_hd95(result, reference, voxelspacing=None):
    """95th percentile Hausdorff distance (mm). result, reference: binary numpy."""
    if not _has_medpy:
        return float('nan')
    try:
        return float(hd95(result, reference, voxelspacing=voxelspacing))
    except Exception:
        return float('nan')


def numpy_hd(result, reference, voxelspacing=None):
    """Hausdorff distance (mm)."""
    if not _has_medpy:
        return float('nan')
    try:
        return float(hd(result, reference, voxelspacing=voxelspacing))
    except Exception:
        return float('nan')


def numpy_dc(result, reference):
    """Dice from medpy (same as numpy_dice)."""
    if not _has_medpy:
        return numpy_dice(result, reference)
    try:
        return float(dc(result, reference))
    except Exception:
        return numpy_dice(result, reference)


def compute_metrics_batch(pred, target, voxelspacing=None):
    """
    pred, target: (B, 1, D, H, W) tensors or numpy, values in [0,1]; threshold at 0.5.
    Returns dict with mean dice, iou, hd95 over batch.
    """
    if hasattr(pred, 'detach'):
        pred = pred.detach().cpu().numpy()
    if hasattr(target, 'detach'):
        target = target.detach().cpu().numpy()
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    dices, ious, hd95s = [], [], []
    for i in range(pred.shape[0]):
        p = pred[i, 0]
        t = target[i, 0]
        if t.sum() == 0 and p.sum() == 0:
            dices.append(1.0)
            ious.append(1.0)
            hd95s.append(0.0)
        elif t.sum() == 0 or p.sum() == 0:
            dices.append(0.0)
            ious.append(0.0)
            hd95s.append(float('nan'))
        else:
            dices.append(numpy_dice(t, p))
            ious.append(numpy_iou(t, p))
            hd95s.append(numpy_hd95(p, t, voxelspacing))
    hd95s_valid = [x for x in hd95s if not np.isnan(x)]
    return {
        'dice': np.mean(dices),
        'iou': np.mean(ious),
        'hd95': np.mean(hd95s_valid) if hd95s_valid else float('nan'),
    }
