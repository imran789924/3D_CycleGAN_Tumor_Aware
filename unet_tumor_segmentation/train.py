"""
Train 3D UNet for tumor segmentation. Saves best model to checkpoint_dir.
Run from project root or from this directory (dataset imports from parent).
"""
import os
import sys
import argparse

# Prefer local unet_tumor_segmentation modules (options, dataset, model, metrics) over project root
_unet_dir = os.path.dirname(os.path.abspath(__file__))
if _unet_dir not in sys.path:
    sys.path.insert(0, _unet_dir)
_root = os.path.dirname(_unet_dir)
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from unet_options import parse_args
from dataset import TumorSegDataset
from model import build_unet
from metrics import compute_metrics_batch


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    gpu_ids = [int(x) for x in args.gpu_ids.split(',') if x.strip()]
    device = torch.device('cuda:%d' % gpu_ids[0] if gpu_ids and torch.cuda.is_available() else 'cpu')

    train_set = TumorSegDataset(
        args.data_path,
        args.mask_dir,
        patch_size=args.patch_size,
        new_resolution=args.new_resolution,
        resample=args.resample,
        min_pixel=args.min_pixel,
        drop_ratio=args.drop_ratio,
        train=True,
    )
    if len(train_set) == 0:
        print('No training samples. Check data_path and mask_dir.')
        return
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    model = build_unet(patch_size=args.patch_size)
    if len(gpu_ids) > 0 and torch.cuda.is_available():
        model = nn.DataParallel(model, gpu_ids)
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.load:
        ckpt = torch.load(args.load, map_location=device)
        if hasattr(model, 'module'):
            model.module.load_state_dict(ckpt.get('state_dict', ckpt))
        else:
            model.load_state_dict(ckpt.get('state_dict', ckpt))
        start_epoch = ckpt.get('epoch', 0) + 1
        print('Resumed from epoch', start_epoch)

    best_dice = 0.0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        for i, (img, mask) in enumerate(train_loader):
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, mask)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        # Validation on train set (quick metric)
        model.eval()
        with torch.no_grad():
            metrics_sum = {'dice': 0., 'iou': 0., 'hd95': 0.}
            n_b = 0
            for img, mask in train_loader:
                img, mask = img.to(device), mask.to(device)
                pred = model(img)
                m = compute_metrics_batch(pred, mask)
                for k in metrics_sum:
                    v = m[k]
                    metrics_sum[k] += v if not (k == 'hd95' and (v != v)) else 0.0
                n_b += 1
        if n_b:
            for k in metrics_sum:
                metrics_sum[k] /= n_b
        print('Epoch %d  loss: %.4f  dice: %.4f  iou: %.4f  hd95: %.4f' % (
            epoch, train_loss, metrics_sum['dice'], metrics_sum['iou'], metrics_sum['hd95']))
        if metrics_sum['dice'] > best_dice:
            best_dice = metrics_sum['dice']
            save_path = os.path.join(args.checkpoint_dir, 'best.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'patch_size': args.patch_size,
                'dice': best_dice,
            }, save_path)
            print('  -> saved best to', save_path)
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'patch_size': args.patch_size,
            }, os.path.join(args.checkpoint_dir, 'epoch_%d.pth' % (epoch + 1)))
    print('Done. Best Dice:', best_dice)


if __name__ == '__main__':
    main()
