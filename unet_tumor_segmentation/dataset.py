"""
Dataset for 3D UNet tumor segmentation: loads image + mask pairs.
Uses same patch_size and compatible transforms as the main project.
Images and masks are paired by index (same count and order after sorting). Keep lists consistent.
"""
import os
import sys
import random
import numpy as np
import torch
import SimpleITK as sitk

# Import from parent project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.NiftiDataset import (
    lstFiles, Normalization, resample_sitk_image, _safe_roi,
)
_interpolator_image = 'linear'


class TumorSegDataset(torch.utils.data.Dataset):
    """
    Loads 3D image + binary tumor mask. Input volumes from data_path/images_subdir,
    masks from mask_dir. Paired by index (same count and order after sorting).
    """
    def __init__(self, data_path, mask_dir, patch_size, new_resolution=(0.45, 0.45, 0.45),
                 resample=False, min_pixel=0.1, drop_ratio=0., train=True, images_subdir='labels'):
        self.data_path = data_path
        self.mask_dir = mask_dir
        self.patch_size = tuple(patch_size) if not isinstance(patch_size, tuple) else patch_size
        self.new_resolution = new_resolution
        self.resample = resample
        self.min_pixel = min_pixel
        self.drop_ratio = drop_ratio
        self.train = train
        self.images_subdir = images_subdir
        self.bit = sitk.sitkFloat32

        images_dir = os.path.join(data_path, images_subdir)
        self.images_list = lstFiles(images_dir)
        if mask_dir and os.path.isdir(mask_dir):
            self.masks_list = lstFiles(mask_dir)
            assert len(self.masks_list) == len(self.images_list), \
                'mask_dir must have same number of files as images (got %d vs %d)' % (
                    len(self.masks_list), len(self.images_list))
        else:
            self.masks_list = []

    def read_image(self, path):
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        return reader.Execute()

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image = self.read_image(self.images_list[index])
        image = Normalization(image)
        cast_filter = sitk.CastImageFilter()
        cast_filter.SetOutputPixelType(self.bit)
        image = cast_filter.Execute(image)

        if self.masks_list:
            label = self.read_image(self.masks_list[index])
            # Keep mask binary: threshold and cast (no intensity normalization)
            label_np = sitk.GetArrayFromImage(label)
            label_np = (label_np > 0).astype(np.float32)
            label = sitk.GetImageFromArray(label_np)
            label.SetDirection(image.GetDirection())
            label.SetOrigin(image.GetOrigin())
            label.SetSpacing(image.GetSpacing())
        else:
            label = sitk.Image(image.GetSize(), self.bit)
            label.SetOrigin(image.GetOrigin())
            label.SetSpacing(image.GetSpacing())

        if self.resample:
            image = resample_sitk_image(image, spacing=self.new_resolution, interpolator=_interpolator_image)
            if self.masks_list:
                label = resample_sitk_image(label, spacing=self.new_resolution, interpolator='nearest')

        # Padding to at least patch_size
        for d in range(3):
            if image.GetSize()[d] < self.patch_size[d]:
                pad_size = [max(image.GetSize()[i], self.patch_size[i]) for i in range(3)]
                resampler = sitk.ResampleImageFilter()
                resampler.SetSize(pad_size)
                resampler.SetOutputSpacing(image.GetSpacing())
                resampler.SetOutputOrigin(image.GetOrigin())
                resampler.SetOutputDirection(image.GetDirection())
                resampler.SetDefaultPixelValue(0)
                image = resampler.Execute(image)
                label = resampler.Execute(label)

        # Random crop (train) or center crop (val)
        size_img = list(image.GetSize())
        crop = [min(self.patch_size[i], size_img[i]) for i in range(3)]
        if self.train and self.masks_list:
            # Random crop with optional min_pixel on mask
            thr = sitk.BinaryThresholdImageFilter()
            thr.SetLowerThreshold(0.5)
            thr.SetUpperThreshold(1.5)
            mask = thr.Execute(label)
            max_start = [max(0, size_img[i] - crop[i]) for i in range(3)]
            for _ in range(50):
                start = [random.randint(0, max_start[i]) if max_start[i] > 0 else 0 for i in range(3)]
                label_crop = _safe_roi(label, start, crop)
                stat = sitk.StatisticsImageFilter()
                stat.Execute(label_crop)
                if stat.GetSum() >= self.min_pixel or random.random() < self.drop_ratio:
                    image = _safe_roi(image, start, crop)
                    label = _safe_roi(label, start, crop)
                    break
            else:
                start = [max(0, (size_img[i] - crop[i]) // 2) for i in range(3)]
                image = _safe_roi(image, start, crop)
                label = _safe_roi(label, start, crop)
        else:
            start = [max(0, (size_img[i] - crop[i]) // 2) for i in range(3)]
            image = _safe_roi(image, start, crop)
            label = _safe_roi(label, start, crop)

        image_np = np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0)).astype(np.float32)
        label_np = np.transpose(sitk.GetArrayFromImage(label), (2, 1, 0)).astype(np.float32)
        label_np = (label_np > 0.5).astype(np.float32)
        image_np = (image_np / 127.5 - 1.0).astype(np.float32)  # [-1, 1] like CycleGAN
        image_np = image_np[np.newaxis, ...]
        label_np = label_np[np.newaxis, ...]
        return torch.from_numpy(image_np), torch.from_numpy(label_np)
