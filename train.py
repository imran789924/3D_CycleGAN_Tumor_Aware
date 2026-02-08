import sys
from utils.NiftiDataset import *
import utils.NiftiDataset as NiftiDataset
from torch.utils.data import DataLoader
from options.train_options import TrainOptions
# from logger import *
import time
from models import create_model
from utils.visualizer import Visualizer
from utils.save_slices import save_epoch_slices
from test import inference
from torch.utils.data.dataloader import default_collate


def collate_optional_mask(batch):
    """Collate (image, label) or (image, label, mask). If mask is None for all, third element is None."""
    if not batch:
        return default_collate(batch)
    first = batch[0]
    if len(first) == 2:
        return default_collate(batch)
    # len == 3: (image, label, mask or None)
    imgs = default_collate([b[0] for b in batch])
    labels = default_collate([b[1] for b in batch])
    masks = [b[2] for b in batch]
    if all(m is None for m in masks):
        return imgs, labels, None
    masks = default_collate(masks)
    return imgs, labels, masks


if __name__ == '__main__':

    # -----  Loading the init options -----
    opt = TrainOptions().parse()

    # -----  Transformation and Augmentation process for the data  -----
    min_pixel = int(opt.min_pixel * ((opt.patch_size[0] * opt.patch_size[1] * opt.patch_size[2]) / 100))
    trainTransforms = [
                NiftiDataset.Resample(opt.new_resolution, opt.resample),
                NiftiDataset.Augmentation(),
                NiftiDataset.Padding((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2])),
                NiftiDataset.RandomCrop((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]), opt.drop_ratio, min_pixel),
                ]

    mask_dir = getattr(opt, 'mask_dir', '') or ''
    train_set = NifitDataSet(opt.data_path, which_direction='AtoB', transforms=trainTransforms, shuffle_labels=True, train=True, mask_dir=mask_dir)
    print('lenght train list:', len(train_set))
    train_loader = DataLoader(
        train_set,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True,
        collate_fn=collate_optional_mask,
        persistent_workers=opt.workers > 0,
        prefetch_factor=4 if opt.workers > 0 else None,
    )

    # -----------------------------------------------------
    model = create_model(opt)  # creation of the model
    model.setup(opt)
    if opt.epoch_count > 1:
        model.load_networks(opt.epoch_count)
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(train_loader):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()

        # Save middle slices every save_slices_freq to reduce I/O (default: every 10 epochs; set 1 for every epoch)
        save_slices_freq = getattr(opt, 'save_slices_freq', 10)
        if epoch % save_slices_freq == 0 or epoch == 1:
            try:
                visuals = model.get_current_visuals()
                save_epoch_slices(visuals, model.save_dir, epoch)
            except Exception as e:
                print('Warning: could not save epoch slices: %s' % e)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()










