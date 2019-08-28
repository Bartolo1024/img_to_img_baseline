import pathlib
import torch
import numpy as np
import torch.utils.data as data
from . import dataset
import torchvision.transforms as transforms


DEFAULT_DATA_ROOT = pathlib.Path('/media/bartolo/Archive/Datasets/super_res/BSDS300')
DEFAULT_NORM_STATS = dict(std=(0.5, 0.5, 0.5), mean=(0., 0., 0.))
DEFAULT_NORM_STATS = dict(std=(0.5,), mean=(0.,))
DEFAULT_SIZE = 256


def get_dataloader(upscale_factor):
    dset = dataset.DatasetFromFolder(DEFAULT_DATA_ROOT / 'images/train',
                                     input_transform=get_input_tranforms((DEFAULT_SIZE // upscale_factor,
                                                                          DEFAULT_SIZE // upscale_factor)),
                                     target_transform=get_target_tranforms())
    return data.dataloader.DataLoader(dataset=dset, batch_size=4,
                                      collate_fn=lambda s: [torch.stack(el) for el in list(zip(*s))])


def get_input_tranforms(size):
    ret = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(**DEFAULT_NORM_STATS)
    ])
    return ret


def get_target_tranforms():
    ret = transforms.Compose([
        transforms.Resize((DEFAULT_SIZE, DEFAULT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(**DEFAULT_NORM_STATS)
    ])
    return ret


def denormalize(out_img):
    mean = DEFAULT_NORM_STATS['mean']
    std = DEFAULT_NORM_STATS['std']
    return ((out_img * std + mean) * 255).astype(np.uint8)
