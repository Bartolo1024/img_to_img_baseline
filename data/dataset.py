import os
import PIL.Image
import torch.utils.data as data


IMG_EXTENSIONS = [".png", ".jpg", ".jpeg"]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def load_img_gray(filepath):
    img = PIL.Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


def load_img_rgb(filepath):
    return PIL.Image.open(filepath).convert('RGB')


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None, mode='gray'):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [image_dir / x for x in os.listdir(image_dir) if is_image_file(x)]
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.load_img_fn = load_img_gray if mode == 'gray' else load_img_rgb

    def __getitem__(self, index):
        input = self.load_img_fn(self.image_filenames[index])
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)
