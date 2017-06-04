import torch.utils.data as data

from PIL import Image
import os
import os.path
import pandas as pd

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []

    for target in os.listdir(dir):
        d = os.path.join(dir, target)

        if is_image_file(d):
            images.append((d, target.split('.')[0]))

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    # from torchvision import get_image_backend
    # if get_image_backend() == 'accimage':
    #     return accimage_loader(path)
    # else:
    #     return pil_loader(path)
    return pil_loader(path)

'''
The images should all be in a single directory given by data_dir
The labels for each image should be in the csv given by labels_file
The list of all labels should be in the text file given by label_list_file
'''
class MultiLabelImageFolderTest(data.Dataset):

    def __init__(self, data_dir, transform=None, loader=default_loader):
        imgs = make_dataset(data_dir)

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + data_dir + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.data_dir = data_dir
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path, filename = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, filename

    def __len__(self):
        return len(self.imgs)
