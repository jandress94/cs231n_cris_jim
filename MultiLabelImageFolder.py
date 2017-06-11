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


def find_classes(label_list_file):
    f = open(label_list_file)
    classes = [line.strip() for line in f]
    f.close()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def get_img_to_labels(labels_file, class_to_idx):
    labels = pd.read_csv(labels_file)
    img_label_dict = {}

    for index, row in labels.iterrows():
        split_labels = row['tags'].split()
        numeric_labels = [class_to_idx[x] for x in split_labels]
        img_label_dict[row['image_name']] = numeric_labels

    return img_label_dict


def make_dataset(dir, labels_file, class_to_idx, augment=False):
    img_label_dict = get_img_to_labels(labels_file, class_to_idx)
    images = []

    for target in os.listdir(dir):
        d = os.path.join(dir, target)

        if is_image_file(d):
            labels = img_label_dict[target.split('.')[0]]
            item = (d, labels)
            if augment:
                if not set([11, 12, 13, 14, 15, 16]).isdisjoint(labels):
                    images += 6 * [item]
                elif not set([10]).isdisjoint(labels):
                    images += 4 * [item]
                elif not set([7, 8, 9]).isdisjoint(labels):
                    images += 2 * [item]
            else:
                images.append(item)

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
class MultiLabelImageFolder(data.Dataset):

    def __init__(self, data_dir, labels_file, label_list_file, transform=None, target_transform=None,
                 loader=default_loader, augment=False):
        classes, class_to_idx = find_classes(label_list_file)
        imgs = make_dataset(data_dir, labels_file, class_to_idx, augment)

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + data_dir + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.data_dir = data_dir
        self.labels_file = labels_file
        self.label_list_file = label_list_file
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)