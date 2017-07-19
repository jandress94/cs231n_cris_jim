import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import sys

import numpy as np
import pandas as pd

import torchvision
import torchvision.transforms as T
from MultiLabelImageFolderTest import *
from MultiLabelImageFolder import *
from per_class_utils import *
from affine_transform import *
from torchvision.datasets import ImageFolder

parser = argparse.ArgumentParser()

parser.add_argument('--train_dir', default='../cs231n_data/train-jpg-all/')
parser.add_argument('--train_labels_file', default='../cs231n_data/train_v2-all.csv')
parser.add_argument('--test_dir', default='../cs231n_data/test-jpg/')
parser.add_argument('--sub_file', default='../cs231n_data/submission.csv')
parser.add_argument('--label_list_file', default = '../cs231n_data/labels.txt')

parser.add_argument('--save_path', default='../cs231n_data/saved_models/best_model.cris')
parser.add_argument('--save_thresholds_path', default='../cs231n_data/saved_models/best_thresh.npy')

parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--use_gpu', action='store_true')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

#label_thresholds = np.array([[ 0.174,  0.157,  0.11,   0.084,  0.125,  0.127,  0.078,  0.187,  0.225,  0.172,  0.049,  0.128,  0.267,  0.056,  0.03,   0.014,  0.273]])

def find_classes(label_list_file):
    f = open(label_list_file)
    classes = np.array([line.strip() for line in f])
    f.close()
    return classes

def main(args):
  global label_thresholds

  # Figure out the datatype we will use; this will determine whether we run on
  # CPU or on GPU. Run on GPU by adding the command-line flag --use_gpu
  dtype = torch.FloatTensor
  if args.use_gpu:
    dtype = torch.cuda.FloatTensor

  # Set up a transform to use for validation data at test-time. For validation
  # images we will simply resize so the smaller edge has 224 pixels, then take
  # a 224 x 224 center crop. We will then construct an ImageFolder Dataset object
  # for the validation data, and a DataLoader for the validation set.
  test_transform = T.Compose([
    T.Scale(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ])
  test_dset = MultiLabelImageFolderTest(args.test_dir, transform=test_transform)
  test_loader = DataLoader(test_dset,
                  batch_size=args.batch_size,
                  num_workers=args.num_workers)


  # Use the torchvision.transforms package to set up a transformation to use
  # for our images at training time. The train-time transform will incorporate
  # data augmentation and preprocessing. At training time we will perform the
  # following preprocessing on our images:
  # (1) Resize the image so its smaller side is 256 pixels long
  # (2) Take a random 224 x 224 crop to the scaled image
  # (3) Horizontally flip the image with probability 1/2
  # (4) Convert the image from a PIL Image to a Torch Tensor
  # (5) Normalize the image using the mean and variance of each color channel
  #     computed on the ImageNet dataset.
  train_transform = T.Compose([
    T.Scale(256),
    T.RandomSizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),            
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ])
  
  # You load data in PyTorch by first constructing a Dataset object which
  # knows how to load individual data points (images and labels) and apply a
  # transform. The Dataset object is then wrapped in a DataLoader, which iterates
  # over the Dataset to construct minibatches. The num_workers flag to the
  # DataLoader constructor is the number of background threads to use for loading
  # data; this allows dataloading to happen off the main thread. You can see the
  # definition for the base Dataset class here:
  # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataset.py
  #
  # and you can see the definition for the DataLoader class here:
  # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataloader.py#L262
  #
  # The torchvision package provides an ImageFolder Dataset class which knows
  # how to read images off disk, where the image from each category are stored
  # in a subdirectory.
  #
  # You can read more about the ImageFolder class here:
  # https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
  train_dset = MultiLabelImageFolder(args.train_dir, args.train_labels_file, args.label_list_file, \
    transform=train_transform, target_transform = transform_target_to_1_0_vect)

  
  train_loader = DataLoader(train_dset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=True)

  # Now that we have set up the data, it's time to set up the model.
  # For this example we will finetune a densenet-169 model which has been
  # pretrained on ImageNet. We will first reinitialize the last layer of the
  # model, and train only the last layer for a few epochs. We will then finetune
  # the entire model on our dataset for a few more epochs.

  # First load the pretrained densenet-169 model; this will download the model
  # weights from the web the first time you run it.
  model = torchvision.models.resnet50(pretrained=True)

  # Reinitialize the last layer of the model. Each pretrained model has a
  # slightly different structure, but from the densenet class definition
  # we see that the final fully-connected layer is stored in model.classifier:
  # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L111
  num_classes = 17
  model.fc = nn.Linear(model.fc.in_features, num_classes)
  model.load_state_dict(torch.load(args.save_path))

  # Cast the model to the correct datatype, and create a loss function for
  # training the model.
  model.type(dtype)

  model.eval()

  label_thresholds = recompute_thresholds(model, train_loader, dtype)

  #label_thresholds = np.load(args.save_thresholds_path, allow_pickle = False)
  thresholds = torch.Tensor(label_thresholds).type(dtype)
  classes = find_classes(args.label_list_file)

  y_pred = np.zeros((len(test_dset), 17))
  filenames_list = []

  test_loaders = []
  '''
  test_transform = T.Compose([
        T.Scale(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
      ])
  test_dset = MultiLabelImageFolderTest(args.test_dir, transform=test_transform)
  test_loader = DataLoader(test_dset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers)
  test_loaders.append(test_loader)
  test_transform = T.Compose([
        T.Scale(280),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
      ])
  test_dset = MultiLabelImageFolderTest(args.test_dir, transform=test_transform)
  test_loader = DataLoader(test_dset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers)
  test_loaders.append(test_loader)
  test_transform = T.Compose([
        T.Scale(232),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
      ])
  test_dset = MultiLabelImageFolderTest(args.test_dir, transform=test_transform)
  test_loader = DataLoader(test_dset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers)
  test_loaders.append(test_loader)
  test_transform = T.Compose([
        T.Scale(300),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
      ])
  test_dset = MultiLabelImageFolderTest(args.test_dir, transform=test_transform)
  test_loader = DataLoader(test_dset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers)
  test_loaders.append(test_loader)
  '''

  for i in range(8):
    angle = (i % 4) * 90
    if i > 3:
      test_transform = T.Compose([
        T.Scale(224),
        T.CenterCrop(224),
        T.ToTensor(),
        RandomChoiceRotate(values = [angle], p = [1.0]),
        Transpose(1, 2),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
      ])
    else:
      test_transform = T.Compose([
        T.Scale(224),
        T.CenterCrop(224),
        T.ToTensor(),
        RandomChoiceRotate(values = [angle], p = [1.0]),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
      ])

    test_dset = MultiLabelImageFolderTest(args.test_dir, transform=test_transform)
    test_loader = DataLoader(test_dset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers)
    test_loaders.append(test_loader)

  for i in range(8):
    angle = (i % 4) * 90
    if i > 3:
      test_transform = T.Compose([
        T.Scale(256),
        T.CenterCrop(224),
        T.ToTensor(),
        RandomChoiceRotate(values = [angle], p = [1.0]),
        Transpose(1, 2),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
      ])
    else:
      test_transform = T.Compose([
        T.Scale(256),
        T.CenterCrop(224),
        T.ToTensor(),
        RandomChoiceRotate(values = [angle], p = [1.0]),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
      ])

    test_dset = MultiLabelImageFolderTest(args.test_dir, transform=test_transform)
    test_loader = DataLoader(test_dset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers)
    test_loaders.append(test_loader)

  for i in range(8):
    angle = (i % 4) * 90
    if i > 3:
      test_transform = T.Compose([
        T.Scale(288),
        T.CenterCrop(224),
        T.ToTensor(),
        RandomChoiceRotate(values = [angle], p = [1.0]),
        Transpose(1, 2),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
      ])
    else:
      test_transform = T.Compose([
        T.Scale(288),
        T.CenterCrop(224),
        T.ToTensor(),
        RandomChoiceRotate(values = [angle], p = [1.0]),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
      ])

    test_dset = MultiLabelImageFolderTest(args.test_dir, transform=test_transform)
    test_loader = DataLoader(test_dset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers)
    test_loaders.append(test_loader)

  y_pred_sum = np.zeros((len(test_dset), 17))
  for i, test_loader in enumerate(test_loaders):
    print('Test Time Augmentation ' + str(i))
    count = 0
    for x, filenames in test_loader:
      print_progress(count, len(test_dset), 'Running example')

      x_var = Variable(x.type(dtype), volatile = True)
      scores = model(x_var)
      normalized_scores = torch.sigmoid(scores)

      if thresholds.size(0) != x.size(0):
        thresholds = torch.Tensor(label_thresholds).type(dtype)
        thresholds = torch.cat([thresholds for _ in range(x.size(0))], 0)

      normalized_scores = normalized_scores.data
      preds = normalized_scores >= thresholds
      preds = preds.cpu().numpy()
      # make sure that at least one class is predicted for each
      num_predicted = np.sum(preds, 1, keepdims=True)
      no_preds = num_predicted == 0
      no_preds = np.squeeze(no_preds.astype(np.int))

      indices = np.argmax(normalized_scores.cpu().numpy(), 1)
      backup_preds = np.zeros_like(preds)
      backup_preds[np.arange(len(backup_preds)), indices] = no_preds
      preds += backup_preds

      #preds = preds.numpy()

      y_pred[count:count + x.size(0), :] = preds
      if i == 0:
        filenames_list += filenames
      count += x.size(0)

    y_pred = y_pred.astype(np.int)
    y_pred_sum += y_pred

  y_pred = y_pred_sum / float(len(test_loaders))
  predictions = [' '.join(classes[y_pred_row >= 0.5]) for y_pred_row in y_pred]

  subm = pd.DataFrame()
  subm['image_name'] = filenames_list
  subm['tags'] = predictions
  subm.to_csv(args.sub_file, index=False)

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)