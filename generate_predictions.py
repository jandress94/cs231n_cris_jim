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
from torchvision.datasets import ImageFolder

parser = argparse.ArgumentParser()

parser.add_argument('--test_dir', default='../cs231n_data/test-jpg/')
parser.add_argument('--sub_file', default='../cs231n_data/submission.csv')
parser.add_argument('--label_list_file', default = '../cs231n_data/labels.txt')

parser.add_argument('--save_path', default='../cs231n_data/saved_models/best_model.cris')
parser.add_argument('--save_thresholds_path', default='../cs231n_data/saved_models/best_thresh.npy')

parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--use_gpu', action='store_true')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

label_thresholds = np.array([[ 0.174,  0.157,  0.11,   0.084,  0.125,  0.127,  0.078,  0.187,  0.225,  0.172,  0.049,  0.128,  0.267,  0.056,  0.03,   0.014,  0.273]])

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

  # Now that we have set up the data, it's time to set up the model.
  # For this example we will finetune a densenet-169 model which has been
  # pretrained on ImageNet. We will first reinitialize the last layer of the
  # model, and train only the last layer for a few epochs. We will then finetune
  # the entire model on our dataset for a few more epochs.

  # First load the pretrained densenet-169 model; this will download the model
  # weights from the web the first time you run it.
  model = torchvision.models.densenet169(pretrained=True)

  # Reinitialize the last layer of the model. Each pretrained model has a
  # slightly different structure, but from the densenet class definition
  # we see that the final fully-connected layer is stored in model.classifier:
  # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L111
  num_classes = 17
  model.classifier = nn.Linear(model.classifier.in_features, num_classes)
  model.load_state_dict(torch.load(args.save_path))

  # Cast the model to the correct datatype, and create a loss function for
  # training the model.
  model.type(dtype)

  model.eval()


  #label_thresholds = np.load(args.save_thresholds_path, allow_pickle = False)
  thresholds = torch.Tensor(label_thresholds).type(dtype)
  classes = find_classes(args.label_list_file)

  y_pred = np.zeros((len(test_dset), 17))
  filenames_list = []

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

    # make sure that at least one class is predicted for each
    num_predicted = torch.sum(preds, 1)
    no_preds = num_predicted == 0

    _, indices = torch.max(normalized_scores, 1)
    backup_preds = torch.zeros(preds.size(0), preds.size(1)).byte()
    backup_preds[indices] = no_preds
    preds += backup_preds

    preds = preds.cpu().numpy()

    y_pred[count:count + x.size(0), :] = preds
    filenames_list += filenames
    count += x.size(0)

  y_pred = y_pred.astype(np.int)
  predictions = [' '.join(classes[y_pred_row == 1]) for y_pred_row in y_pred]

  subm = pd.DataFrame()
  subm['image_name'] = filenames_list
  subm['tags'] = predictions
  subm.to_csv(args.sub_file, index=False)


def print_progress(index, total, prompt):
    print('%s %d / %d' % (prompt, index, total))

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
