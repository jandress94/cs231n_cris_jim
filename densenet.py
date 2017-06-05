import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import sys

import numpy as np
import os
#from sklearn.metrics import fbeta_score

import torchvision
import torchvision.transforms as T
from MultiLabelImageFolder import *
from torchvision.datasets import ImageFolder


parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='../cs231n_data/train-jpg/')
#parser.add_argument('--train_dir', default='../cs231n_data/train-jpg-small/')
parser.add_argument('--train_labels_file', default = '../cs231n_data/train_v2.csv')
#parser.add_argument('--train_labels_file', default = '../cs231n_data/train_v2-small.csv')
parser.add_argument('--label_list_file', default = '../cs231n_data/labels.txt')

parser.add_argument('--val_dir', default='../cs231n_data/val-jpg/')
parser.add_argument('--val_labels_file', default='../cs231n_data/val_v2.csv')

parser.add_argument('--save_model_path', default='../cs231n_data/saved_models/best_model.cris')
parser.add_argument('--save_thresholds_path', default='../cs231n_data/saved_models/best_thresh.npy')
parser.add_argument('--save_loss_path', default='../cs231n_data/saved_models/loss.txt')

parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
#parser.add_argument('--num_epochs', default=30, type=int)
parser.add_argument('--num_epochs1', default=5, type=int)
parser.add_argument('--num_epochs2', default=25, type=int)
parser.add_argument('--lr1', default=1e-3, type=float)
parser.add_argument('--lr2', default=1e-4, type=float)
parser.add_argument('--use_gpu', action='store_true')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

label_thresholds = np.zeros((17,))


def main(args):

  if os.path.isfile(args.save_model_path):
    print('The model file %s already exists' % (args.save_model_path))
    sys.exit(1)
  elif os.path.isfile(args.save_thresholds_path):
    print('The thresholds file %s already exists' % (args.save_thresholds_path))
    sys.exit(1)
  elif os.path.isfile(args.save_loss_path):
    print('The loss file %s already exists' % (args.save_loss_path))
    sys.exit(1)

  # Figure out the datatype we will use; this will determine whether we run on
  # CPU or on GPU. Run on GPU by adding the command-line flag --use_gpu
  dtype = torch.FloatTensor
  if args.use_gpu:
    dtype = torch.cuda.FloatTensor

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

  def transform_target_to_1_0_vect(target):
    vect = np.zeros((17,))
    vect[target] = 1
    return vect
  
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

  # Set up a transform to use for validation data at test-time. For validation
  # images we will simply resize so the smaller edge has 224 pixels, then take
  # a 224 x 224 center crop. We will then construct an ImageFolder Dataset object
  # for the validation data, and a DataLoader for the validation set.
  val_transform = T.Compose([
    T.Scale(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ])
  val_dset = MultiLabelImageFolder(args.val_dir, args.val_labels_file, args.label_list_file, \
	transform=val_transform, target_transform = transform_target_to_1_0_vect)
  val_loader = DataLoader(val_dset,
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
  num_classes = len(train_dset.classes)
  model.classifier = nn.Linear(model.classifier.in_features, num_classes)

  # Cast the model to the correct datatype, and create a loss function for
  # training the model.
  model.type(dtype)
  loss_fn = nn.MultiLabelSoftMarginLoss().type(dtype)

  # First we want to train only the reinitialized last layer for a few epochs.
  # During this phase we do not need to compute gradients with respect to the
  # other weights of the model, so we set the requires_grad flag to False for
  # all model parameters, then set requires_grad=True for the parameters in the
  # last layer only.
  
  for param in model.parameters():
    param.requires_grad = False
  for param in model.classifier.parameters():
    param.requires_grad = True
  
  # Construct an Optimizer object for updating the last layer only.
  optimizer = torch.optim.Adam(model.classifier.parameters(), lr = args.lr1)

  # set up to save the best model
  max_f2 = -np.inf

  # Update only the last layer for a few epochs.
  for epoch in range(args.num_epochs1):
    # Run an epoch over the training data.
    print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs1))
    run_epoch(model, loss_fn, train_loader, optimizer, dtype, args.save_loss_path)

    # Check accuracy on the train and val sets.
    val_f2 = check_f2(model, val_loader, dtype, recomp_thresh = True)
    train_f2 = check_f2(model, train_loader, dtype)
    print('Val f2: ', val_f2)
    if val_f2 > max_f2:
        print('found a new best!')
        max_f2 = val_f2
        torch.save(model.state_dict(), args.save_model_path)
        np.save(args.save_thresholds_path, label_thresholds, allow_pickle = False)
    print('Train f2: ', train_f2)
    print()

  # Now we want to finetune the entire model for a few epochs. To do thise we
  # will need to compute gradients with respect to all model parameters, so
  # we flag all parameters as requiring gradients.
  for param in model.parameters():
    param.requires_grad = True
  
  # Construct a new Optimizer that will update all model parameters. Note the
  # small learning rate.
  lr2 = args.lr2

  # Train the entire model for a few more epochs, checking accuracy on the
  # train and validation sets after each epoch.
  for epoch in range(args.num_epochs2):
    print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs2))
    if epoch >= 10  and epoch < 20:
      lr2 = lr2 / 10.0
    elif epoch >= 20:
      lr2 = lr2 / 10.0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr2)
    run_epoch(model, loss_fn, train_loader, optimizer, dtype, args.save_loss_path)

    val_f2 = check_f2(model, val_loader, dtype, recomp_thresh = True)
    train_f2 = check_f2(model, train_loader, dtype)
    print('Val f2: ', val_f2)
    if val_f2 > max_f2:
        print('found a new best!')
        max_f2 = val_f2
        torch.save(model.state_dict(), args.save_model_path)
        np.save(args.save_thresholds_path, label_thresholds, allow_pickle = False)
    print('Train f2: ', train_f2)
    print()



def print_progress(index, collection_len, prompt, print_every = 10, loss = None):
    if index % print_every == 0:
        total = collection_len if isinstance(collection_len, int) else len(collection_len)
        if loss is None: print('%s %d / %d' % (prompt, index, total))
        else: print('%s %d / %d   loss: %f' % (prompt, index, total, loss))

def run_epoch(model, loss_fn, loader, optimizer, dtype, save_loss_path):
  """
  Train the model for one epoch.
  """
  # Set the model to training mode
  model.train()
  mini_index = 0

  running_loss = 0.0
  loss_list = []

  for x, y in loader:
    mini_index += 1
    print_progress(mini_index, loader, 'Running minibatch', print_every = 1, loss = running_loss)

    # The DataLoader produces Torch Tensors, so we need to cast them to the
    # correct datatype and wrap them in Variables.
    #
    # Note that the labels should be a torch.LongTensor on CPU and a
    # torch.cuda.LongTensor on GPU; to accomplish this we first cast to dtype
    # (either torch.FloatTensor or torch.cuda.FloatTensor) and then cast to
    # long; this ensures that y has the correct type in both cases.
    x_var = Variable(x.type(dtype))
    y_var = Variable(y.type(dtype).float())

    # Run the model forward to compute scores and loss.
    scores = model(x_var)
    loss = loss_fn(scores, y_var)

    loss_num = loss.data.cpu().numpy()[0]
    loss_list.append(loss_num)
    if mini_index == 1:
        running_loss = loss_num
    else:
        running_loss = 0.9 * running_loss + 0.1 * loss_num

    # Run the model backward and take a step using the optimizer.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  with open(save_loss_path, "a") as loss_file:
    for loss in loss_list:
        loss_file.write("%f\n" % (loss))

def compute_f2(scores, y, threshold, axis, eps = 1e-8):
  preds = scores >= threshold
  preds = preds.cpu().int()
  y = y.int()
    
  pred_pos = torch.sum(preds, axis).float()
  real_pos = torch.sum(y, axis).float()
  true_pos = torch.sum(y * preds, axis).float()
  
  p = 1.0 * true_pos / (pred_pos + eps)
  r = 1.0 * true_pos / (real_pos + eps)

  beta = 2
  return (1.0 + beta**2)*p*r / (beta**2 * p + r + eps)

def recompute_thresholds(model, loader, dtype, eps = 1e-8):
  scores_list = []
  ys = []

  for x, y in loader:
    x_var = Variable(x.type(dtype), volatile = True)
    scores = model(x_var)
    normalized_scores = torch.sigmoid(scores)
    scores_list.append(normalized_scores)
    ys.append(y)

  scores = torch.cat(scores_list, 0).data
  ys = torch.cat(ys, 0)

  best_thresh = np.zeros((17,))
  best_f2 = -np.ones((17,))

  for t in range(1000):
    print_progress(t, 1000, 'Recomputing thresholds')
    thresh = (1 + t) * 0.001

    f2 = compute_f2(scores, ys, thresh, 0).numpy()

    better_mask = f2 > best_f2
    better_mask = better_mask.astype(np.int)
    best_thresh = (1 - better_mask) * best_thresh + better_mask * thresh
    best_f2 = (1 - better_mask) * best_f2 + better_mask * f2

  return best_thresh


def check_f2(model, loader, dtype, recomp_thresh = False, eps = 1e-8):
  """
  Check the accuracy of the model.
  """
  global label_thresholds
  # Set the model to eval mode
  model.eval()

  if recomp_thresh: 
    label_thresholds = recompute_thresholds(model, loader, dtype)
    print('Computed new thresholds:', label_thresholds)

  running_f2, num_samples = 0.0, 0
  mini_index = 0

  #thresholds = torch.Tensor([0.2625, 0.2375, 0.245, 0.21, 0.205, 0.1625, 0.265, 0.2175, 0.1925, 0.12, 0.2225, 0.14, 0.1375, 0.19, 0.085, 0.0475, 0.0875]).type(dtype)
  thresholds = torch.Tensor(label_thresholds).type(dtype)

  for x, y in loader:
    if thresholds.size() != y.size():
      thresholds = torch.Tensor(label_thresholds).type(dtype)
      thresholds = torch.cat([thresholds for _ in range(x.size(0))], 0)

    mini_index += 1
    print_progress(mini_index, loader, 'Evaluating minibatch')
    # Cast the image data to the correct type and wrap it in a Variable. At
    # test-time when we do not need to compute gradients, marking the Variable
    # as volatile can reduce memory usage and slightly improve speed.
    x_var = Variable(x.type(dtype), volatile=True)

    scores = model(x_var)
    normalized_scores = torch.sigmoid(scores)

    f2 = compute_f2(normalized_scores.data, y, thresholds, 1)
    running_f2 += torch.sum(f2)
    num_samples += x.size(0)

  f2 = running_f2 / num_samples
  return f2


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
