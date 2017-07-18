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
from torchvision.datasets import ImageFolder
from cnn_rnn_caption_model import *
#from cnn_rnn_model import *

parser = argparse.ArgumentParser()

parser.add_argument('--test_dir', default='../cs231n_data/test-jpg/')
parser.add_argument('--sub_file', default='../cs231n_data/submission.csv')
parser.add_argument('--label_list_file', default = '../cs231n_data/labels.txt')

parser.add_argument('--cnn_load_path', type=str, default='../cs231n_data/saved_rnn_caption_models/best_cnn_caption_model.cris')
#parser.add_argument('--cnn_load_path', type=str, default='../cs231n_data/saved_rnn_models/best_cnn_model.cris')
parser.add_argument('--rnn_load_path', type=str, default='../cs231n_data/saved_rnn_caption_models/best_rnn_caption_model.cris')
#parser.add_argument('--rnn_load_path', type=str, default='../cs231n_data/saved_rnn_models/best_rnn_model.cris')

parser.add_argument('--label_embed_size', type=int, default=32)
parser.add_argument('--lstm_hidden_size', type=int, default=128)
parser.add_argument('--combined_hidden_size', type=int, default=64)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--use_gpu', action='store_true')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def find_classes(label_list_file):
    f = open(label_list_file)
    classes = np.array([line.strip() for line in f])
    f.close()
    return classes

def main(args):

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

  def transform_target_to_1_0_vect(target):
    vect = np.zeros((17,))
    vect[target] = 1
    return vect
  
  # Now that we have set up the data, it's time to set up the model.
  # For this example we will finetune a densenet-169 model which has been
  # pretrained on ImageNet. We will first reinitialize the last layer of the
  # model, and train only the last layer for a few epochs. We will then finetune
  # the entire model on our dataset for a few more epochs.

  # First load the pretrained densenet-169 model; this will download the model
  # weights from the web the first time you run it.
  #model = torchvision.models.densenet169(pretrained=True)
  encoder = EncoderCNN(dtype, model_type = 'densenet')
  encoder.load_state_dict(torch.load(args.cnn_load_path))
  encoder.type(dtype)
  encoder.eval()
  #decoder = DecoderRNN(args.label_embed_size, args.lstm_hidden_size, encoder.output_size, 17, args.combined_hidden_size)
  decoder = DecoderCaptionRNN(args.label_embed_size, args.lstm_hidden_size, encoder.output_size, 17)
  decoder.load_state_dict(torch.load(args.rnn_load_path))
  decoder.type(dtype)
  decoder.eval()

  # Reinitialize the last layer of the model. Each pretrained model has a
  # slightly different structure, but from the densenet class definition
  # we see that the final fully-connected layer is stored in model.classifier:
  # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L111
  num_classes = 17

  classes = find_classes(args.label_list_file)

  y_pred = np.zeros((len(test_dset), 17))
  filenames_list = []
  predictions = []

  count = 0
  for x, filenames in test_loader:
    print_progress(count, len(test_dset), 'Running example')

    x_var = Variable(x.type(dtype), volatile = True)
    preds = decoder.sample(encoder(x_var))

    for i in range(preds.size(0)):
        pred = preds[i].data.cpu().numpy().tolist()
        if 17 in pred:
            ind = pred.index(17)
            pred = pred[:ind]
        predictions.append(' '.join([classes[j] for j in pred]))

    filenames_list += filenames
    count += x.size(0)

  subm = pd.DataFrame()
  subm['image_name'] = filenames_list
  subm['tags'] = predictions
  subm.to_csv(args.sub_file, index=False)


def print_progress(index, total, prompt):
    print('%s %d / %d' % (prompt, index, total))

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
