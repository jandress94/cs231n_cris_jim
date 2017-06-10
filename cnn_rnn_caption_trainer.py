import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from cnn_rnn_caption_model import EncoderCNN, DecoderCaptionRNN 
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import torchvision.transforms as T
from MultiLabelImageFolder import *
from torch.utils.data import DataLoader

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--cnn_save_path', type=str, default='../cs231n_data/saved_rnn_binary_models/best_cnn_binary_model.cris')
parser.add_argument('--rnn_save_path', type=str, default='../cs231n_data/saved_rnn_binary_models/best_rnn_binary_model.cris')
parser.add_argument('--save_loss_path', default='../cs231n_data/saved_rnn_binary_models/loss.txt')

parser.add_argument('--train_dir', default='../cs231n_data/train-jpg/')
#parser.add_argument('--train_dir', default='../cs231n_data/train-jpg-small/')
parser.add_argument('--train_labels_file', default = '../cs231n_data/train_v2.csv')
#parser.add_argument('--train_labels_file', default = '../cs231n_data/train_v2-small.csv')
parser.add_argument('--label_list_file', default = '../cs231n_data/labels.txt')
parser.add_argument('--val_dir', default='../cs231n_data/val-jpg/')
parser.add_argument('--val_labels_file', default = '../cs231n_data/val_v2.csv')

parser.add_argument('--label_embed_size', type=int, defauly=32)
parser.add_argument('--lstm_hidden_size', type=int, default=128)
parser.add_argument('--num_epochs1', type=int, default=15)
parser.add_argument('--num_epochs2', type=int, default=20)
parser.add_argument('--lr1', type=float, default=1e-3)
parser.add_argument('--lr2', type=float, default=1e-4)
parser.add_argument('--use_gpu', action='store_true')

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=4)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)
    
def main(args):

    if os.path.isfile(args.rnn_save_path):
        print('The rnn model file %s already exists' % (args.rnn_save_path))
        sys.exit(1)

    # Figure out the datatype we will use; this will determine whether we run on
    # CPU or on GPU. Run on GPU by adding the command-line flag --use_gpu
    dtype = torch.FloatTensor
    if args.use_gpu:
        dtype = torch.cuda.FloatTensor
    
    # Image preprocessing
    # For normalization, see https://github.com/pytorch/vision#models
    train_transform = T.Compose([
        T.Scale(256),
        T.RandomSizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),            
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    def transform_target_add_end(target):
        return torch.Tensor(sorted(target) + [17])

    def transform_target_to_padded(target):
        return torch.Tensor(sorted(target) + [17] * (18 - len(target)))
    
    # Build data loader
    train_dset = MultiLabelImageFolder(args.train_dir, args.train_labels_file, args.label_list_file, \
        transform=train_transform, target_transform = transform_target_add_end)

    def collate_fn(data):
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, labels = zip(*data)
        images = torch.stack(images, 0)
        lengths = [len(label) for label in labels]
        targets = torch.zeros(len(labels), max(lengths)).long()
        for i, label in enumerate(labels):
            end = lengths[i]
            targets[i, :end] = label[:end]
        return images, targets, lengths

    train_loader = DataLoader(train_dset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            collate_fn = collate_fn,
                            shuffle=True)


    val_transform = T.Compose([
        T.Scale(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_dset = MultiLabelImageFolder(args.val_dir, args.val_labels_file, args.label_list_file, \
        transform=val_transform, target_transform = transform_target_add_end)

    val_loader = DataLoader(val_dset, 
                    batch_size = args.batch_size,
                    num_workers = args.num_workers)

    # Build the models
    encoder = EncoderCNN(dtype, model_type = 'densenet')
    decoder = DecoderCaptionRNN(args.label_embed_size, args.lstm_hidden_size, encoder.output_size, 17)
    for param in encoder.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = True
 
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    decay_rate = 1.6
    learning_rate = decay_rate * args.lr1
    best_f2 = -np.inf
 
    # Train the Models (just rnn)
    for epoch in range(args.num_epochs1):
        print('Epoch [%d/%d]' % (epoch, args.num_epochs1))

        learning_rate /= decay_rate
        optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
        run_epoch(encoder, decoder, train_loader, optimizer, args.save_loss_path, is_cnn_training = False)
        
        # Save the models
        f2 = check_f2(encoder, decoder, val_loader, dtype, is_cnn_training = False)
        print('Val f2: %f' % (f2))
        if f2 > best_f2:
            best_f2 = f2
            print('found a new best!')
            torch.save(decoder.state_dict(), args.rnn_save_path)
            torch.save(encoder.state_dict(), args.cnn_save_path)

    for param in encoder.parameters():
        param.requires_grad = True

    decay_rate = 1.4
    learning_rate = args.lr2 * decay_rate
    # Train the Model (cnn and rnn)
    for epoch in range(args.num_epochs2):
        print('Epoch [%d/%d]' % (epoch, args.num_epochs2))

        learning_rate /= decay_rate
        optimizer = torch.optim.Adam(list(decoder.parameters()) + list(encoder.parameters()), lr=learning_rate)
        run_epoch(encoder, decoder, train_loader, optimizer, args.save_loss_path, is_cnn_training = True)
        
        # Save the models
        f2 = check_f2(encoder, decoder, val_loader, dtype, is_cnn_training = True)
        print('Val f2: %f' % (f2))
        if f2 > best_f2:
            best_f2 = f2
            print('found a new best!')
            torch.save(decoder.state_dict(), args.rnn_save_path)
            torch.save(encoder.state_dict(), args.cnn_save_path)

    
def run_epoch(encoder, decoder, loader, optimizer, save_loss_path, is_cnn_training = False):
    if is_cnn_training:
        encoder.train()
    decoder.train()
    loss_list = []

    for i, (images, labels, lengths) in enumerate(loader):
        # Set mini-batch dataset
        images = to_var(images)
        labels = to_var(labels)
        # Forward, Backward and Optimize
        decoder.zero_grad()
        encoder.zero_grad()

        features = encoder(images)
        outputs = decoder(features, labels, lengths)
        unbound_labels = torch.unbind(labels, 1)
        unbound_outputs = torch.unbind(outputs, 1)

        T_max = len(unbound_labels)

        log_soft_scores = [-nn.LogSoftmax()(time_step) for time_step in unbound_outputs]
        loss_terms = [torch.gather(log_soft_scores[t], 1, torch.unsqueeze(unbound_labels[t], 1)) for t in range(T_max)]
        masks = [torch.autograd.Variable((torch.Tensor(lengths) > t).float().cuda(), requires_grad = True) for t in range(T_max)]
        masked_loss_terms = sum([loss_terms[t] * masks[t] for t in range(T_max)])
        loss = torch.sum(masked_loss_terms) / torch.sum(sum(masks))

        loss.backward()
        optimizer.step()

        # Print log info
        if i % 10 == 0:
            print('Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (i, len(loader), loss.data[0], np.exp(loss.data[0]))) 

    with open(save_loss_path, "a") as loss_file:
        for loss in loss_list:
            loss_file.write("%f\n" % (loss))

def check_f2(encoder, decoder, loader, dtype, is_cnn_training = False, eps = 1e-8):
    encoder.eval()
    decoder.eval()

    running_f2, num_samples = 0.0, 0
    for i, (images, labels) in enumerate(loader):
        if i % 10 == 0: print('Evaluating minibatch %d / %d' % (i, len(loader)))
        
        images = to_var(images, volatile = True)

        features = encoder(images)
        preds = decoder.sample(features)

        for j in range(images.size(0)):
            pred = preds[j].data.cpu().numpy().tolist()
            if 17 in pred:
                ind = pred.index(17)
                pred = pred[:ind]
            pred = set(pred)

            true_labels = set(labels[j])
            
            true_pos = len(pred & true_labels)
            p = 1.0 * true_pos / (len(pred) + eps)
            r = 1.0 * true_pos / (len(true_labels) + eps)
            beta = 2
            running_f2 += (1.0 + beta**2)*p*r / (beta**2 * p + r + eps)
            num_samples += 1

    if is_cnn_training:
        encoder.train()
    decoder.train()
    return running_f2 / num_samples


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
