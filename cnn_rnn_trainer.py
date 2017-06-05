import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from cnn_rnn_model import EncoderCNN, DecoderRNN 
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import torchvision.transforms as T
from MultiLabelImageFolder import *
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--cnn_load_path', type=str, default='../cs231n_data/saved_models_resnet18_06-04/best_model.cris')
parser.add_argument('--load_thresholds_path', default='../cs231n_data/saved_models_resnet18_06-04/best_thresh.npy')
parser.add_argument('--rnn_save_path', type=str, default='../cs231n_data/saved_rnn_models/best_rnn_model.cris')

parser.add_argument('--train_dir', default='../cs231n_data/train-jpg/')
#parser.add_argument('--train_dir', default='../cs231n_data/train-jpg-small/')
parser.add_argument('--train_labels_file', default = '../cs231n_data/train_v2.csv')
#parser.add_argument('--train_labels_file', default = '../cs231n_data/train_v2-small.csv')
parser.add_argument('--label_list_file', default = '../cs231n_data/labels.txt')

parser.add_argument('--label_embed_size', type=int, default=32)
parser.add_argument('--lstm_hidden_size', type=int, default=128)
parser.add_argument('--combined_hidden_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--use_gpu', action='store_true')

parser.add_argument('--batch_size', type=int, default=49)
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

    def transform_target_to_1_0_vect(target):
        return torch.Tensor(sorted(target))
    
    # Build data loader
    train_dset = MultiLabelImageFolder(args.train_dir, args.train_labels_file, args.label_list_file, \
        transform=train_transform, target_transform = transform_target_to_1_0_vect)

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

    # Build the models
    encoder = EncoderCNN(args.cnn_load_path, dtype)
    decoder = DecoderRNN(args.label_embed_size, args.lstm_hidden_size, 
                    encoder.output_size, 17, args.combined_hidden_size, 18)
    
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Loss and Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    
    # Train the Models
    total_step = len(train_loader)
    for epoch in range(args.num_epochs):
        for i, (images, labels, lengths) in enumerate(train_loader):

            #images = Variable(images.type(dtype))
            #print(labels)
            #labels = Variable(torch.Tensor(np.array(labels))).type(dtype)
            #y_var = Variable(y.type(dtype).float())

            
            # Set mini-batch dataset
            images = to_var(images)
            labels = to_var(labels)
            targets = pack_padded_sequence(labels, lengths, batch_first=True)[0]
            
            # Forward, Backward and Optimize
            decoder.zero_grad()
            encoder.zero_grad()
            features = encoder(images)
            outputs = decoder(features, labels, lengths)
            unbound_labels = torch.unbind(labels, 1)
            unbound_outputs = torch.unbind(outputs, 1)
            losses = [loss_fn(unbound_outputs[i], unbound_labels[i]) for i in range(len(unbound_labels))]
            loss = torch.sum(torch.stack(losses, 0))
            loss.backward()
            optimizer.step()

            # Print log info
            if i % 1 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      %(epoch, args.num_epochs, i, total_step, 
                        loss.data[0], np.exp(loss.data[0]))) 
                
            # Save the models
            # if (i+1) % args.save_step == 0:
            #     torch.save(decoder.state_dict(), 
            #                os.path.join(args.model_path, 
            #                             'decoder-%d-%d.pkl' %(epoch+1, i+1)))
            #     torch.save(encoder.state_dict(), 
            #                os.path.join(args.model_path, 
            #                             'encoder-%d-%d.pkl' %(epoch+1, i+1)))
                
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
