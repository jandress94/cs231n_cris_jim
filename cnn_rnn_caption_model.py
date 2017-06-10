import torch
import torch.nn as nn
from torch.nn import init
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import sys

class Identity(nn.Module):
    def forward(self, x):
        return x

class EncoderCNN(nn.Module):
    def __init__(self, dtype, model_type = 'resnet'):
        """Load the trained model"""
        super(EncoderCNN, self).__init__()

        if model_type == 'densenet':
            model = models.densenet169(pretrained=True)
            self.output_size = model.classifier.in_features
            model.classifier = Identity()
        elif model_type == 'resnet':
            model = models.resnet18(pretrained=True)
            self.output_size = model.fc.in_features
            model.fc = Identity()
        else:
            print('unknown model type: %s' % (model_type))
            sys.exit(1)

        model.type(dtype)
        self.model = model
        
    def forward(self, images):
        """Extract the image feature vectors."""
        return self.model(images)
    
    
class DecoderCaptionRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, cnn_output_size, num_labels):
        """Set the hyper-parameters and build the layers."""
        super(DecoderCaptionRNN, self).__init__()

        self.start_vect = nn.Parameter(torch.zeros(1, 1, embed_size), requires_grad = True)
        self.embed = nn.Embedding(num_labels + 1, embed_size)
        self.num_labels = num_labels
        self.lstm = nn.LSTM(embed_size, hidden_size, 1, batch_first=True)
        self.linear_img_to_lstm = nn.Linear(cnn_output_size, hidden_size)
        self.linear_final = nn.Linear(hidden_size, num_labels + 1)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform(self.start_vect)
        nn.init.xavier_uniform(self.embed.weight)
        nn.init.xavier_uniform(self.linear_img_to_lstm.weight)
        self.linear_img_to_lstm.bias.data.fill_(0)
        nn.init.xavier_uniform(self.linear_final.weight)
        self.linear_final.bias.data.fill_(0)
        
    def forward(self, cnn_features, labels, lengths):
        embeddings = self.embed(labels)
        stacked_start = torch.cat([self.start_vect for _ in range(embeddings.size(0))])
        embeddings = torch.cat((stacked_start, embeddings), 1)

        h0 = torch.unsqueeze(self.linear_img_to_lstm(cnn_features), 0)
        c0 = torch.autograd.Variable(torch.zeros(h0.size(0), h0.size(1), h0.size(2)).cuda(), requires_grad = False)

        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed, (h0, c0))
        
        unpacked = nn.utils.rnn.pad_packed_sequence(hiddens, batch_first = True)[0]
        unbound = torch.unbind(unpacked, 1)
        
        combined = [self.linear_final(elem) for elem in unbound]
        combined = torch.stack(combined, 1)
        return combined
    
    def sample(self, cnn_features):
        """Samples captions for given image features (Greedy search)."""
        sampled_labels = []
        inputs = torch.cat([self.start_vect for _ in range(cnn_features.size(0))])

        h0 = torch.unsqueeze(self.linear_img_to_lstm(cnn_features), 0)
        c0 = torch.autograd.Variable(torch.zeros(h0.size(0), h0.size(1), h0.size(2)).cuda(), requires_grad = False)
        states = (h0, c0)

        for i in range(self.num_labels):             # maximum sampling length
            hiddens, states = self.lstm(inputs, states)
            scores = self.linear_final(hiddens.squeeze(1))
            predicted = scores.max(1)[1]
            sampled_labels.append(predicted)
            inputs = self.embed(predicted)

        sampled_labels = torch.cat(sampled_labels, 1)
        
        return sampled_labels.squeeze()

