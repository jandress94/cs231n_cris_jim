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
    def __init__(self, saved_model_params, dtype, model_type = 'resnet'):
        """Load the trained model"""
        super(EncoderCNN, self).__init__()

        if model_type == 'densenet':
            model = models.densenet169(pretrained=True)
            #model.classifier = nn.Linear(model.classifier.in_features, 17)
            self.output_size = model.classifier.in_features
            #model.load_state_dict(torch.load(saved_model_params))

            model.classifier = Identity()
        elif model_type == 'resnet':
            model = models.resnet18(pretrained=True)
            #model.fc = nn.Linear(model.fc.in_features, 17)
            self.output_size = model.fc.in_features
            #model.load_state_dict(torch.load(saved_model_params))

            model.fc = Identity()
        else:
            print('unknown model type: %s' % (model_type))
            sys.exit(1)

        model.type(dtype)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.model = model
        
    def forward(self, images):
        """Extract the image feature vectors."""
        return self.model(images)
    
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, cnn_output_size, num_labels, combined_size):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()

        self.start_vect = nn.Parameter(torch.zeros(1, 1, embed_size), requires_grad = True)
        self.embed = nn.Embedding(num_labels + 1, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, 1, batch_first=True)
        self.linear_lstm = nn.Linear(hidden_size, combined_size)
        self.linear_cnn = nn.Linear(cnn_output_size, combined_size)
        self.linear_final = nn.Linear(combined_size, num_labels + 1)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform(self.start_vect)
        nn.init.xavier_uniform(self.embed.weight)
        nn.init.xavier_uniform(self.linear_lstm.weight)
        self.linear_lstm.bias.data.fill_(0)
        nn.init.xavier_uniform(self.linear_cnn.weight)
        self.linear_cnn.bias.data.fill_(0)
        nn.init.xavier_uniform(self.linear_final.weight)
        self.linear_final.bias.data.fill_(0)
        
    def forward(self, cnn_features, labels, lengths):
        embeddings = self.embed(labels)
        stacked_start = torch.cat([self.start_vect for _ in range(embeddings.size(0))])
        embeddings = torch.cat((stacked_start, embeddings), 1)

        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        unpacked = nn.utils.rnn.pad_packed_sequence(hiddens, batch_first = True)[0]
        unbound = torch.unbind(unpacked, 1)
        combined = [self.linear_lstm(elem) for elem in unbound]
        combined = torch.stack(combined, 1)
        #cnn_features = cnn_features.squeeze()
        projected_image = self.linear_cnn(cnn_features)
        combined += torch.stack([projected_image for _ in range(combined.size(1))], 1)
        
        divided = torch.unbind(nn.functional.relu(combined), 1)
        outputs = [self.linear_final(elem) for elem in divided]

        return torch.stack(outputs, 1)
    
    def sample(self, cnn_features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_labels = []
        inputs = torch.cat([self.start_vect for _ in range(cnn_features.size(0))])
        for i in range(18):                                      # maximum sampling length
            #print('running lstm')
            #print(inputs.size())
            #if states is not None:  print(states)
            hiddens, states = self.lstm(inputs, states)
            #print('done running lstm')
            combined = self.linear_lstm(hiddens.squeeze(1))
            combined += self.linear_cnn(cnn_features)
            outputs = self.linear_final(nn.functional.relu(combined))
            predicted = outputs.max(1)[1]
            sampled_labels.append(predicted)
            inputs = self.embed(predicted)

        sampled_labels = torch.cat(sampled_labels, 1)
        #print('done sampling')
        return sampled_labels.squeeze()
