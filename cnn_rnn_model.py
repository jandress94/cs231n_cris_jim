import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import sys


class EncoderCNN(nn.Module):
    def __init__(self, saved_model_params, dtype, model_type = 'densenet'):
        """Load the trained model"""
        super(EncoderCNN, self).__init__()

        if model_type == 'densenet':
            model = models.densenet169(pretrained=False)
            model.classifier = nn.Linear(model.classifier.in_features, 17)
            self.output_size = model.classifier.in_features
            model.load_state_dict(torch.load(saved_model_params))

            # Cast the model to the correct datatype
            model.type(dtype)
            model.eval()
        else:
            print('unknown model type: %s' % (model_type))
            sys.exit(1)

        model_layers = list(model.children())[:-1]      # delete the last fc layer.
        self.model = nn.Sequential(*model_layers)
        
    def forward(self, images):
        """Extract the image feature vectors."""
        return self.model(images)
    
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, cnn_output_size, num_labels, combined_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()

        self.start_vect = nn.Parameter(torch.zeros(embed_size), requires_grad = True)
        self.embed = nn.Embedding(num_labels + 1, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
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
        embeddings = torch.cat((self.start_vect, embeddings), 1)

        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)

        combined = self.linear_lstm(hiddens[0])
        combined += self.linear_cnn(cnn_features)

        outputs = self.linear_final(nn.functional.relu(combined))

        return outputs
    
    def sample(self, cnn_features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_labels = []
        inputs = self.start_vect
        for i in range(18):                                      # maximum sampling length
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 

            combined = self.linear_lstm(hiddens.squeeze(1))
            combined += self.linear_cnn(cnn_features)

            outputs = self.linear_final(nn.functional.relu(combined))
            predicted = output.max(1)[1]
            sampled_labels.append(predicted)

            inputs = self.embed(predicted)

        sampled_labels = torch.cat(sampled_labels, 1)                  # (batch_size, 20)
        return sampled_labels.squeeze()