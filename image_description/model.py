import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        #super(DecoderRNN,self).__init__()
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        self.word_embeddings = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers=num_layers,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size,vocab_size)

    def init_hidden(self,batch_size):
        return (torch.zeros(1,batch_size,self.hidden_size),
                torch.zeros(1,batch_size,self.hidden_size))

    def forward(self, features, captions):
        batch_size = features.shape[0]
        seq_length = captions.shape[1]

        embeds = self.word_embeddings(captions[:,:-1])
        #embeds = self.word_embeddings(captions)
        inputs = torch.cat((features.unsqueeze(1),embeds),1)
        lstm_output,hidden = self.lstm(inputs)
        outputs = self.linear(lstm_output)
        #outputs = torch.nn.functional.softmax(outputs)
        
        #outputs = torch.narrow(outputs.view(batch_size,-1),1,outputs.shape[2],outputs.shape[2] * seq_length)
        #outputs = outputs.view(batch_size,seq_length,-1)

        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        outputs = []
        for i in range(max_len):
            lstm_output,states = self.lstm(inputs,states)
            liner_output = self.linear(lstm_output)
            predict_output = torch.argmax(liner_output)
            outputs.append(int(predict_output))
            inputs = self.word_embeddings(predict_output).view(1,1,-1)
            
        return outputs
