import torch
import torch.nn as nn
import torchvision.models as models


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
        super(DecoderRNN,self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        self.word_embeddings = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers=num_layers,
                            batch_first=False)
        self.linear = nn.Linear(hidden_size,vocab_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1,1,self.hidden_size),
                torch.zeros(1,1,self.hidden_size))

    def forward(self, features, captions):
        embeds = self.word_embeddings(captions)
        inputs = torch.cat((features.unsqueeze(1),embeds),1)

        for i in range(inputs.shape[0]):
            input_x = inputs[i]
            lstm_output,self.hidden = self.lstm(input_x.view(len(input_x),1,-1),self.hidden)
            output = self.linear(lstm_output.view(len(input_x),-1))
            pred_output = nn.functional.softmax(output).view(1,len(input_x),-1)

            if i == 0:
                outputs = pred_output
                continue
            outputs = torch.cat((outputs,pred_output))

        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass
