import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 

class Tsum(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.dec = nn.Sequential(*[
            nn.Embedding(vocab_size, embed_size), 
            nn.LSTM(embed_size, embed_size, bidirectional=True) # think about how choosing hiddensize
        ])
        self.enc = nn.LSTM(embed_size, embed_size)
        self.out = nn.Linear(embed_size, vocab_size)


    def forward(self, x, y):
        embeddings = self.dec(x)
        context = torch.zeros_like(embeddings[0])
        output = []
        while True:
            weights = F.cosine_similarity(embeddings, context, dim=1)
            context, y = self.enc(context, weights)
            output.append(y)
        return torch.cat(output, dim=0)