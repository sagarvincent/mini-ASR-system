import pandas as pd
import numpy as np
import torch as t
import torch.nn as nn


class Audio2Embedding(nn.module):
    def __init__(self):
        super(Audio2Embedding,self).__init__()
        self.conv1 = nn.Conv2d(1,32, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(32,64, kernel_size=(3,3))
        self.conv3 = nn.Conv2d(63,128,kernel_size=(3,3))
        self.fc1 = nn.Linear(128*3*3, 512)
    def forward(self,x):
        x = t.relu(self.conv1(x))
        x = t.relu(self.conv2(x))
        x = t.relu(self.conv3(x))
        x = x.view(-1,128*3*3) # --> infer the dimension(arg:-1) and reshapes to 128*3*3
        x = t.relu(self.fc1(x))
        return x

class TransformerEncoder(nn.module):
    def __init__(self, num_heads, hidden_size):
        super(TransformerEncoder(nn.module))
        self.self_attn = nn.MultiheadAttention(num_heads,hidden_size)
        self.ff = nn.Linear(hidden_size,hidden_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self,x):
        x = self.self_attn(x,x)
        x = self.ff(x)
        return x

class LangModel(nn.module):
    def __init__(self,hidden_size,vocab_size):
        super(LangModel,self).__init__()
        self.fc1 = nn.Linear(hidden_size, vocab_size)

    def forward(self,x):
        x = t.relu(self.fc1(x))
        return x

class ASRnet():

    def __init__(self,num_heads, hidden_size, vocab_size) -> None:
        super(ASRnet,self).__int__()
        self.Audio2Embedding = Audio2Embedding()
        self.TransformerEncoder = TransformerEncoder(num_heads, hidden_size)
        self.LangModel = LangModel(vocab_size)

    def forward(self):
        x = self.audio_feature_encoder(x)
        x = self.transformer_encoder(x)
        x = self.language_modeling_head(x)
        return x         

