# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 08:52:42 2019

@author: tensor19
"""



#Data loader
#class having simple emmbedding layer
#News dataset

import pandas as pd
import numpy as np

path = r"E:\ML_training\news-classification\train_data.csv"
data__ = pd.read_csv(path)

data__.head()

y = data__['CATEGORY'].values

x = data__['TITLE'].values

import sklearn.model_selection as model_selection

x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size = 0.20, random_state = 2)

y_train


from sklearn.preprocessing import LabelEncoder

encoder  = LabelEncoder()

y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)



import keras.preprocessing.text as text
import keras.preprocessing.sequence as sequence #for padding


MAX_WORDS = 10000
MAX_LEN = 16

tokenizer = text.Tokenizer(num_words=MAX_WORDS)

tokenizer.fit_on_texts(x_train)


x_train_sequence = tokenizer.texts_to_sequences(x_train)

x_test_sequence = tokenizer.texts_to_sequences(x_test)

x_train_sequence = sequence.pad_sequences(x_train_sequence, maxlen=MAX_LEN)


x_test_sequence = sequence.pad_sequences(x_test_sequence, maxlen=MAX_LEN)


x_train_sequence


import torch
from torch.utils.data import DataLoader, Dataset


class seq_data(Dataset):
    def __init__(self,seq,label):     #overload the classes
        self.x = seq
        self.y = label
        
    def __len__(self):      #we need the len to create batches
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.x[idx] #its  a numpy array
        y = self.y[idx]
        batch = {"X":x, "y":y}
        return batch
    
t = seq_data(seq=x_train_sequence , label=y_train)


next(iter(t))



# make generator whick can emmit batches


train_gen = DataLoader(t, batch_size=32, shuffle=True, drop_last=True)  #RNN needs fix sequence of data so we need to drop the small dataset






#pytorch needs:

# index should be long intergers se make changes before feeding it to pytorch

import torch.nn as nn
import torch.nn.functional as F

class embedding_model(nn.Module):
    def __init__(self):
        super(embedding_model,self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=MAX_WORDS, embedding_dim=100)
        self.linear = nn.Linear(in_features=16*100,out_features= 4)
        
    def forward(self,x):
        x = self.embedding_layer(x)
        x = x.view(-1,16*100)
        x = self.linear(x)
        return x
        
        ##sequence length should be managed by us
        
        # input is a long int
        
        
        
m = embedding_model()

sample  = next(iter(train_gen))


m(sample['X'].long()).shape
        

import torch.optim as optim


criterian = nn.CrossEntropyLoss()

optimizer = optim.Adam(m.parameters())



#train loop

for epoch in range(2):
    for batch in train_gen:
        x = batch['X'].long()
        y = batch['y'].long()
        output = m(x)
        loss = criterian(output,y)
        optimizer.zero_grad()
        loss.backward()
    print("Loss is {}".format(loss.item()))
        
from tqdm import tqdm      
    
def Train(Model,Data_Loader,Opt,Epoch,Criterion):
    Model
    Model.train() # Put the model in training mode
    for batch in tqdm(Data_Loader):
        inputs=batch['X'].long()    #long is not the input required by all models ex cnn
        labels=batch['y'].long()
        outputs=Model(inputs)
        Opt.zero_grad()
        loss=Criterion(outputs,labels)
        loss.backward()
        Opt.step()
    print("Epoch {}: Loss {}".format(Epoch+1,loss.item()))    
    
    
    
def Test(Model,Data_Loader):
    Model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for batch in Data_Loader:
            inputs=batch['X'].long()  #inputs=batch['X'].to(device).long()
            labels=batch['y'].long()
            outputs=Model(inputs)
            test_loss=test_loss+F.cross_entropy(outputs,labels,reduction='sum').item() #sum up batch loss
            pred=outputs.argmax(dim=1,keepdim=True)
            correct=correct+pred.eq(labels.view_as(pred)).sum().item()
    test_loss=test_loss/len(Data_Loader.dataset)
    accuracy=(correct/len(Data_Loader.dataset))*100.0
    print("-"*10)
    print("Test loss {}, Test accuracy {}/{} {}".format(test_loss,correct,len(Data_Loader.dataset),accuracy))    



te = seq_data(seq=x_test_sequence , label=y_test)

test_gen = DataLoader(te, batch_size=32, shuffle=True, drop_last=True)


for epoch in range(2):
    Train(Model = m, Data_Loader=train_gen , Opt=optimizer, Epoch=epoch, Criterion=criterian)
    Test(Model=m, Data_Loader=test_gen)

#error

#gunnvant@jigsawacademy.com







##### training on GPU ###################
    
#device  = torch.device("cuda:0" if tourch.cuda.is_available())