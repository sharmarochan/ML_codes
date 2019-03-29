# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:59:10 2019

@author: Rochan.Sharma

Embedding layer: Sparcity, sequence of words

RNN, LSTM

we need:
    ability to tokenise the text
    we use padding to make the sentece upto max_text limit. some sentences are chopped
    data will be a sequce of number and each word is mapped to a word
    rows = unique words in the corpus, each row represent each uniqu word
    we want to learn vector represetation of the each words, weight are random numbers
    
    sentense------>mapped to integers------->padding--------->emmbedding laye ---------> dense layer ---------------> softmax layer
    
    
    
    each word is represened by a vector this----->(43,3,4,5)
    
    vectors that are learn also learn the semantic meaning.
    
    we lean the cosine distance for each word
    
    embedding layer is non-trainable, keeping the dense and softmax layer trainable.
    
    
    do not remove stop words, so that meaning will not change.
"""

import pandas as pd
import numpy as np


path = r"E:\ML_training\news-classification\train_data.csv"

data = pd.read_csv(path)

data.head(5)

data['CATEGORY'].unique()   #soft max classes, target values

data.shape

import sklearn.model_selection as model_selection
x_train, x_test, y_train, y_test = model_selection.train_test_split(data['TITLE'],data['CATEGORY'], test_size=0.20)

import keras.preprocessing.text as text


MAX_WORDS = 10000   #the maximum number of words to keep, based on word frequency. Only the most common num_words-1 words will be kept
SEQ_LEN  = 16

tokenizer = text.Tokenizer(num_words=MAX_WORDS)


tokenizer.fit_on_texts(x_train)

tokenizer.word_index

x_sequence_train = tokenizer.texts_to_sequences(x_train) #these are just sequences, intergers maps for each word

x_sequence_test = tokenizer.texts_to_sequences(x_test) #these are just sequences, intergers maps for each word



#make all the sequences of same size


import keras.preprocessing.sequence as sequence


x_sequence_train = sequence.pad_sequences(x_sequence_train, maxlen=SEQ_LEN)   #chop or add padding

x_sequence_test = sequence.pad_sequences(x_sequence_test, maxlen=SEQ_LEN)

x_sequence_train.shape   #now the max len is equal to 16, we can feed this kind of sequence in keras

x_sequence_train


#now focus on y_train
y_train

from sklearn.preprocessing import LabelEncoder #its a module


encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)


y_train


from keras.utils.np_utils import to_categorical

y_train =to_categorical(y_train)
y_test = to_categorical(y_test)


y_train



from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten


#one array will have 16 words, each word will have  vector lenth of 100


model = Sequential()

model.add(Embedding(input_dim=MAX_WORDS, output_dim=100, input_length=SEQ_LEN))
model.add(Flatten())

model.add(Dense(units=4, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics = ['accuracy'])

model.fit(x_sequence_train, y_train, batch_size=32, epochs=2, validation_split=0.2)

y_test.shape

x_test.shape





#  50100054068130
#  HDFC0001716



#glove file


#take a word and see the neighbouring words, take the neighbouring words as input. predict the target using neighbours, then during testing take the target and predict the neighbours.

#pre trained word vectors in embedding layers

import numpy as np
path = r"E:\ML_training\glove\glove.6B\glove.6B.100d.txt"




con = open(path,"r", encoding = 'utf-8')  #utf for removing UnicodeDecodeError 


word_index={}

for line in con:
    word = line.split()[0]
    vector = line.split()[1:]
    vector = np.asarray(vector, dtype='float32')
    word_index[word] = vector


word_index['the']   

word_index.get('the') #use this as a lookup
tokenizer.word_index


embed_weight= np.zeros((10000,100))

for word, i in tokenizer.word_index.items():
    if i<10000:
        vector = word_index.get(word)
        if vector is not None:
            embed_weight[i] =vector
            
embed_weight.shape

embed_weight


model = Sequential()

model.add(Embedding(input_dim=MAX_WORDS, output_dim=100, weights = [embed_weight], input_length=SEQ_LEN))
model.add(Flatten())
model.add(Dense(units=4, activation='softmax'))
model.summary()


model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])


model.fit(x_sequence_train, y_train, epochs=4, batch_size=32, validation_split=0.20)

flat=[]
for i in y_test:
    flat.append(np.argmax(i))
    
flat=np.array(flat)

flat

pred=model.predict_classes(x_sequence_test)

pred==flat



np.sum(pred==flat)/len(flat)   #accuracy 0.92



###################################lets do the same with stack overr flow data##########

path = r"E:\ML_training\stack-overflow-questions\stack-overflow-data.csv"

data_ = pd.read_csv(path)


data_.head()



len(data_['tags'].unique() )  #soft max classes, target values






word_index={}

for line in con:
#    print(line)
    word = line.split()[0]
    vector = line.split()[1:]
    vector = np.asarray(vector, dtype='float32')
    word_index[word] = vector



            
embed_weight.shape

embed_weight


model = Sequential()

model.add(Embedding(input_dim=MAX_WORDS, output_dim=100, weights = [embed_weight], input_length=SEQ_LEN))
model.add(Flatten())
model.add(Dense(units=4, activation='softmax'))
model.summary()


model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])


model.fit(x_sequence_train, y_train, epochs=4, batch_size=32, validation_split=0.20)

flat=[]
for i in y_test:
    flat.append(np.argmax(i))
    
flat=np.array(flat)

flat

pred=model.predict_classes(x_sequence_test)

pred==flat



np.sum(pred==flat)/len(flat)   #accuracy 0.91

#embedded layer can be pre-trained or we can train our own



"""

RNN and LSTM

RNN used for document classification

previosly we use embedded layer for the postions of the text. Emmmeding layer is a array of sequeces of words and words are represented by vetors

optput of embedding layer is the sequence of vectors for each word.

state - f(w*x1.u*s0,b)

this is sentece one.    [1,2,3,4]           [(v1,v2)()()()]

x1 = this or 1 or (v1,v2)


s1 (this)--------> s2(is)------------>sentence(s3)------------>s4(one)

next sate depends on previous state. 

recursive weights depends on states. 


weights are not updated untill we do backward pass. the states are re initialised.
"""


from keras.layers import SimpleRNN

model = Sequential()

model.add(Embedding(input_dim=MAX_WORDS, output_dim=100, input_length=SEQ_LEN))

model.add(SimpleRNN(100))

model.add(Dense(units=4, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

model.fit(x_sequence_train, y_train, epochs=20, batch_size=32, validation_split=0.20)

flat=[]
for i in y_test:
    flat.append(np.argmax(i))
    
flat=np.array(flat)

flat

pred=model.predict_classes(x_sequence_test)

pred==flat



np.sum(pred==flat)/len(flat)   #accuracy 0.9148241087069741


###########################  LSTM  #####################################################    

##rather RNN people prefer LSTM layer


#LSTM has 3 depedency states, but in RNN we have 1 depedency states

# si = f(w*x, u*si,v*ci,b)   c= carry states


#input gate, forget state. output gate, carry state


from keras.layers import LSTM

model = Sequential()

model.add(Embedding(input_dim=MAX_WORDS, output_dim=100, input_length=SEQ_LEN))

model.add(LSTM(units=100))

model.add(Dense(units=4, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

model.fit(x_sequence_train, y_train, epochs=2, batch_size=32, validation_split=0.20)




#######################################################################################












