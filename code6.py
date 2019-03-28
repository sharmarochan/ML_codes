# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 04:27:29 2019

@author: tensor19

text analysis

"""
#https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/

import pandas as pd
import numpy as np


path = r"E:\ML_training\ted-talks\transcripts.csv"

data = pd.read_csv(path)

data.head()

data['transcript'].iloc[0]

data['url'].iloc[0]



data['title'] = data['url'].map(lambda x: x.split("/")[-1])

data.head()


corpus = ['This sentence is one','This sentence is two','This sentence is three']


import sklearn.feature_extraction.text as text

#api 
#TFIDF
tfidf = text.TfidfVectorizer(corpus)

tfidf_matrix =tfidf.fit_transform(corpus)       #first we give a corpus and the we wiill fit the corpus on that corpus

tfidf_matrix.toarray()

tfidf.get_feature_names()

#tfidf is good for context retrival one, it can not give any interpretaion of the words having same meaning.


pd.DataFrame(tfidf_matrix.toarray(),columns=tfidf.get_feature_names()) #lets make it in datafram for better imortance of each word



#let's see other options in the TfidfVectorizer
corpus = data['transcript'].tolist()

# lets remove the stop words from the csv that we have loaded
tfidf = text.TfidfVectorizer(corpus,stop_words="english", max_features=5000)

tfidf_matrix = tfidf.fit_transform(corpus)

tfidf_matrix.shape
#by default python loop over rows, this rows we have is very low level array. make the row as series object as series object and look at the top or other index


words=[]
for row in tfidf_matrix.toarray():
    words.append(pd.Series(row,index=tfidf.get_feature_names()).sort_values(ascending=False).head(5).index)
    

words[0]

#lets make a sentence out of this words

" ".join(["c","b","a"])


#we wil extract the important words to get the sence of the sentence with  help of "imp_words" only
#we have made a new column called imp_words in the data frame

data['imp_terms'] = words

data.head()

#Problem 2 : how to find out we have same document?


# most of the time, similar sentece wil have siilar vector representations or tfids

#cosine distace or cos angle will be small in similar sentences

# we have tfidf matrix, find similarity of a transcript with each other transcript


#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html#sklearn.metrics.pairwise.cosine_similarity

import sklearn.metrics as metrics

similarity_matrics = metrics.pairwise.cosine_similarity(tfidf_matrix)

#mapping between each row with each other row in the table
#we have created each row in to corresponding vectors, now we will find the cosine of each vetor
#mini the angle, max is the similarity



index=[]
for row in similarity_matrics:
    index.append(np.argsort(row)[-5:-1] ) #get all the indexs excluding the first one, for each row
    

index


#dummy argsort
np.argsort([100,2,500,3])  #array([1, 3, 0, 2], dtype=int64) returns indexs not actual data


# lets see the row number which are similar to the index[0]
index[0]

#lets see in data
data.iloc[index[0]]

#lets se the title similarity

data['title'].iloc[0]








##model using tfid's inputs

path =r"E:\ML_training\stack-overflow-questions\stack-overflow-data.csv"


data = pd.read_csv(path)
data.head()

#lets see unique tags

len(data['tags'].unique())   #20

data.shape  #no of ques is 40000

###label the tags column numerically

import sklearn.preprocessing as preprocessing

encoder = preprocessing.LabelEncoder()

y = encoder.fit_transform(data["tags"])

# all the tags that will be given labels
encoder.classes_


corpus = data['post'].tolist()

tfidf = text.TfidfVectorizer(corpus, stop_words="english", max_features=5000)

tfidf_matrix = tfidf.fit_transform(corpus)

tfidf_matrix.shape

import sklearn.linear_model as linear_model

clf = linear_model.LogisticRegression()

clf.fit(tfidf_matrix,y)


import sklearn.model_selection as model_selection

x_train, x_test, y_train, y_test = model_selection.train_test_split(tfidf_matrix,y,test_size=0.20, random_state=42 )


from keras.layers import Conv2D, MaxPool2D, Flatten

from keras.layers import Dense, Activation

from keras.models import Sequential

from keras.utils import to_categorical

y_train.shape
x_train.shape

input_dim = x_train.shape[1]
y_train=to_categorical(y_train,20)
y_train.shape
print(type(x_train))
#x_train=np.array(x_train)

x_test = np.array(x_test)
model = Sequential()

model.add(Dense(10, input_shape=(5000,), activation='relu'))

model.add(Dense(100,activation='relu'))
model.add(Dense(20, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()




history = model.fit(x_train, y_train, epochs=10, verbose=True,batch_size=10)


# convert the x_train in np array
