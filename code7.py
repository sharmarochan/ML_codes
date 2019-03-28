# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 08:26:44 2019

@author: tensor19

pytourch

text representation

Embedding layers

pytouch uses code object oriented feartures

we will write our own model in a class

details to take care of:  Shape

we can only work with tensors, it will not run with numpy data.

how pytourch compute gradiants 
"""

#some meanings in tfid get lostt like cat & rat because they are similar to each other


import torch

a = torch.tensor([1,2,3,4])

b = torch.tensor([5,6,7,8],dtype=torch.float)



b = b.unsqueeze(0)   #batch of tensor, similar to np.expand_dim
b

b.shape


b = b.squeeze(0)   #take away extra dim

b
b.shape

b.view((2,2))


#gradinet property is enables in the tensors so that we can do back propogations


#we can not do gradient descent on a unbounded values



x1 = torch.tensor([2],requires_grad=True, dtype=torch.float)




x2 = torch.tensor([3],requires_grad=True, dtype=torch.float)


y = x1*x2

y    #tensor([6.], grad_fn=<MulBackward0>)


# this shows that this tensor mutipls back ward



y.backward()

x1.grad.data

x2.grad.data

#pytouch collects the gradient values after each pass, so we need to clear the gradient accumulated during first pass

y  = x1*x2

y   #tensor([6.], grad_fn=<MulBackward0

y.backward()

x1.grad.data     #tensor([6.])


x1.grad.zero_()   #tensor([0.])

#disable gradient tracking for testing purpose, we no longer track the gradient
#this is a context statemnt, we have to write code in this
'''

with torch.no_grad():

    for i in ...
'''





'''
summary upto this: 
    when we do forward pass we can calculte th output, now when we update the weight after iteration 1, calculate loss
    This gradient will be sum of what ever in iteration 1 + in iteration 2. So we do not need previour loss values in updating the values in the weights
    during second iterations.
    
    In RNN lasyers we need to accumulte the gradients, but in sequentional we do not need gradient accumulations.

    
''''
#get data insinde pytourch  (pytourch + oops + deep learning), we have to create custome classes

#we have to overload some default methods

class reliance():
    def __init__(self):
        self.headquaters = "Mumbai"
        self.employees = "many"
        
    def get_info(self):
        s = "The headquaters are in {} and there are {} employees".format(self.headquaters,self.employees)
        print(s)
a = reliance()     
        
a.employees
a.get_info()


print(dir(a))


"""
['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', 
'__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__',
 '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', 
 '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__',
 'employees', 'get_info', 'headquaters']


anything with " " are defaout stored by the python, we are overiding __len__, __getitem__ method


"""

#lets override length methods

class reliance():
    def __init__(self):
        self.headquaters = "Mumbai"
        self.employees = "many"
        
    def get_info(self):
        s = "The headquaters are in {} and there are {} employees".format(self.headquaters,self.employees)
        print(s)
        
    def __len__(self):
        return 2
    
a = reliance()

len(a)

#self is a place holder for the objects itself

# define class in such a way that it inherits the properties of the super class

class RIL():
    def __init__(self):
        self.sector = "telecome"
        self.rev = "alot"
        
    def show_info(self):
        print("Daughter company is in {} sector and makes {} of money".format(self.sector, self.rev))
        

b = RIL()

b.show_info()


#inheritance: inherit from a class and meke our own classes by override some of the fuctions


class RIL(reliance):
    def __init__(self):
        self.sector = "telecome"
        self.rev = "alot"
        
    def show_info(self):
        print("Daughter company is in {} sector and makes {} of money".format(self.sector, self.rev))
        

c = RIL()

# it cam only inherit the funtions defined by the user, if we want the predified methods, we need 
# use super(child,self).__init__()
class RIL(reliance):
    def __init__(self):
        super(RIL,self).__init__()
        self.sector = "telecome"
        self.rev = "alot"
        
    def show_info(self):
        print("Daughter company is in {} sector and makes {} of money".format(self.sector, self.rev))
        

c = RIL()


path = r"E:\ML_training\Iris.csv"

data_ = pd.read_csv(path)


data_.head()



data_['Species'].unique()


data_['Species'] = data_['Species'].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})

data_.head()

from  torch.utils.data import  Dataset, DataLoader
#lets override the default methods

class pytorch_iris(Dataset):
    
    #lets override the init method, we have define the new __init__ without inheritance
    def __init__(self,data_,target_name):
        self.x = data_.drop(target_name,axis=1)
        self.y = data_[target_name]#.map({"Iris-setosa":0,"Iris-virginica":1,"Iris-setosa":2, "Iris-versicolor":3, "Iris-virginica":4})
        
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self,idx):
#        x = self.x.iloc[idx]
        x = self.x.iloc[idx].values
        y = self.y.iloc[idx]
        batch = {'X':x,'Y':y}
        return batch
    
    
d = pytorch_iris(data_ = data_, target_name= 'Species')


next(iter(d))


data_gen = DataLoader(d, batch_size=5, shuffle =True)        

next(iter(data_gen))





#data loader to get input as "tfidf scipy array"

path =r"E:\ML_training\stack-overflow-questions\stack-overflow-data.csv"


data = pd.read_csv(path)

print(type(data))
data.head()

#data = data.drop('Id',axis=1)

import sklearn.preprocessing as preprocessing


import sklearn.feature_extraction.text as text
corpus = data['post'].tolist()
encoder = preprocessing.LabelEncoder()
y  =encoder.fit_transform(data['tags'])

tfidf = text.TfidfVectorizer(corpus,stop_words="english",max_features =5000)
tfidf_matrix = tfidf.fit_transform(corpus)

tfidf_matrix= tfidf_matrix.toarray()




class tfidf_data(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        x = self.x[idx]
        y = self.y[idx]
        batch = {"X":x,"y":y}
        return batch
    
a = tfidf_data(x = tfidf_matrix, y= y)
        
next(iter(a))


###now we should make a fnx to get batches of the data

b = DataLoader(a,batch_size=4,shuffle=True)

next(iter(b))


###build a simple neural net on iris data


next(iter(data_gen))        #error in output




#in pytorch the function that calculate the loss has a activation filter, so we have
#to choose no of neurons based on classes using crossentropy
##init we define whick layer
##


import torch.nn as nn
import torch.nn.functional as F


class simple_model(nn.Module):
    def __init__(self):
        super(simple_model,self).__init__()
        self.layer1 = nn.Linear(in_features=5, out_features=4)                 #Dense layer: infeatures = input shaepe
        self.layer2 = nn.Linear(in_features = 4 , out_features = 3)
        self.layer3 = nn.Linear(in_features = 3 , out_features = 3)
        
    def forward(self,x):
        x = self.layer1(x)
        x = F.tanh(self.layer2(x))      #appling activation
        x = self.layer3(x)
        return x               #model will return everything that has happeded


mod = simple_model()

# lets do senity check and pass one batch

b = next(iter(data_gen))

b['X']
mod(b['X'].float())




#lets train a model

import torch.optim as optim

optimizer = optim.Adam(mod.parameters())

criterian = nn.CrossEntropyLoss()

for epoch in range(10):
    for batch in data_gen:
        x = batch['X'].float()
        y = batch['Y']
        output = mod(x)
        mod.zero_grad()
        loss = criterian(output,y)
        loss.backward()   #gradient
        optimizer.step()      #update the parameter
    print("Loss: {}".format(loss.item()))






