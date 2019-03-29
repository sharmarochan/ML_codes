# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 08:52:42 2019

@author: tensor19
"""

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






















