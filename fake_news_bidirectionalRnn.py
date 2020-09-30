# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 19:40:49 2020

@author: rhitc
"""

"""
Fake News Classification Using Bidirectional RNN
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import tensorflow as tf #version 2.1.0
from tensorflow.keras.layers import Embedding,Dense,LSTM,Dropout,Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
#from nltk.stem import WordNetLemmatizer

df=pd.read_csv('train.csv')
df.dropna(inplace=True)

#dependent Features
X=df.drop('label',axis=1)

#independent features
y=df.label


message=X.copy()
message.reset_index(inplace=True)
message['title'][1]

#text preprocessing
ps=PorterStemmer()

corpus=[]
for i in range(0,len(message)):
    review=re.sub('[^A-Za-z]',' ',message['title'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if word not in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)
    
    

## One Hot representation
voc_size=5000
one_hot_repr=[one_hot(sent,voc_size) for sent in corpus]

##Padding
sent_length=20
embedded=pad_sequences(one_hot_repr,padding='pre',maxlen=sent_length)

#creating the model
embed_feature_no=40
model=Sequential()
model.add(Embedding(voc_size,embed_feature_no,input_length=sent_length))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(100)))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
print(model.summary())

#converting to array
X_final=np.array(embedded)
y_final=np.array(y)


##Splitting the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X_final,y_final,random_state=42,test_size=0.33)


## model Training
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=64)

#prediciton
y_pred=model.predict_classes(x_test)


#Evaluation
from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(y_test,y_pred)
print(accuracy_score(y_test,y_pred)*100)
