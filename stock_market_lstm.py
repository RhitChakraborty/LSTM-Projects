# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 23:00:12 2020

@author: rhitc
"""

"""Stock MArket Forecasting Using LSTM"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

df=pd.read_csv('AAPL.csv')
#taking the closing price
df=df['close'].reset_index(drop=True)
plt.plot(df)

#Scaling
sc=MinMaxScaler(feature_range=(0,1))
df_sc=sc.fit_transform(df.values.reshape(-1,1))

#splitting
train_size=int(len(df)*0.65)
train_data,test_data=df_sc[:train_size],df_sc[train_size:]

#Creating the dataset for LSTM
#1,2,3,4  5
#2,3,4,5  6
def create_dataset(dataset,time_step=1):
    """
    dataset should be an numpy array
    """
    dataX,dataY=[],[]
    for i in range(len(dataset)-time_step):
        a=dataset[i:i+time_step]
        dataX.append(a)
        dataY.append(dataset[i+time_step])
    return np.asarray(dataX),np.asarray(dataY)

x_train,y_train=create_dataset(train_data,time_step=100)
x_test,y_test=create_dataset(test_data,100)

#### Create  model
from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras.models import Sequential

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse',metrics=['mse'])
print(model.summary())
          
#fitting the model
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100,batch_size=64,verbose=1)


#Prediction
train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

#inverse scaling
train_pred=sc.inverse_transform(train_pred)
test_pred=sc.inverse_transform(test_pred)


#evaluation of the model
train_err=np.sqrt(mean_squared_error(y_train,train_pred))
test_err=np.sqrt(mean_squared_error(y_test,test_pred))
print("Training RMSE:",train_err)
print("Test RMSE:",test_err)



### Plotting
y_train=sc.inverse_transform(y_train)
y_test=sc.inverse_transform(y_test)
y1=[w for x in y_train.tolist() for w in x]
y2=[w for x in y_test.tolist() for w in x]
y1.extend(y2)

yp1=[w for x in train_pred.tolist() for w in x]
yp2=[w for x in test_pred.tolist() for w in x]
yp1.extend(yp2)


plt.plot(y1,label='Actual')
plt.plot(yp1,label='Predicted')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()



#prediction into the future (30days)
    
df_temp=x_test[-1].reshape(1,100,1)
predictions=[]
for i in range(30):
    pred=model.predict(df_temp)
    predictions.append(pred[0])
    df_temp=np.append(df_temp,pred)
    df_temp=df_temp[1:].reshape(1,100,1)

prediction=sc.inverse_transform(np.asarray(predictions))


