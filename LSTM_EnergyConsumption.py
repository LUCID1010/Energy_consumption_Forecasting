#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential

from sklearn.preprocessing import MinMaxScaler
import os
import warnings
warnings.filterwarnings("ignore", message="The name tf.losses.sparse_softmax_cross_entropy is deprecated")


# In[18]:


input_company = input("Enter the Company Name : ")

if (input_company == "AEP"):
    df = pd.read_csv(r"C:\Users\Mahendra\Downloads\Company_data\AEP_hourly.csv")
elif (input_company == "COMED"):
    df = pd.read_csv(r"C:\Users\Mahendra\Downloads\Company_data\COMED_hourly.csv")
elif (input_company == "DAYTON"):
    df = pd.read_csv(r"C:\Users\Mahendra\Downloads\Company_data\DAYTON_hourly.csv")
elif (input_company == "PJME"):
    df = pd.read_csv(r"C:\Users\Mahendra\Downloads\Company_data\PJME_hourly.csv")

else:
    Print("Choose a correct option")



# In[19]:


df1 = df.reset_index()['MW']


# In[20]:


print(df1.shape)
print(df)


# In[21]:


df = df.set_index("Datetime")
df.index = pd.to_datetime(df.index)
df.plot(style=".", title=f"{input_company} Energy use in MW")


# In[22]:


scaler = MinMaxScaler(feature_range=(0,1)) 
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

training_size = int(len(df)*0.7) 
test_size = len(df1)-training_size
train_data, test_data= df1[0:training_size:],df1[training_size:len(df1),:1] 
print(train_data), print(test_data)



# In[23]:


x_train, y_train = [], []
for i in range(len(train_data)-100-1):
    a = train_data[i:(i+100), 0]
    x_train.append(a)
    y_train.append(train_data[i+100, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# In[24]:


x_test, y_test = [], []
for i in range(len(test_data)-100-1):
    a = test_data[i:(i+100), 0]
    x_test.append(a)
    y_test.append(test_data[i+100, 0])
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# In[25]:


model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(100, 1)))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


# In[26]:


model.summary()


# In[27]:


model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=25, batch_size=1000, verbose=1)


# In[28]:


train_predict = model.predict(x_train)
test_predict = model.predict(x_test)


# In[29]:


train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)


# In[30]:


import math
math.sqrt(mean_squared_error(y_train, train_predict))


# In[31]:


math.sqrt(mean_squared_error(y_test, test_predict))


# In[32]:


look_back = 100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back : len(train_predict)+look_back, :] = train_predict
 
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1: len(df1)-1, :] = test_predict

plt.figure(figsize=(15, 5))
plt.scatter(df.index,scaler.inverse_transform(df1),  color="blue", label="Actual Data", marker=".")
plt.scatter(df.index,trainPredictPlot,  label="Train predictions", marker=".", color="green")
plt.scatter(df.index,testPredictPlot,  label="Test Predictions", marker=".", color="red")
plt.title("Actual vs Predicted Data (LSTM)")
plt.legend()
plt.show()


# In[ ]:




