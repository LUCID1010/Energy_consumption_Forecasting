#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt


# In[2]:


# Read the CSV file

input_company = input("Enter the Company Name : ")

if (input_company == "AEP"):
    df = pd.read_csv(r"C:\Users\Mahendra\Downloads\Company_data\AEP_hourly.csv",index_col='Datetime', parse_dates=True)
elif (input_company == "COMED"):
    df = pd.read_csv(r"C:\Users\Mahendra\Downloads\Company_data\COMED_hourly.csv",index_col='Datetime', parse_dates=True)
elif (input_company == "DAYTON"):
    df = pd.read_csv(r"C:\Users\Mahendra\Downloads\Company_data\DAYTON_hourly.csv",index_col='Datetime', parse_dates=True)
elif (input_company == "PJME"):
    df = pd.read_csv(r"C:\Users\Mahendra\Downloads\Company_data\PJME_hourly.csv",index_col='Datetime', parse_dates=True)

else:
    Print("Choose a correct option")



# In[3]:


# ADF Test for Stationarity
def ad_test(dataset):
    dftest = adfuller(dataset, autolag='AIC')
    print("1. ADF : ", dftest[0])
    print("2. P-Value : ", dftest[1])
    print("3. Num Of Lags : ", dftest[2])
    print("4. Num Of Observations Used For ADF Regression:", dftest[3])
    print("5. Critical Values :")
    for key, val in dftest[4].items():
        print("\t", key, ": ", val)

ad_test(df['MW'])


# In[4]:


# # Auto ARIMA Model Selection
# stepwise_fit = auto_arima(df['MW'], trace=True, suppress_warnings=True)
# print(df.shape)


# In[5]:


train = df.iloc[:int(0.8 * len(df))]
test = df.iloc[int(0.8 * len(df)):]
print(train.shape, test.shape)


# In[6]:


model = ARIMA(train['MW'], order=(5, 1, 3))
model_fit = model.fit()
model_fit.summary()


# In[ ]:


predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1)
plt.figure(figsize=(15,5))
plt.scatter(test.index, test['MW'], label='Actual')
plt.scatter(test.index, predictions, label='Predicted', color='red')
plt.legend()
plt.xlabel('Date')
plt.ylabel('MW')
plt.show()


# In[ ]:




