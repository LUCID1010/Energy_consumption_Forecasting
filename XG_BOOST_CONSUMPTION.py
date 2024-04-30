#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import xgboost as xgb




# In[2]:


input_company = input("Enter the Company Name : ")


# In[3]:


if (input_company == "AEP"):
    df1 = pd.read_csv(r"C:\Users\Mahendra\Downloads\Company_data\AEP_hourly.csv")
elif (input_company == "COMED"):
    df1 = pd.read_csv(r"C:\Users\Mahendra\Downloads\Company_data\COMED_hourly.csv")
elif (input_company == "DAYTON"):
    df1 = pd.read_csv(r"C:\Users\Mahendra\Downloads\Company_data\DAYTON_hourly.csv")
elif (input_company == "PJME"):
    df1 = pd.read_csv(r"C:\Users\Mahendra\Downloads\Company_data\PJME_hourly.csv")

else:
    Print("Choose a correct option")
df = df1.set_index("Datetime")
df.index = pd.to_datetime(df.index)


# In[4]:


print(df1)


# In[5]:


df.plot(style ='.', figsize=(15, 5),  title=f"{input_company} Energy use in MW")


# In[6]:


train = df.loc[df.index < '01-01-2015']
test = df.loc[df.index >= '01-01-2015']

fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label="train data", title="Train/Test Dataset")
test.plot(ax=ax, label="test data")
ax.axvline('01-01-2015', linewidth=2, color="black", ls='--')
ax.legend(['train data', 'test data'])
plt.show()


# In[7]:


def create_features(df):
    
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

df = create_features(df)


# In[8]:


train = create_features(train)
test = create_features(test)

FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
TARGET = 'MW'

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


# In[9]:


reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)


# In[10]:


fi = pd.DataFrame(data=reg.feature_importances_,
             index=reg.feature_names_in_,
             columns=['importance'])
fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
plt.show()


# In[11]:


test['prediction'] = reg.predict(X_test)
df = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)
ax = df[['MW']].plot(figsize=(15, 5))
df['prediction'].plot(ax=ax, style='.')
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Raw Data and Prediction')
plt.show()


# In[12]:


ax = df.loc[(df.index > '07-03-2018') & (df.index < '08-03-2018')]['MW'] \
    .plot(figsize=(15, 5), title='Last Month Of Data')
df.loc[(df.index > '07-03-2018') & (df.index < '08-03-2018')]['prediction'] \
    .plot(style='.')
plt.legend(['Actual Data','Prediction'])
plt.show()


# In[13]:


score = np.sqrt(mean_squared_error(test['MW'], test['prediction']))
print(f'RMSE Score on Test set: {score:0.2f}')


# In[14]:


test['error'] = np.abs(test[TARGET] - test['prediction'])
test['date'] = test.index.date
test.groupby(['date'])['error'].mean().sort_values(ascending=False).head(10)


# In[15]:


data = pd.DataFrame()
data['ds'] = (df.index)
data['y'] = df1['MW']
data.head()


# In[16]:


# Define the number of future time steps to predict (one year)
num_future_steps = 365 * 24  # Assuming hourly data

# Get the last timestamp in the dataset
last_timestamp = df.index[-1]

# Create a date range for the future time steps
future_timestamps = pd.date_range(start=last_timestamp, periods=num_future_steps+1, freq='H')[1:]

# Create features for the future timestamps
future_features = create_features(pd.DataFrame(index=future_timestamps))

# Predict using the XGBoost model
future_predictions = reg.predict(future_features[FEATURES])

# Create a DataFrame for the future predictions
future_df = pd.DataFrame(index=future_timestamps, data={'prediction': future_predictions})

# Plot the forecasted data
ax = df[['MW']].plot(figsize=(15, 5))
future_df['prediction'].plot(ax=ax, style='.')
plt.legend(['Historical Data', 'Forecasted Data'])
plt.title('Forecast for the Next One Year')
plt.show()


# In[17]:


df.plot(style ='.', figsize=(15, 5),  title=f"{input_company} Energy use in MW")


# In[ ]:





# In[ ]:





# In[ ]:




