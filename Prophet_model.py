#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go

input_company = input("Enter the comapny name : ")

if (input_company == "AEP"):
    df1 = pd.read_csv()"C:\Users\Mahendra\Downloads\Company_data\AEP_hourly.csv"
elif (input_company == "COMED"):
    df1 = pd.read_csv(r"C:\Users\Mahendra\Downloads\Company_data\COMED_hourly.csv")
elif (input_company == "DAYTON"):
    df1 = pd.read_csv(r"C:\Users\Mahendra\Downloads\Company_data\DAYTON_hourly.csv")
elif (input_company == "PJME"):
    df1 = pd.read_csv(r"C:\Users\Mahendra\Downloads\Company_data\PJME_hourly.csv")

else:
    Print("Choose a correct option")



# In[ ]:


df = pd.DataFrame()
df['ds'] = (df1['Datetime'])
df['y'] = df1['MW']
df.head()


# In[ ]:


df.rename(columns={"Datetime": "ds", "MW": "y"}, inplace=True)
df['ds'] = pd.to_datetime(df['ds'])


# In[ ]:


model = Prophet()
model.fit(df)


# In[ ]:


future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
print(forecast)


# In[ ]:


fig1 = model.plot(forecast, include_legend=True)
fig2 = model.plot_components(forecast)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual'))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
fig.update_layout(title='Actual vs Forecast',
                  xaxis_title='Date',
                  yaxis_title='Value')
fig.show()


# In[ ]:




