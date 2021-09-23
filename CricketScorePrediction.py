#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('ipl.csv')


# In[3]:


df


# In[4]:


df.drop(['mid', 'date', 'venue', 'bowler', 'batsman', 'striker', 'non-striker'], axis='columns', inplace=True)


# In[5]:


df['bat_team'].unique()


# In[6]:


current_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
       'Mumbai Indians', 'Kings XI Punjab',
       'Royal Challengers Bangalore', 'Delhi Daredevils','Sunrisers Hyderabad']


# In[7]:


df = df[(df['bat_team'].isin(current_teams)) & (df['bowl_team'].isin(current_teams))]


# In[8]:


df = df[df['overs']>=5.0]


# In[9]:


df


# In[10]:


encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])


# In[11]:


encoded_df


# In[12]:


target = encoded_df.total
train_data = encoded_df.drop(['total'], axis='columns')


# In[13]:


train_data


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


x_train, x_test, y_train, y_test = train_test_split(train_data, target, test_size=0.2)


# In[16]:


len(x_train)


# In[17]:


from sklearn.linear_model import LogisticRegression


# In[18]:


model = LogisticRegression()


# In[19]:


model.fit(x_train, y_train)


# In[20]:


model.score(x_test, y_test)


# In[33]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV 


# In[34]:


model2 = Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
ridge_regressor=RandomizedSearchCV(model2,parameters,scoring='neg_mean_squared_error',cv=10)
ridge_regressor.fit(x_train,y_train)


# In[38]:


prediction = ridge_regressor.predict(x_test)


# In[40]:



import seaborn as sns
sns.distplot(y_test-prediction)


# In[41]:


from sklearn import metrics
import numpy as np
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# In[42]:


import pickle


# In[44]:


filename = 'Cricket-score-Prediction-lr-model.pkl'
pickle.dump(ridge_regressor, open(filename, 'wb'))


# In[ ]:




