#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split


# In[5]:


X = pd.read_csv('C:/Users/vivek/Pictures/DataSet/housing price train.csv',index_col = 'Id')
X_test_full =pd.read_csv('C:/Users/vivek/Pictures/DataSet/housing price test.csv',index_col = 'Id')


# In[6]:


X.dropna(axis =0,subset = ['SalePrice'],inplace = True)


# In[7]:


y = X.SalePrice


# In[8]:


X.drop(['SalePrice'],axis = 1,inplace = True)


# In[9]:


X_train_full,X_valid_full,y_train,y_valid = train_test_split(X,y,train_size = 0.8,test_size = 0.2,random_state = 0)


# In[10]:


low_cardinality_cols = [cols for cols in X_train_full.columns
                       if X_train_full[cols].nunique() < 10 and X_train_full[cols].dtype == 'object'] 


# In[11]:


numerical_cols = [cols for cols in X_train_full.columns
                 if X_train_full[cols].dtype in ['int64','float64']]


# In[12]:


my_cols = low_cardinality_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()


# In[13]:


# One Hot Encode The Data (to shorten we use pandas)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)


# In[14]:


# Aligning The X_valid and X_test with X_train
X_train,X_valid = X_train.align(X_valid,join = 'left',axis = 1)
X_train,X_test =  X_train.align(X_test,join = 'left',axis = 1)


# In[15]:


from xgboost import XGBRegressor
my_model_1 = XGBRegressor()
my_model_1.fit(X_train,y_train)


# In[17]:


from sklearn.metrics import mean_absolute_error
predictions_1 = my_model_1.predict(X_valid)
mae = mean_absolute_error(predictions_1,y_valid)
print("MAE in my_model_1 approach",mae)


# In[20]:


my_model_2 = XGBRegressor(n_estimators = 1000,learning_rate = 0.05,early_stopping_rounds = 5,n_jobs =4)
my_model_2.fit(X_train,y_train,
              eval_set = [(X_valid,y_valid)],
              verbose = False)
predictions_2 = my_model_2.predict(X_valid)
mae2 = mean_absolute_error(predictions_2,y_valid)
print("MAE in my_model_2 approach",mae2)


# In[21]:


# Try setting wrong parameters and get worst mae:
my_model_3 = XGBRegressor(n_estimators = 1)
my_model_3.fit(X_train,y_train)
predictions_3 = my_model_3.predict(X_valid)
mae3 = mean_absolute_error(predictions_3,y_valid)
print("MAE in my_model_3 approach",mae3)


# In[ ]:




