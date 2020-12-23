#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import GridSearchCV


# In[2]:


Churn_data=pd.read_csv("D:\Study\Python\scripts\Deep_Learning\Banking_Customer_Churn_Modeling\Churn_Modelling.csv")


# In[3]:


Churn_data.info()


# In[4]:


Churn_data.shape


# In[5]:


Churn_data.isnull().sum()


# In[6]:


Churn_data.duplicated().sum()


# In[7]:


Churn_data['Geography'].value_counts()


# In[8]:


Churn_data['Gender'].value_counts()


# In[9]:


Churn_data['Gender']=pd.get_dummies(Churn_data['Gender'], drop_first=True)


# In[10]:


Geography=pd.get_dummies(Churn_data['Geography'], drop_first=True)


# In[11]:


Churn_data=pd.concat([Churn_data, Geography], axis=1)


# In[12]:


Churn_data=Churn_data.drop(['Geography'], axis=1)


# In[13]:


Churn_data=Churn_data.drop(['RowNumber','CustomerId','Surname'], axis=1)


# In[14]:


Churn_data.head(10)


# In[15]:


X=Churn_data.drop(['Exited'], axis=1)
Y=Churn_data['Exited']


# In[16]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X=scaler.fit_transform(X)


# In[17]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2, random_state=0)


# In[18]:


from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.activations import relu, sigmoid


# # Hyperparameter Tuning to decide number of Neurons and hidden layers:

# In[19]:


def fun(layers, activation):
    model=Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes, input_dim=X_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
    model.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
estimator=KerasClassifier(build_fn=fun, verbose=0)

layers=[(20,),(20,15),(40,30,20)]
activations=['relu','sigmoid']
param_grid=dict(layers=layers, activation=activations, batch_size=[128,256], epochs=[30])
grid=GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5)


# In[20]:


grid_result=grid.fit(X_train,Y_train)


# In[21]:


print(grid_result.best_score_, grid.best_params_)


# In[24]:


grid_result_predict=grid_result.predict(X_test)
grid_result_predict=grid_result_predict>0.6


# In[25]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[26]:


cm=confusion_matrix(Y_test, grid_result_predict)
cm


# In[28]:


accuracy_score(Y_test,grid_result_predict)

