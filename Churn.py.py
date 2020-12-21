#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


# In[54]:


Churn_data=pd.read_csv("D:\Study\Python\scripts\Deep_Learning\Banking_Customer_Churn_Modeling\Churn_Modelling.csv")


# In[55]:


Churn_data.info()


# In[13]:


Churn_data.shape


# In[15]:


Churn_data.isnull().sum()


# In[18]:


Churn_data.duplicated().sum()


# In[56]:


Churn_data['Geography'].value_counts()


# In[32]:


Churn_data['Gender'].value_counts()


# In[57]:


Churn_data['Gender']=pd.get_dummies(Churn_data['Gender'], drop_first=True)


# In[58]:


Geography=pd.get_dummies(Churn_data['Geography'], drop_first=True)


# In[59]:


Churn_data=pd.concat([Churn_data, Geography], axis=1)


# In[60]:


Churn_data=Churn_data.drop(['Geography'], axis=1)


# In[62]:


Churn_data=Churn_data.drop(['RowNumber','CustomerId','Surname'], axis=1)


# In[63]:


Churn_data.head(10)


# In[65]:


X=Churn_data.drop(['Exited'], axis=1)
Y=Churn_data['Exited']


# In[68]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X=scaler.fit_transform(X)


# In[71]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2, random_state=0)


# In[78]:


from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU, ELU, PReLU,ReLU
from keras.layers import Dropout


# # lets initializing the ANN:

# In[79]:


model=Sequential()              #Empty ANN


# In[86]:


# let create the input layer and first hidden layer:
model.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu', input_dim=11))   # first hidden layer
model.add(Dense(units=6, kernel_initializer='glorot_uniform', activation='relu'))             # second hidden layer
model.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))          # output layer


# In[87]:


# lets compile the ANN:
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[88]:


model_fit=model.fit(X_train,Y_train, validation_split=0.33, epochs=100, batch_size=10)


# In[111]:


model_fit.history.keys()


# In[118]:


# Lets check Accuracy on training and test dataset to get to know overfitting:
plt.plot(model_fit.history['accuracy'])
plt.plot(model_fit.history['val_accuracy'])
plt.title("Model_Accuracy Training vs validation")
plt.xlabel("epoch")
plt.ylabel("Accuracy")
plt.legend(['train','test'], loc='lower right')
plt.show()


# In[105]:


model_predict=model.predict(X_test)
model_predict=(model_predict>0.6)


# In[102]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[108]:


cm=confusion_matrix(Y_test, model_predict)
cm


# In[109]:


accuracy_score(Y_test,model_predict)


# In[ ]:





# In[ ]:





# In[ ]:




