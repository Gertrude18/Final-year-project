#!/usr/bin/env python
# coding: utf-8

# # PACHAGES USED

# In[53]:


import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# # DATASETS 

# In[54]:


data = pd.read_csv(r'C:\Users\HP\Desktop\FYP_ME\Datasets.csv')


# In[55]:


data.head()


# In[56]:


data.tail()


# In[57]:


data.shape


# In[58]:


data['target']


# In[59]:


x = data.drop("target", axis=1)
y = data["target"]


# In[60]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)


# # GETTING THE MODEL

# In[61]:


cls = SVC(kernel = 'linear')


# # FITTING THE MODEL AND DATA

# In[62]:


model = cls.fit(x_train,  y_train)


# # testing 

# In[63]:


model.score(x_test, y_test )


# In[64]:


x = data[['D001', 'D111', 'D777', 'D454']]
y = data["target"]


# In[65]:


cls = SVC(kernel = 'linear')


# In[66]:


model.score(x_test, y_test )


# In[ ]:




