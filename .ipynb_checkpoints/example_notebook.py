#!/usr/bin/env python
# coding: utf-8

# ### Load Modules

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ### Prepare/collect data

# In[2]:


import os

path = os.listdir('D:/project/brain_tumor/Training/')
classes = {'no_tumor':0, 'pituitary_tumor':1}


# In[3]:


import cv2
X = []
Y = []
for cls in classes:
    pth = 'D:/project/brain_tumor/Training/'+cls
    for j in os.listdir(pth):
        img = cv2.imread(pth+'/'+j, 0)
        img = cv2.resize(img, (200,200))
        X.append(img)
        Y.append(classes[cls])


# In[4]:


X = np.array(X)
Y = np.array(Y)


# In[5]:


np.unique(Y)


# In[6]:


pd.Series(Y).value_counts()


# In[7]:


X.shape


# ### Visualize data

# In[8]:


plt.imshow(X[0], cmap='gray')


# ### Prepare data

# In[9]:


X_updated = X.reshape(len(X), -1)
X_updated.shape


# ### Split Data

# In[10]:


xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10,
                                               test_size=.20)


# In[11]:


xtrain.shape, xtest.shape


# ### Feature Scaling

# In[12]:


print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())
xtrain = xtrain/255
xtest = xtest/255
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())


# ### Feature Selection: PCA

# In[13]:


from sklearn.decomposition import PCA


# In[14]:


print(xtrain.shape, xtest.shape)

pca = PCA(.98)
# pca_train = pca.fit_transform(xtrain)
# pca_test = pca.transform(xtest)
pca_train = xtrain
pca_test = xtest


# ### Train Model

# In[16]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[18]:


sv = SVC()
sv.fit(pca_train, ytrain)


# ### TEST MODEL

# In[21]:


dec = {0:'No Tumor', 1:'Positive Tumor'}


# In[ ]:





# In[22]:


plt.figure(figsize=(12,8))
p = os.listdir('D:/project/brain_tumor/Testing/')
c=1
for i in os.listdir('D:/project/brain_tumor/Testing/no_tumor/')[:9]:
    plt.subplot(3,3,c)
    
    img = cv2.imread('D:/project/brain_tumor/Testing/no_tumor/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1


# In[24]:





# In[23]:


plt.figure(figsize=(12,8))
p = os.listdir('D:/project/brain_tumor/Testing/')
c=1
for i in os.listdir('D:/project/brain_tumor/Testing/pituitary_tumor/')[:16]:
    plt.subplot(4,4,c)
    
    img = cv2.imread('D:/project/brain_tumor/Testing/pituitary_tumor/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1


# In[ ]:




