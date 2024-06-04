#!/usr/bin/env python
# coding: utf-8

# ## Data Preprocessing-Before Building machine Learning Model

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ## Read Dataset

# In[9]:


df=pd.read_csv('bankloan2.csv')
df


# In[8]:


#Head
df.head()


# In[11]:


#Tail
df.tail()


# ## Sanity check data

# In[13]:


#Shape
df.shape


# In[15]:


#Info
df.info()


# In[16]:


#Finding missing values
df.isnull().sum()


# In[17]:


#Finding duplicate
df.duplicated().sum()


# In[19]:


#Identifying garbage values
for i in df.select_dtypes(include="object").columns:
    print(df[i].value_counts())
    print("***"*10)


# ## Exploratory Data Analysis

# In[22]:


#Descriptive statistics
df.describe().T


# In[23]:


df.describe(include="object")


# In[27]:


#Histogram to understand the distribution
for i in df.select_dtypes(include="number").columns:
    sns.histplot(data=df,x=i)
    plt.show()


# In[28]:


#Boxplotbto identifynoutliers
for i in df.select_dtypes(include="number").columns:
    sns.boxplot(data=df,x=i)
    plt.show()


# In[33]:


df.select_dtypes(include='number').corr()


# ## Outliers treatment

# In[38]:


def wisker(col):
    q1,q3=np.percentile(col,[25,75])
    iqr=q3-q1
    lw=q1-1.5*iqr
    uw=q3+1.5*iqr
    return lw,uw
    


# In[ ]:





# In[35]:


df.columns


# In[ ]:




