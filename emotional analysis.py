#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


# In[2]:


data=pd.read_excel(r'emotionsdatavignesh.xlsx',
                  dtype='str')


# In[3]

totaly=data['emotions']
totalX=data['text']

# In[4]:


totaly=totaly[0:40]
totalX=totalX[0:40]


# In[5]:


tf=CountVectorizer()
tf.fit(totalX)
X_train_tf1=tf.transform(totalX).toarray()

# In[6]:


X_train_tf1.shape
print(X_train_tf1)

# In[ ]:





# In[7]:


X_train, X_test, y_train, y_test=train_test_split(X_train_tf1,totaly,
                                                 test_size=0.2,
                                                 random_state=102)


# In[8]:


mnb= MultinomialNB()


# In[9]:


mnb.fit(X_train, y_train)


# In[10]:


mnb.score(X_train,y_train)


# In[11]:

mnb.score(X_test,y_test)


# In[12]:


mnb1= RandomForestClassifier()


# In[13]:

mnb1.fit(X_train,y_train)


# In[14]:


mnb1.score(X_train,y_train)


# In[15]:

mnb1.score(X_test,y_test)


# In[16]:


#msg_test=tf.transform(["sit beside me "]).toarray()
#mnb.predict(msg_test)


# In[17]:


#mnb1.predict(msg_test)


# In[ ]:


while(1):
     print("plese enter the sentence ")
     option = str(input("your sentence is : "))
     msg_test=tf.transform([option]).toarray()
     print(option)
     s=mnb1.predict(msg_test)
     t=mnb.predict(msg_test)
     t=t.tolist()
     s=s.tolist()
     print("multinomial naivebase ")
     print(t)
     print(" ")
     print("random forest ")
     print(s)
     print(" ")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




