
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
import os
import pandas as pd


# In[2]:

out=pd.read_csv('E:\\1.csv')


# In[3]:

out


# In[4]:

X,y=out.iloc[:,0:-1],out.iloc[:,-1]


# In[5]:

X


# In[6]:

y.value_counts()


# In[7]:

X=np.asarray(X)
y=np.asarray(y)


# In[8]:

#proprocessing the string s
for i in range(0,4177):
    if X[i][0]=='M':
        X[i][0]=0
    elif X[i][0]=='F':
        X[i][0]=1
    else :
        X[i][0]=2


# In[9]:

print(X)


# In[10]:

t=0
for i in range(0,4177):
    if y[i] in [1,2,25,26,29,24,27]:
        y[i]=1
    else :
        t+=1


# In[11]:

#over sampling
from imblearn.over_sampling import SMOTE 


# In[12]:

sm=SMOTE()
X_res,y_res=sm.fit_sample(X,y)


# In[13]:

y_i=pd.Series(y_res)
y_i.value_counts()


# In[14]:

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_res,y_res)
clf.feature_importances_


# In[15]:

X,y=out.iloc[:,1:-1],out.iloc[:,-1]


# In[ ]:

sm=SMOTE()
X_res1,y_res1=sm.fit_sample(X,y)


# In[ ]:

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_res1,y_res1, test_size=0.30, random_state=42)


# In[ ]:

from sklearn.metrics import accuracy_score
reg = linear_model.LinearRegression()
reg.fit(X_train,y_train)
pred5=reg.predict(X_test)
li=[]
for i in range(0,len(pred5)):
    li.append(round(pred5[i]))
accuracy_score(y_test,li)


# In[ ]:

logreg = linear_model.LogisticRegression(C=1e5)



# In[ ]:

logreg.fit(X_train,y_train)


# In[ ]:

pred=logreg.predict(X_test)


# In[ ]:

from sklearn.metrics import accuracy_score

accuracy_score(y_test,pred)


# In[ ]:

from sklearn.metrics import confusion_matrix


print(confusion_matrix(y_test,pred))


# In[ ]:

from sklearn.ensemble import RandomForestClassifier
clf1 = RandomForestClassifier()
clf1.fit(X_train,y_train)

pred4=clf1.predict(X_test)
accuracy_score(y_test,pred4)


# In[ ]:



