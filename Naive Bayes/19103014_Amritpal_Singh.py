#!/usr/bin/env python
# coding: utf-8

# In[263]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


# In[264]:


train_data = pd.read_csv('Titanic_train.csv')
test_data =  pd.read_csv('Titanic_test.csv')
submissions = pd.read_csv('Titanic_gender_submission.csv')
del test_data['Name']
del test_data['PassengerId']
del test_data['Ticket']
del test_data['Cabin']

train_data['Sex'] = train_data['Sex'].replace(["female", "male"], [0, 1])
train_data['Age'] = pd.qcut(train_data['Age'], 10, labels=False)
train_data['Fare'] = pd.qcut(train_data['Fare'], 10, labels=False)

test_data['Sex'] = test_data['Sex'].replace(["female", "male"], [0, 1])
test_data['Age'] = pd.qcut(test_data['Age'], 10, labels=False)
test_data['Fare'] = pd.qcut(test_data['Fare'], 10, labels=False)

test_data = test_data.values

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']


# In[265]:


y_counts=train_data['Survived'].value_counts()
p_y=(y_counts/len(train_data)).to_dict()


# In[266]:


features_prob = []
for i in features:
    count = train_data[i].value_counts()
    features_prob.append((count/len(train_data)).to_dict())


# In[267]:


df_survived=train_data.loc[train_data['Survived'] == 1]
df_died=train_data.loc[train_data['Survived'] == 0]


# In[268]:


conditional_prob_survived = []
for i in features:
    count = df_survived[i].value_counts()
    conditional_prob_survived.append((count/len(df_survived)).to_dict())


# In[269]:


conditional_prob_died = []
for i in features:
    count = df_died[i].value_counts()
    conditional_prob_died.append((count/len(df_died)).to_dict())


# In[ ]:





# In[270]:


def Bayes(row):
    res_survived = p_y[0]
    num = 1
    den = 1
    for i in range(len(row)):
        if math.isnan(row[i]):
            continue
        try:
            num*= conditional_prob_survived[i][row[i]]
            den*= features_prob[i][row[i]]
        except KeyError:
            continue
    res_survived*= num
    res_survived/=den
    
    res_died = p_y[0]
    num = 1
    den = 1
    for i in range(len(row)):
        if math.isnan(row[i]):
            continue
        try:
            num*= conditional_prob_died[i][row[i]]
            den*= features_prob[i][row[i]]
        except KeyError:
            continue
    res_died*= num
    res_died/=den
    
    if res_survived>=res_died:
        return 1
    else:
        return 0
    
    


# In[271]:


Output = []
for row in test_data:
    Output.append(Bayes(row))
    


# In[272]:


x = submissions['Survived'].values
accuracy = 0
for i in range(len(x)):
    if(x[i]==Output[i]):
        accuracy+=1
print('Accuracy : ',accuracy/len(x) * 100)


# In[ ]:




