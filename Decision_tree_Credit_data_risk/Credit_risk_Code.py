#!/usr/bin/env python
# coding: utf-8

# # Decision Tree Classification

# In[1]:


#Importing necessary dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


# Importing the credit_dataset

credit_data = pd.read_csv(r"C:\Users\kondu\OneDrive\Desktop\Projects\Machine Learning\Decision_tree_Credit_data_risk\credit_risk.csv")

credit_data.head()


# In[10]:


credit_data.info()


# In[12]:


credit_data.shape


# In[14]:


credit_data.isnull().sum()


# In[17]:


credit_data.duplicated().sum()


# In[18]:


credit_data.describe()


# In[20]:


#Setting predictors and target

#Predictors
X = credit_data.columns.drop('class')

#Target
Y = credit_data['class']


# In[21]:


X.shape


# In[23]:


Y.shape


# In[25]:


X


# In[28]:


# Encoding the categorical values

credit_data_encoded = pd.get_dummies(credit_data[X])

print("No of credit data encoded columns: ", len(credit_data_encoded.columns))

credit_data_encoded.columns


# In[30]:


# Splitting the data into train and test sets

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(credit_data_encoded,Y,test_size = 0.15, random_state = 100)

print(X_train.shape,Y_train.shape)


# In[31]:


print(X_test.shape,Y_test.shape)

# Building the Model
# In[33]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state = 1)

model.fit(X_train,Y_train)


# In[38]:


X_train_predictions = model.predict(X_train)

X_test_predictions = model.predict(X_test)



# In[51]:


from sklearn.tree import export_graphviz

import graphviz

dot_data = export_graphviz(model,out_file=None,
                          feature_names = credit_data_encoded.columns,
                          class_names = model.classes_,)
graph = graphviz.Source(dot_data)

graphw


# In[43]:


train_accuracy = model.score(X_train,Y_train)
test_accuracy  = model.score(X_test,Y_test)

print("Train accuracy: ",train_accuracy)
print("Test accuracy: ",test_accuracy)


# In[44]:


# We see Overfitting for the model


# In[58]:


model_1 = DecisionTreeClassifier(min_samples_split = 10, min_impurity_decrease = 0.005)

model_1.fit(X_train,Y_train)

print("Train_accuracy: ", model_1.score(X_train,Y_train))
print("Test_accuracy: ", model_1.score(X_test,Y_test))

# Making predictive system
# In[75]:


input_data = ["<0", 6, "critical/other existing credit", "radio/tv", 1169, "no known savings", ">=7", 4, 
              "male single", "none", 4, "real estate", 67, "none", "own", 2, "skilled", 1, "yes", "yes", "good"]

input_data_df = pd.DataFrame([input_data],columns =['over_draft', 'credit_usage', 'credit_history', 'purpose',
       'current_balance', 'Average_Credit_Balance', 'employment', 'location',
       'personal_status', 'other_parties', 'residence_since',
       'property_magnitude', 'cc_age', 'other_payment_plans', 'housing',
       'existing_credits', 'job', 'num_dependents', 'own_telephone',
       'foreign_worker', 'class'] )
                                                    
df_train_columns = X_train.columns
    
input_data_encoded = pd.get_dummies(input_data_df).reindex(columns = df_train_columns,fill_value = 0)
        
prediction = model_1.predict(input_data_encoded)
    
if prediction == 'good':
    print("Person is in Good class")
else:
    print("Person is in Bad class")


# In[ ]:





# In[ ]:





# In[ ]:




