#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[61]:



import warnings
warnings.filterwarnings('ignore')


# In[62]:


housing = pd.read_csv("Housing.csv")
housing.head()


# In[63]:


housing.shape


# In[64]:


housing.info()


# In[65]:


housing.describe()


# In[66]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[67]:


sns.pairplot(housing)


# In[68]:


plt.figure(figsize=(20, 12))
plt.subplot(2,3,1)
sns.violinplot(x = 'mainroad', y = 'price', data = housing)
plt.subplot(2,3,2)
sns.violinplot(x = 'guestroom', y = 'price', data = housing)
plt.subplot(2,3,3)
sns.violinplot(x = 'basement', y = 'price', data = housing)
plt.subplot(2,3,4)
sns.violinplot(x = 'hotwaterheating', y = 'price', data = housing)
plt.subplot(2,3,5)
sns.violinplot(x = 'airconditioning', y = 'price', data = housing)
plt.subplot(2,3,6)
sns.violinplot(x = 'furnishingstatus', y = 'price', data = housing)
plt.show()


# In[69]:


plt.figure(figsize = (10, 5))
sns.violinplot(x = 'furnishingstatus', y = 'price', hue = 'airconditioning', data = housing)
plt.show()


# In[70]:


varlist =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

# Defining the map function
def binary_map(x):
    return x.map({'yes': 1, "no": 0})

# Applying the function to the housing list
housing[varlist] = housing[varlist].apply(binary_map)


# In[71]:


housing.head()


# In[72]:


status = pd.get_dummies(housing['furnishingstatus'])


# In[73]:


status.head()


# In[74]:


status = pd.get_dummies(housing['furnishingstatus'], drop_first = True)


# In[75]:


housing = pd.concat([housing, status], axis = 1)


# In[76]:


housing.head()


# In[77]:


housing.drop(['furnishingstatus'], axis = 1, inplace = True)


# In[78]:


housing.head()


# In[79]:


from sklearn.model_selection import train_test_split

# We specify this so that the train and test data set always have the same rows, respectively
np.random.seed(0)
df_train, df_test = train_test_split(housing, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[80]:


from sklearn.preprocessing import MinMaxScaler


# In[81]:


scaler = MinMaxScaler()


# In[82]:


# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])


# In[83]:


df_train.head()


# In[84]:


df_train.describe()


# In[85]:


# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (16, 10))
sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")
plt.show()


# In[86]:


plt.figure(figsize=[6,6])
plt.scatter(df_train.area, df_train.price)
plt.show()


# In[87]:


y_train = df_train.pop('price')
X_train = df_train


# In[88]:


housing.columns


# In[89]:


import statsmodels.api as sm
X_train_lm = sm.add_constant(X_train)

lr_1 = sm.OLS(y_train, X_train_lm).fit()

lr_1.params


# In[90]:


print(lr_1.summary())


# In[91]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[92]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[93]:


X = X_train.drop('semi-furnished', 1,)


# In[94]:


# Build a third fitted model
X_train_lm = sm.add_constant(X)

lr_2 = sm.OLS(y_train, X_train_lm).fit()


# In[95]:


print(lr_2.summary())


# In[96]:


# Calculate the VIFs again for the new model

vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[97]:


X = X.drop('bedrooms', 1)


# In[98]:


X_train_lm = sm.add_constant(X)

lr_3 = sm.OLS(y_train, X_train_lm).fit()


# In[99]:


print(lr_3.summary())


# In[100]:


vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[101]:


X = X.drop('basement', 1)


# In[102]:


# Build a fourth fitted model
X_train_lm = sm.add_constant(X)

lr_4 = sm.OLS(y_train, X_train_lm).fit()


# In[103]:


print(lr_4.summary())


# In[104]:


# Calculate the VIFs again for the new model
vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[105]:


y_train_price = lr_4.predict(X_train_lm)


# In[106]:


# Plot the histogram of the error terms
fig = plt.figure()
plt.grid()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label
plt.show()


# In[107]:


num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']

df_test[num_vars] = scaler.transform(df_test[num_vars])


# In[108]:


df_test.describe()


# In[109]:


y_test = df_test.pop('price')
X_test = df_test


# In[110]:


# Adding constant variable to test dataframe
X_test_m4 = sm.add_constant(X_test)


# In[111]:


# Creating X_test_m4 dataframe by dropping variables from X_test_m4

X_test_m4 = X_test_m4.drop(["bedrooms", "semi-furnished", "basement"], axis = 1)


# In[112]:


# Making predictions using the fourth model

y_pred_m4 = lr_4.predict(X_test_m4)


# In[113]:


# Plotting y_test and y_pred to understand the spread

fig = plt.figure()
plt.grid()
plt.scatter(y_test, y_pred_m4)
fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y_test', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)   
plt.show()


# In[114]:


from sklearn.metrics import r2_score


# In[115]:


r2_score_lr_train=0.676
print("R-squared Train:",r2_score_lr_train)


# In[116]:


r2_score_lr_test=round(r2_score(y_test, y_pred_m4),3)
print("R-squared Test:",r2_score_lr_test)


# In[117]:


lr_4.params.sort_values(ascending = False) 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




