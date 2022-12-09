#!/usr/bin/env python
# coding: utf-8

# In[63]:


import seaborn as sns
import sklearn as sk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris


# # Introduction: 
# 
# I chose the dataset of FIFA 18, because I used to play the game a lot. This makes it easier to see if there are errors or wrong outcomes. I chose the next variables to predict the Preferred Position:
# - GK diving
# - CB
# - CM
# - CAM
# - ST
# - Positioning
# - Preferred Positions
# 
# I chose these variables, because they provide a general estimate of the position. For example, the variable GK diving. Only goalkeepers have this variable. Another example is Positioning. It is known that attacking players have a higher positioning than defenders. In this way, the algorithm can predict the Preferred Position better.

# # Data cleaning
# 
# I used this code to select the seven variables I want to use.
# 

# In[66]:


import os
res = []

dir_path = os.getcwd()
# Iterate directory
for path in os.listdir():
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        res.append(path)
print(res)
df = pd.read_csv('CompleteDataset.csv')

CleanSet = df[['GK diving','CB','CM','CAM','ST','Positioning','Preferred Positions']]

CleanSet = CleanSet.dropna()

CleanSet.head()


# # Exploratory data analysis
# 
# present relevant graphs and tables with respect to your problem
# 

# In[70]:


sns.distplot(CleanSet['CB'].dropna(), kde=False) #Selecting the rating column. I need to drop the NA's for the plot
plt.title('Rating of Google Play apps')
plt.xlabel('Rating (stars)')
plt.show()


# In[72]:


import seaborn as sns
import os
for dirname, _, filenames in os.walk('CompleteDataset.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
df = pd.read_csv('CompleteDataset.csv')
df = df.assign(Position=df['Preferred Positions'].str.split().str[0])
df = df.assign(Positioning=df['Positioning'].str.split('+').str[0])
df = df.assign(Positioning=df['Positioning'].str.split('-').str[0])
filtered_df = df[['GK diving', 'CB', 'CM', 'CAM', 'ST', 'Positioning', 'Position']]

cleaned_df = filtered_df.dropna()
pd.set_option('display.max_rows', 10)
cleaned_df.head()


# I deleted the GK column because it had NaN values. Some values in the Positioning column contained multiple options (such as ST and RW or ST, RW and LW). I changed those values to the first position to get a graph. The graphs show that a striker is most likely to have a high Positioning level. CB's are lower and the CM's are in between it.

# In[86]:


import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt



ax = sns.lmplot(x='ST', y='Positioning', hue='Position', 
           data=cleaned_df.loc[cleaned_df['Position'].isin(['ST','CM','CB'])], 
           fit_reg=False)

for ax in ax.axes.flat:
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

plt.show()


# In[87]:


import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt



ax = sns.lmplot(x='CM', y='Positioning', hue='Position', 
           data=cleaned_df.loc[cleaned_df['Position'].isin(['ST','CM','CB'])], 
           fit_reg=False)

for ax in ax.axes.flat:
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

plt.show()


# In[88]:


import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt



ax = sns.lmplot(x='CB', y='Positioning', hue='Position', 
           data=cleaned_df.loc[cleaned_df['Position'].isin(['ST','CM','CB'])], 
           fit_reg=False)

for ax in ax.axes.flat:
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

plt.show()


# # Predictive model
# 
# I chose the K-nn model, because this algorithm provides the ability to divide datasets into groups (in this case ST, CB and CM) based on the closest data points. The algorith takes 
# 
# 
# Explain briefly in your own words how the algorithm works
# Split the data set into a training and test set
# Train the model
# Evaluation
# Calculate the accuracy, precision and recall. Describe and interpret the results in a Markdown cell.
# Conclusion

# In[95]:


X = cleaned_df[:, 7:]
y = cleaned_df

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, stratify=y, random_state=5)

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()

clf


# In[ ]:




