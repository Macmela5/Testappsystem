#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install flask


# In[2]:


pip install flask-login


# In[3]:


pip install flask-sqlalchemy


# # Import Libraries 

# In[4]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import math
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import joblib
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import warnings; warnings.simplefilter('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[5]:


from flask import Flask


# def create_app():

# In[6]:


app = Flask(__name__)


# In[7]:


app.config["SECRET_KEY"] = "PAMELA"


# return app

# # Load the Dataset and Add headers

# In[8]:


electronics_data=pd.read_csv("ratings_electronics (1).csv",names=['userId', 'productId','Rating','timestamp'])


# In[9]:


electronics_data


# In[10]:


# Display the data

electronics_data.head()


# In[11]:



#Shape of the data
electronics_data.shape


# In[12]:


#Taking subset of the dataset
electronics_data=electronics_data.iloc[:1048576,0:]


# In[13]:


#Check the datatypes
electronics_data.dtypes


# In[14]:


electronics_data.info()


# In[15]:


#Five point summary 

electronics_data.describe()['Rating'].T


# In[16]:


#Find the minimum and maximum ratings
print('Minimum rating is: %d' %(electronics_data.Rating.min()))
print('Maximum rating is: %d' %(electronics_data.Rating.max()))


# The rating of the product range from 0 to 1

# ## Handling Missing values
# 

# In[17]:


#Check for missing values
print('Number of missing values across columns: \n',electronics_data.isnull().sum())


# ## Ratings

# In[18]:


# Check the distribution of the rating
with sns.axes_style('white'):
    g = sns.factorplot("Rating", data=electronics_data, aspect=2.0,kind='count')
    g.set_ylabels("Total number of ratings")


# Most of the people has given the rating of 5

# ## Unique Users and products
# 

# In[19]:


print("Total data ")
print("-"*50)
print("\nTotal no of ratings :",electronics_data.shape[0])
print("Total No of Users   :", len(np.unique(electronics_data.userId)))
print("Total No of products  :", len(np.unique(electronics_data.productId)))


# ## Dropping the TimeStamp Column

# In[20]:


#Dropping the Timestamp column

electronics_data.drop(['timestamp'], axis=1,inplace=True)


# # Analyzing the rating

# In[21]:


#Analysis of rating given by the user 

no_of_rated_products_per_user = electronics_data.groupby(by='userId')['Rating'].count().sort_values(ascending=False)

no_of_rated_products_per_user.head()


# In[22]:


no_of_rated_products_per_user.describe()


# In[23]:


quantiles = no_of_rated_products_per_user.quantile(np.arange(0,1.01,0.01), interpolation='higher')


# In[24]:


plt.figure(figsize=(10,10))
plt.title("Quantiles and their Values")
quantiles.plot()
# quantiles with 0.05 difference
plt.scatter(x=quantiles.index[::5], y=quantiles.values[::5], c='orange', label="quantiles with 0.05 intervals")
# quantiles with 0.25 difference
plt.scatter(x=quantiles.index[::25], y=quantiles.values[::25], c='m', label = "quantiles with 0.25 intervals")
plt.ylabel('No of ratings by user')
plt.xlabel('Value at the quantile')
plt.legend(loc='best')
plt.show()


# In[25]:


print('\n No of rated product more than 50 per user : {}\n'.format(sum(no_of_rated_products_per_user >= 50)) )


# In[26]:


#Getting the new dataframe which contains users who has given 50 or more ratings

new_df=electronics_data.groupby("productId").filter(lambda x:x['Rating'].count() >=50)


# In[27]:


no_of_ratings_per_product = new_df.groupby(by='productId')['Rating'].count().sort_values(ascending=False)

fig = plt.figure(figsize=plt.figaspect(.5))
ax = plt.gca()
plt.plot(no_of_ratings_per_product.values)
plt.title('# RATINGS per Product')
plt.xlabel('Product')
plt.ylabel('No of ratings per product')
ax.set_xticklabels([])

plt.show()


# In[28]:


#Average rating of the product 

new_df.groupby('productId')['Rating'].mean().head()


# In[29]:


new_df.groupby('productId')['Rating'].mean().sort_values(ascending=False).head()


# In[30]:


#Total no of rating for product

new_df.groupby('productId')['Rating'].count().sort_values(ascending=False).head()


# In[31]:


ratings_mean_count = pd.DataFrame(new_df.groupby('productId')['Rating'].mean())


# In[32]:


ratings_mean_count['rating_counts'] = pd.DataFrame(new_df.groupby('productId')['Rating'].count())


# In[33]:


ratings_mean_count.head()


# In[34]:


ratings_mean_count['rating_counts'].max()


# In[35]:


plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_count['rating_counts'].hist(bins=50)


# In[36]:


plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_count['Rating'].hist(bins=50)


# In[37]:


plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
sns.jointplot(x='Rating', y='rating_counts', data=ratings_mean_count, alpha=0.4)


# In[38]:


popular_products = pd.DataFrame(new_df.groupby('productId')['Rating'].count())
most_popular = popular_products.sort_values('Rating', ascending=False)
most_popular.head(30).plot(kind = "bar")


# # Collaberative filtering (Item-Item recommedation)

# In[39]:


from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
import os
from surprise.model_selection import train_test_split


# In[ ]:


#Reading the dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(new_df,reader)


# In[ ]:


#Splitting the dataset
trainset, testset = train_test_split(data, test_size=0.3,random_state=10)


# In[ ]:


# Use user_based true/false to switch between user-based or item-based collaborative filtering
algo = KNNWithMeans(k=5, sim_options={'name': 'pearson_baseline', 'user_based': False})
algo.fit(trainset)


# In[ ]:


# run the trained model against the testset
test_pred = algo.test(testset)


# In[ ]:


test_pred


# In[ ]:


# get RMSE
print("Item-based Model : Test Set")
accuracy.rmse(test_pred, verbose=True)


# # Model-based collaborative filtering system

# In[ ]:



new_df1=new_df.head(10000)
ratings_matrix = new_df1.pivot_table(values='Rating', index='userId', columns='productId', fill_value=0)
ratings_matrix.head()


# As expected, the utility matrix obtaned above is sparce, I have filled up the unknown values wth 0.
# 
# 

# In[ ]:


ratings_matrix.shape


# Transposing the matrix

# In[ ]:


X = ratings_matrix.T
X.head()


# In[ ]:


X.shape


# Unique products in subset of data
# 

# In[ ]:


X1 = X


# In[ ]:


#Decomposing the Matrix
from sklearn.decomposition import TruncatedSVD
SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)
decomposed_matrix.shape


# In[ ]:


#Correlation Matrix

correlation_matrix = np.corrcoef(decomposed_matrix)
correlation_matrix.shape


# In[ ]:


X.index[75]


# Index # of product ID purchased by customer
# 
# 

# In[ ]:


i = "B00000K135"

product_names = list(X.index)
product_ID = product_names.index(i)
product_ID


# Correlation for all items with the item purchased by this customer based on items rated by other customers people who bought the same product

# In[ ]:


correlation_product_ID = correlation_matrix[product_ID]
correlation_product_ID.shape


# Recommending top 25 highly correlated products in sequence
# 
# 

# In[ ]:


Recommend = list(X.index[correlation_product_ID > 0.65])

# Removes the item already bought by the customer
Recommend.remove(i) 

Recommend[0:24]


# Here are the top 10 products to be displayed by the recommendation system to the above customer based on the purchase history of other customers in the website.
# 

# In[ ]:




