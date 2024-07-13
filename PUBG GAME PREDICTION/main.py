#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import catboost as cb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
plt.rcParams["figure.figsize"] = (16,6)


# # Reading the Data

# In[2]:


data = pd.read_csv("PUBG_Game_Prediction_data.csv")
data.head()


# In[3]:


data.shape


# In[4]:


data.info()


# # Data Wrangling

# #### checking for the rows with missing values

# In[5]:


#remove the row with null winprediction
data[data['winPlacePerc'].isnull()]


# In[6]:


data = data.drop(2744604)


# In[7]:


# how many players are there in a match
data['playerJoined'] = data.groupby('matchId')['matchId'].transform('count')


# In[8]:


data.head(15)


# In[9]:


data.shape


# In[10]:


plt.hist(data[data['playerJoined'] >= 75]['playerJoined'])
plt.show()


# ### Analyzing the data 

# In[11]:


data['totaldistance'] = data['swimDistance'] + data['walkDistance'] + data['rideDistance']
data['killswithoutmoving'] = ((data['kills'] > 0) & (data['totaldistance'] == 0))


# In[12]:


data[data['killswithoutmoving'] == True].head(10)


# In[13]:


data = data.drop(data[data['killswithoutmoving'] == True].index)


# In[14]:


data.shape


# ### Extra_ordinary Road Kills

# In[15]:


data = data.drop(data[data['roadKills'] > 5].index)


# ### So many Kills

# In[16]:


sns.countplot(x = data[data['kills'] >= 40]['kills'], data = data)
plt.show()


# In[17]:


data = data.drop(data[data['kills'] > 20].index)


# ### Head Shots

# In[18]:


data['headshotrate'] = data['headshotKills']/data['kills']
data['headshotrate'].fillna(0)


# In[19]:


sns.countplot(x = data[(data['headshotrate'] == 1) & (data['kills'] > 5)]['headshotrate'], data = data)
plt.show()


# In[20]:


data = data.drop(data[(data['headshotrate'] == 1) & (data['kills'] > 5)].index)


# In[21]:


sns.displot(data[data['longestKill']>1000]['longestKill'], bins = 50)
plt.show()


# In[22]:


data = data.drop(data[data['longestKill'] >= 1000].index)


# ### Weapon change

# In[23]:


sns.displot(data[data['weaponsAcquired']>=15]['weaponsAcquired'], bins = 100)
plt.show()


# In[24]:


data = data.drop(data[data['weaponsAcquired'] >= 15].index)


# ## EDA

# In[25]:


new_data = data.drop(['Id', 'matchId', 'groupId', 'killswithoutmoving', 'matchType'], axis = 1)


# In[26]:


new_data.head()


# In[27]:


plt.figure(figsize=(30,50))
sns.heatmap(new_data.corr(), annot=True)
plt.show()


# # Feature Enginnering 

# In[28]:


# Normalizing factor because the match with 64 players and 100 will have different impacts
normalizing_factor = (100 - data['playerJoined'])/100 + 1


# In[29]:


data['killsnorm'] = data['kills']*normalizing_factor
data['damageDealtnorm'] = data['damageDealt']*normalizing_factor
data['matchDurationnorm'] = data['matchDuration']*normalizing_factor
data['maxPlacenorm'] = data['maxPlace']*normalizing_factor

data['healsnboost'] = data['heals'] + data['boosts']
data['assist'] = data['assists'] + data['revives']


# In[30]:


data.columns


# In[31]:


data = data.drop(['Id', 'groupId', 'matchId','assists', 'boosts','damageDealt','heals','kills','matchDuration','maxPlace',
                 'revives','rideDistance','swimDistance','walkDistance'], axis = 1)


# In[32]:


data.head()


# In[33]:


data.head()


# # ML - CATBOOST MODEL

# In[50]:


# Hadling match type and killswithout moving
x = data.drop(['winPlacePerc'], axis = 1)
y = data['winPlacePerc']


# In[51]:


x.head()


# In[52]:


x = pd.get_dummies(x, columns=['matchType', 'killswithoutmoving'])
mapping = {False: 0, True: 1}
for col in x.select_dtypes(include=['bool']).columns:
    x[col] = x[col].map(mapping)


# In[53]:


x.head()


# In[54]:


featuers = x.columns


# In[38]:


x = x.fillna(x.mean())


# In[49]:





# In[39]:


sc = StandardScaler()
x = sc.fit_transform(x)


# In[40]:


x


# In[41]:


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, test_size=0.3, random_state=42)


# In[43]:


train_data = cb.Pool(x_train, y_train)
test_data = cb.Pool(x_test, y_test)


# In[44]:


model = cb.CatBoostRegressor(loss_function='RMSE')


# In[45]:


grid = {
    'iterations': [100, 150],
    'learning_rate': [0.03, 0.1],
    'depth': [2,4,6,8]
}


# In[46]:


model.grid_search(grid, train_data)


# In[55]:


feature_imp = pd.DataFrame()
feature_imp['features'] = featuers
feature_imp['importance'] = model.feature_importances_
feature_imp = feature_imp.sort_values(by = ['importance'], ascending=False)


# In[56]:


sns.barplot(x = feature_imp['features'], y = feature_imp['importance'])
plt.xticks(rotation = 90)
plt.show()


# # Prediction

# In[57]:


pred = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)
r2


# In[58]:


rmse


# ## 
