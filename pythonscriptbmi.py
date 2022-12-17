# %%
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor


# %% [markdown]
# 

# %%
yrbss = pd.read_csv("C:/Users/jjiamei/Desktop/yrbss.csv") 
yrbss.describe()

# %%
yrbss.columns
print(yrbss.dtypes)
replace_map = {'gender': {'female': 1, 'male': 2, 'other': 0}}
labels = yrbss['gender'].astype('category').cat.categories.tolist()
replace_map_comp = {'gender' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
print(replace_map_comp)
yrbss.replace(replace_map_comp, inplace=True)




# %%
yrbss = yrbss.dropna(axis=0)

# %%
y = yrbss.weight

# %%
yrbss_features = ['height', 'age', 'gender']

# %% [markdown]
# surprisingly, physical_activity_7d lowered the explained variance score.

# %%
X = yrbss[yrbss_features]

# %%
X.describe()

# %%
X.head()

# %%
yrbss_model = DecisionTreeRegressor(random_state=1)
yrbss_model.fit(X, y)


# %%
yrbss_model.predict(X)

# %%
from sklearn.metrics import mean_absolute_error, explained_variance_score
from sklearn.tree import DecisionTreeRegressor

# %%
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.20, random_state = 0)

yrbss_model = DecisionTreeRegressor()
yrbss_model.fit(train_X, train_y)


validation_predictions = yrbss_model.predict(val_X)
print(mean_absolute_error(val_y, validation_predictions))
print(explained_variance_score(val_y, validation_predictions))

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, explained_variance_score
forest_model = RandomForestRegressor(random_state=0)
forest_model.fit(train_X, train_y)
weight_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, weight_preds))
print(explained_variance_score(val_y, weight_preds))

# %% [markdown]
# adding more to my github jjiamei


