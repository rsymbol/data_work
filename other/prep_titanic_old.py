import pandas as pd
import numpy as np

# load data
db = pd.read_csv('kaggle_data/titanic/train.csv')
target_column = 'Survived'
target_names = np.array(['No', 'Yes'])
X = db.loc[:, db.columns != target_column].values
y = db.loc[:, db.columns == target_column].values
feature_names = db.columns[db.columns != target_column].values

# save clean data
pd.DataFrame(X).to_csv("data/X.csv", header=False, index=False)
pd.DataFrame(y).to_csv("data/y.csv", header=False, index=False)
pd.DataFrame(target_names).to_csv("data/target_names.csv", header=False, index=False)
pd.DataFrame(feature_names).to_csv("data/feature_names.csv", header=False, index=False)

# print summary info
print("-----------------------------")
print("X.shape: " + str(X.shape))
print("y.shape: " + str(y.shape))
print("target_names: " + str(target_names))
print("feature_names: " + str(feature_names))
print("-----------------------------")
