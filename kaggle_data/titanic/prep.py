import pandas as pd
import numpy as np

# load data
db = pd.read_csv('kaggle_data/titanic/train.csv')
X = db.values[:, 2:]
y = db.values[:, 1]
target_names = np.array(['No', 'Yes']).T
feature_names = np.array(list(db.columns.values))[2:]

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
