import util
import pandas as pd
import numpy as np

from sklearn import datasets
from scipy import stats


# load data
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
feature_names = iris.feature_names

#prepare data



#load noclean data
# df = pd.DataFrame(X_train)
#
# print(stats.describe(X_train))
# print('------------------------------------------------------')
# print(df.info())




print("-----------------------------")
print("X.shape: " + str(X.shape))
print("y.shape: " + str(y.shape))
print("target_names: " + str(target_names))
print("feature_names: " + str(feature_names))
print("-----------------------------")
print()







#save clean data
pd.DataFrame(X).to_csv("data/X.csv", header=False, index=False)
pd.DataFrame(y).to_csv("data/y.csv", header=False, index=False)
pd.DataFrame(target_names).to_csv("data/target_names.csv", header=False, index=False)
pd.DataFrame(feature_names).to_csv("data/feature_names.csv", header=False, index=False)


