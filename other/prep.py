import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import datasets
from scipy import stats

# load data
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
feature_names = iris.feature_names
y = y.reshape(y.shape[0],1)
db = np.hstack((y, X))
print(db[:5, :])

# data info
print(stats.describe(X))
print('------------------------------------------------------')
print(pd.DataFrame(X).info())

corr = np.corrcoef(db, rowvar=False)
# draw heatmap
sns.set()
ax = sns.heatmap(corr)
sns.plt.show()


# save clean data
# pd.DataFrame(X).to_csv("data/X.csv", header=False, index=False)
# pd.DataFrame(y).to_csv("data/y.csv", header=False, index=False)
# pd.DataFrame(target_names).to_csv("data/target_names.csv", header=False, index=False)
# pd.DataFrame(feature_names).to_csv("data/feature_names.csv", header=False, index=False)

# print summary info
print("-----------------------------")
print("X.shape: " + str(X.shape))
print("y.shape: " + str(y.shape))
print("target_names: " + str(target_names))
print("feature_names: " + str(feature_names))
print("-----------------------------")
