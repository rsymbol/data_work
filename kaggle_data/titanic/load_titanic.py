import pandas as pd
import util

# load data
db = pd.read_csv('kaggle_data/titanic/train.csv')

# define basic terms
target_column = 'Survived'
dictionaries = {
    'target_name': {0: 'No', 1: 'Yes'}
}

X = db.drop(target_column, axis=1)
y = db[target_column]
data = X.assign(target_column = y).rename(columns={"target_column": target_column})
feature_names = data.columns.values

# save prepared data to txt
X.to_csv("data/txt/X.csv", header=False, index=False)
y.to_csv("data/txt/y.csv", header=False, index=False)
pd.DataFrame(dictionaries.get('target_names')).to_csv("data/txt/target_names.csv", header=False, index=False)
pd.DataFrame(feature_names).to_csv("data/txt/feature_names.csv", header=False, index=False)

util.__save_obj('data', data)
util.__save_obj('dict', dictionaries)

# print summary info
print("-----------------------------")
print("data.shape: " + str(data.shape))
print("X.shape: " + str(X.shape))
print("y.shape: " + str(y.shape))
print("target_names: " + str(dictionaries.get('target_names')))
print("feature_names: " + str(feature_names))
print("-----------------------------")
