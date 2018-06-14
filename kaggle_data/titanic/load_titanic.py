import pandas as pd
import numpy as np
import util

# load data
db = pd.read_csv('kaggle_data/titanic/train.csv')

# count unique values of columns
print('Unique values of columns: {}'.format({c: len(db[c].unique()) for c in db.columns}))
print('---------------------------------')

# unique values of specified columns
print(db['Pclass'].value_counts())
print('---------------------------------')

# define basic terms
target_column = 'Survived'
dictionaries = {
    'target_name': {0: 'No', 1: 'Yes'}
}
index = db['PassengerId']

X = db.loc[:, db.columns != target_column].values
y = db.loc[:, db.columns == target_column].values
data = np.hstack((X, y))
feature_names = np.append(db.columns[db.columns != target_column].values, target_column)

# save prepared data to txt
db_pd = pd.DataFrame(data=data, index=index, columns=feature_names)
pd.DataFrame(X).to_csv("data/txt/X.csv", header=False, index=False)
pd.DataFrame(y).to_csv("data/txt/y.csv", header=False, index=False)
pd.DataFrame(dictionaries.get('target_names')).to_csv("data/txt/target_names.csv", header=False, index=False)
pd.DataFrame(feature_names).to_csv("data/txt/feature_names.csv", header=False, index=False)

util.__save_obj('db_pd', db_pd)
util.__save_obj('dict', dictionaries)

# print summary info
print("-----------------------------")
print("db_pd.shape: " + str(db_pd.shape))
print("X.shape: " + str(X.shape))
print("y.shape: " + str(y.shape))
print("target_names: " + str(dictionaries.get('target_names')))
print("feature_names: " + str(feature_names))
print("-----------------------------")
