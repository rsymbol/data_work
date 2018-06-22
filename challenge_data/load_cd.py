import pandas as pd
import util
from zipfile import ZipFile

# load data
# zip_file = ZipFile('raw/raw_data.zip')
# db = pd.read_csv(zip_file.open('ssd_trans.csv'), nrows=100000)

# dfs = {text_file.filename: pd.read_csv(zip_file.open(text_file.filename))
#        for text_file in zip_file.infolist()
#        if text_file.filename.endswith('ssd_trans.csv')}

db = pd.read_csv('challenge_data/train_data_prepared.zip', index_col=False, compression='zip', delimiter=';')

# print(db.head())

# dt = db.dtypes.get(5)
# print(type(# print(db.shape)
# # print(db.dtypes)
# # print(db.info())dt))
# print(dt)
#
# with open("types.csv", "w") as outfile:
#     for i in range(len(db.dtypes)):
#         outfile.write(list(db)[i] + '\t'*5 + str(db.dtypes.get(i)))
#         outfile.write("\n")
#
# exit()

# define basic terms
target_column = 'target'
# dictionaries = {
#     'target_names': {0: 'No', 1: 'Yes'}
# }

X = db.drop(target_column, axis=1)
y = db[target_column]
data = X.assign(target_column=y).rename(columns={"target_column": target_column})
feature_names = data.columns.values

# save prepared data to csv
X.to_csv("data/csv/X.csv", header=False, index=False)
y.to_csv("data/csv/y.csv", header=False, index=False)
# pd.DataFrame(dictionaries.get('target_names')).to_csv("data/csv/target_names.csv", header=False)
pd.DataFrame(feature_names).to_csv("data/csv/feature_names.csv", header=False, index=False)

util.__save_obj('data', data)
# util.__save_obj('dict', dictionaries)

# print summary info
print("-----------------------------")
print("data.shape: " + str(data.shape))
print("X.shape: " + str(X.shape))
print("y.shape: " + str(y.shape))
# print("target_names: " + str(dictionaries.get('target_names')))
print("feature_names: " + str(feature_names))
print("-----------------------------")
