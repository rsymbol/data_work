import util
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

from scipy import stats


def new_dict(lst):
    uni = np.unique(lst)
    return {i: j for i, j in enumerate(uni)}


# load data
db = util.get_db()

#-------------------------------INFORMATION----------------------------------------------------

# common info
print(db.info())
print('------------------------------')
print(db.describe())
print('------------------------------')

# count unique values of columns
print('Unique values of columns: {}'.format({c: len(db[c].unique()) for c in db.columns}))
print('---------------------------------')

# unique values of specified columns
print(db['Pclass'].value_counts())
print('---------------------------------')


#-------------------------------TRANSFORMATION----------------------------------------------------

db_copy = db.copy()

lb_make = LabelEncoder()
db_copy["Sex_"] = lb_make.fit_transform(db_copy["Sex"])
print(db_copy.head())

# lb_style = LabelBinarizer()
# lb_results = lb_style.fit_transform(db_copy["Sex"])
# print(pd.DataFrame(lb_results, columns=lb_style.classes_).head())










# ????
print(new_dict(db['Sex']))

# EXAMPLE replacing some item to needs
# cleanup_nums = {"num_doors": {"four": 4, "two": 2},
#                 "num_cylinders": {"four": 4, "six": 6, "five": 5, "eight": 8,
#                                   "two": 2, "twelve": 12, "three": 3}}
# db.replace(cleanup_nums, inplace=True)



# corr = np.corrcoef(db, rowvar=False)
# # draw heatmap
# sns.set()
# ax = sns.heatmap(corr)
# sns.plt.show()


# print(stats.describe(db))
# print('------------------------------------------------------')
