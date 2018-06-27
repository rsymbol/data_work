import util
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from scipy import stats

# load data
util.reset_data()
db = util.get_db()

# -------------------------------INFORMATION----------------------------------------------------

# common info
print(db.info())
print('------------------------------')
# print(db.describe())
# print('------------------------------')

# count unique values of columns
# uni = {c: len(db[c].unique()) for c in db.columns}
# print('Unique values of columns: {}'.format(uni))
# print('---------------------------------')
# print('Unique values of columns less thran 10: {}'.format({d: uni[d] for d in uni if uni[d] < 10}))
print('---------------------------------')

# unique values of specified columns
# print(db['cash_in_out'].value_counts())
# print('---------------------------------')

# -------------------------------TRANSFORMATION----------------------------------------------------
db = db.fillna(0)

# preprocessing categorical data
cat_val = ['cash_in_out', 'display_type', 'scanner_code_reader']
lb = {}
for k in cat_val:
    lb[k] = LabelBinarizer()
    db[k] = lb[k].fit_transform(db[k])
util.__save_obj('label_binarizer', lb)

scaler = StandardScaler()
db = pd.DataFrame(scaler.fit_transform(db))
util.__save_obj('scaler', scaler)

corr = np.corrcoef(db.iloc[:, :-1], rowvar=False)

print(type(corr))
print(corr.shape)
print(np.where(corr > 0.5))
#
# for i in np.arange(0.0, 1.0, 0.1):
#     print(str(i) + ": " + str((corr > i).count))



# draw heatmap
# sns.set()
# ax = sns.heatmap(corr)
# sns.plt.show()


# print(stats.describe(db))
# print('------------------------------------------------------')

util.__save_obj('data', db)
