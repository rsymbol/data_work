import util
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats

def new_dict(lst):
    uni = np.unique(lst)
    return {i:j for i,j in enumerate(uni)}

# load data
db = util.get_db()
print(db.info())
print('------------------------------')
print(db.describe())
print('------------------------------')

base = pd.DataFrame()

print(new_dict(db['Sex']))

# corr = np.corrcoef(db, rowvar=False)
# # draw heatmap
# sns.set()
# ax = sns.heatmap(corr)
# sns.plt.show()


# print(stats.describe(db))
# print('------------------------------------------------------')