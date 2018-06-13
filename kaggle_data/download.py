import kaggle as kg
import pandas as pd

# home_path = 'c:\\Users\СоколовКВ\\.kaggle\\competitions\\'

#kaggle competitions files -c titanic
#kaggle competitions download  -c titanic -p C:/Users/СоколовКВ/PycharmProjects/data_work/kaggle_data/titanic

ex_kg = kg.KaggleApi()
res = ex_kg.competition_download_files(competition='titanic', path='/test')
print(res)
# base = 'titanic'
# db = pd.read_csv('data/feature_names.csv')
# print(db.head())