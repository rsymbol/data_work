import util

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = util.load_date()

# fit model no training data
model = XGBClassifier()

max_depth = [1, 2, 3, 5, 10, 15]
n_estimators = [1, 10, 50, 100, 200, 500]

# # Minimum number of samples per leaf
# min_samples_leaf = [1, 2, 4, 6, 8]
#
# # Minimum number of samples to split a node
# min_samples_split = [2, 4, 6, 10]
#
# # Maximum number of features to consider for making splits
# max_features = ['auto', 'sqrt', 'log2', None]


parameters = {'max_depth': max_depth, 'n_estimators': n_estimators}
clf = GridSearchCV(model, parameters, scoring='f1_macro')
clf.fit(X_train, y_train)
print("-----------------------------")
print("default model: " + str(model))
optimal_params = clf.best_params_
print("optimal_params: " + str(optimal_params))
model.set_params(**optimal_params)
print("optimal model: " + str(model))
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = y_pred
# predictions = [round(value) for value in y_pred]
# print(predictions[:5])

# evaluate predictions
util.evaluate(y_test, predictions)


