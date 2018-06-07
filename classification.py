# First XGBoost model for Pima Indians dataset
import numpy as np

from sklearn import datasets
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

# load data
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
feature_names = iris.feature_names

print("-----------------------------")
print("X.shape: " + str(X.shape))
print("y.shape: " + str(y.shape))
print("target_names: " + str(target_names))
print("feature_names: " + str(feature_names))

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

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
print("-----------------------------")
accuracy = accuracy_score(y_test, predictions)
print("accuracy: %.2f%%" % (accuracy * 100.0))

print("-----------------------------")
# print(np.unique(y_test))
# print(np.unique(predictions))
print("confusion_matrix:")
print(confusion_matrix(y_test, predictions))

print("-----------------------------")
fpr, tpr, thresholds = roc_curve(y_test, predictions, pos_label=2)
auc = auc(fpr, tpr)
print(fpr, tpr, thresholds)
print("auc: %.2f%%" % (auc * 100.0))

print("-----------------------------")
precision_macro = precision_score(y_test, predictions, average='macro')
print("precision_macro: %.2f%%" % (precision_macro * 100.0))
precision_micro = precision_score(y_test, predictions, average='micro')
print("precision_micro: %.2f%%" % (precision_micro * 100.0))
precision_weighted = precision_score(y_test, predictions, average='weighted')
print("precision_weighted: %.2f%%" % (precision_weighted * 100.0))

print("-----------------------------")
recall_macro = recall_score(y_test, predictions, average='macro')
print("recall_macro: %.2f%%" % (recall_macro * 100.0))
recall_micro = recall_score(y_test, predictions, average='micro')
print("recall_micro: %.2f%%" % (recall_micro * 100.0))
recall_weighted = recall_score(y_test, predictions, average='weighted')
print("recall_weighted: %.2f%%" % (recall_weighted * 100.0))

print("-----------------------------")
f1_score_macro = f1_score(y_test, predictions, average='macro')
print("f1_score_macro: %.2f%%" % (f1_score_macro * 100.0))
f1_score_micro = f1_score(y_test, predictions, average='micro')
print("f1_score_micro: %.2f%%" % (f1_score_micro * 100.0))
f1_score_weighted = f1_score(y_test, predictions, average='weighted')
print("f1_score_weighted: %.2f%%" % (f1_score_weighted * 100.0))
