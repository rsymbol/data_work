import util

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = util.split_date()

es = [
    util.Estimator("Nearest Neighbors", KNeighborsClassifier(),
                   {'n_neighbors': [1], 'leaf_size': [1]}),
    # util.Estimator("Nearest Neighbors", KNeighborsClassifier(),
    #                {'n_neighbors': [1, 3, 5, 7, 10], 'leaf_size': [1, 5, 10, 30, 50]}),
    util.Estimator("Support Vector Classification", SVC(),
                   {'C': [0.01, 0.1, 1.0, 10.0, 100.0], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'degree': [1, 2, 3, 5]}),
    util.Estimator("Nu-Support Vector Classification", NuSVC(),
                   {'nu': [0.1, 0.3, 0.5, 0.8], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'degree': [1, 2, 3, 5]}),
    util.Estimator("Gaussian Process", GaussianProcessClassifier(),
                   None),
    util.Estimator("Decision Tree", DecisionTreeClassifier(),
                   {'criterion': ['gini', 'entropy'], 'max_depth': [1, 3, 5, 10], 'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 3, 5, 10], 'max_features': [1, 3, 'sqrt', None]}),
    util.Estimator("Random Forest", RandomForestClassifier(),
                   {'n_estimators': [1, 5, 10, 50, 100], 'criterion': ['gini', 'entropy'],
                    'max_features': [1, 3, 'sqrt', None], 'max_depth': [None, 1, 2, 3, 5, 10],
                    'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 3, 5, 10]}),
    util.Estimator("Neural Net", MLPClassifier(),
                   {'activation': ['tanh']}),
    # util.Estimator("Neural Net", MLPClassifier(),
    #                {'hidden_layer_sizes': [10, 50, 100, 200, 500],
    #                 'activation': ['identity', 'logistic', 'tanh', 'relu'],
    #                 'solver': ['lbfgs', 'sgd', 'adam'], 'alpha': [0.0001, 0.01, 1.0, 10, 100],
    #                 'learning_rate': ['constant', 'invscaling', 'adaptive']}),
    util.Estimator("AdaBoost", AdaBoostClassifier(),
                   {'algorithm': ['SAMME'],
                    'n_estimators': [10, 30, 50, 100], 'learning_rate': [0.01, 0.1, 1.0, 5.0]}),
    util.Estimator("Naive Bayes", GaussianNB(),
                   None),
    util.Estimator("Quadratic Discriminant Analysis", QuadraticDiscriminantAnalysis(),
                   {'reg_param': [0.1, 0.3, 0.7, 1.0]}),
    util.Estimator("Gradient Boosting Classifier", GradientBoostingClassifier(),
                   {'learning_rate': [0.1], 'n_estimators': [10, 100], 'max_depth': [1, 3, 5],
                    'min_samples_split': [2], 'subsample': [0.3]})
]

id_est = 0
ex_es = es[id_est]
print(ex_es.getInfo())

# fit model no training data
model = ex_es.getEst()
if ex_es.getPar() != None:
    parameters = ex_es.getPar()
    clf = GridSearchCV(model, parameters)
    clf.fit(X_train, y_train)

    print("-----------------------------")
    optimal_params = clf.best_params_
    print("optimal_params: " + str(optimal_params))
    # print(clf.cv_results_)

    model.set_params(**optimal_params)

# fit model no training data
model.fit(X_train, y_train)

# print('current loss computed with the loss function: ', model.loss_)
# print('coefs: ', model.coefs_)
# print('intercepts: ', model.intercepts_)
# print(' number of iterations the solver: ', model.n_iter_)
# print('num of layers: ', model.n_layers_)
# print('Num of o/p: ', model.n_outputs_)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = y_pred

# evaluate predictions
util.evaluate(y_test, predictions)
