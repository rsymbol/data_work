import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def __save_obj(name, obj):
    with open('data/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def __load_obj(name):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_db():
    return __load_obj('data')


def get_dict(name):
    return __load_obj('dict').get(name)


def get_feature_names():
    return get_db().columns.values[1:-1]


def get_target_names():
    return get_dict('target_name')


def split_date(test_size=0.33, seed=7, visible=True):
    db = get_db().values
    X = db[:, 1:-1]
    y = db[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    if visible:
        print("-----------------------------")
        print("test: {}%, seed: {}".format(round(test_size * 100, 2), seed))
        print("X_train.shape: {}\ty_train.shape: {}".format(X_train.shape, y_train.shape))
        print("X_test.shape: {}\ty_test.shape: {}".format(X_test.shape, y_test.shape))
        print("-----------------------------")
    return X_train, X_test, y_train, y_test


def evaluate(y_test, pred):
    print("-----------------------------")
    accuracy = accuracy_score(y_test, pred)
    print("accuracy: %.2f%%" % (accuracy * 100.0))

    print("-----------------------------")
    # print(np.unique(y_test))
    # print(np.unique(predictions))
    print("confusion_matrix:")
    print(confusion_matrix(y_test, pred))

    print("-----------------------------")
    fpr, tpr, thresholds = roc_curve(y_test, pred, pos_label=2)
    auc_ = auc(fpr, tpr)
    print("fpr: {}, tpr: {}, thresholds: {}".format(fpr, tpr, thresholds))
    print("auc: %.2f%%" % (auc_ * 100.0))

    print("-----------------------------")
    precision_macro = precision_score(y_test, pred, average='macro')
    print("precision_macro: %.2f%%" % (precision_macro * 100.0))
    precision_micro = precision_score(y_test, pred, average='micro')
    print("precision_micro: %.2f%%" % (precision_micro * 100.0))
    precision_weighted = precision_score(y_test, pred, average='weighted')
    print("precision_weighted: %.2f%%" % (precision_weighted * 100.0))

    print("-----------------------------")
    recall_macro = recall_score(y_test, pred, average='macro')
    print("recall_macro: %.2f%%" % (recall_macro * 100.0))
    recall_micro = recall_score(y_test, pred, average='micro')
    print("recall_micro: %.2f%%" % (recall_micro * 100.0))
    recall_weighted = recall_score(y_test, pred, average='weighted')
    print("recall_weighted: %.2f%%" % (recall_weighted * 100.0))

    print("-----------------------------")
    f1_score_macro = f1_score(y_test, pred, average='macro')
    print("f1_score_macro: %.2f%%" % (f1_score_macro * 100.0))
    f1_score_micro = f1_score(y_test, pred, average='micro')
    print("f1_score_micro: %.2f%%" % (f1_score_micro * 100.0))
    f1_score_weighted = f1_score(y_test, pred, average='weighted')
    print("f1_score_weighted: %.2f%%" % (f1_score_weighted * 100.0))

    print("-----------------------------")
    print()


class Estimator:

    def __init__(self, name, estimator, av_param):
        self.__name = name
        self.__estimator = estimator
        self.__av_param = av_param

    def getInfo(self):
        return 'Name: {}\n{}\nParameters: {}'.format(self.__name, self.__estimator, self.__av_param)

    def getName(self):
        return self.__name

    def getEst(self):
        return self.__estimator

    def getPar(self):
        return self.__av_param
