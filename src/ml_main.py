# non-deep machine learning methods
import time

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from .data import get_dataset


def run_svm(train_X, train_y, test_X, test_y):
    # SVM
    param_svm = {
        'gamma': 0.01,
        'C': 100,
        'kernel': 'rbf',
    }
    print("Training SVM classifier...")
    svm = SVC(**param_svm)
    svm.fit(train_X, train_y)
    res = svm.score(test_X, test_y)
    print("SVM:", res)


def run_knn(train_X, train_y, test_X, test_y):
    # kNN
    param_knn = {
        'n_neighbors': 1,
        'metric':  'cosine',
    }
    print("Training k Nearest neighbour classifier...")
    print("parameters:", param_knn)
    knn = KNeighborsClassifier(**param_knn)
    knn.fit(train_X, train_y)
    res = knn.score(test_X, test_y)
    print("kNN:", res)


def run_rf(train_X, train_y, test_X, test_y):
    # random forest
    param_rf = {
        'n_estimators': 100,
        'max_depth':  10,
        'max_features': 'sqrt',
    }
    print("Training Random Forest classifier...")
    print("parameters:", param_rf)
    rfc = RandomForestClassifier(**param_rf)
    rfc.fit(train_X, train_y)
    res = rfc.score(test_X, test_y)
    print("Random Forest:", res)


def run_nb(train_X, train_y, test_X, test_y):
    # naive bayes
    print("Training Gaussian Naive Bayes classifier...")
    gnb = GaussianNB()       # 高斯先验
    gnb.fit(train_X, train_y)
    res = gnb.score(test_X, test_y)
    print("Naive Bayes:", res)


def run_ml(dataset):

    train_set, test_set = get_dataset(
        name=dataset, net_type="mlp", train=True, test=True)
    train_X, train_y = train_set.get_numpy_data()
    test_X, test_y = test_set.get_numpy_data()

    # runs = [run_svm, run_knn, run_rf, run_nb]
    runs = [run_rf, ]

    for run in runs:
        time_start = time.time()
        run(train_X, train_y, test_X, test_y)
        time_end = time.time()
        print('Time cost:', time_end-time_start, '\n')
