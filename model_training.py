from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def train_logistic_regression(x_train, y_train):
    classifier = LogisticRegression(random_state=0)
    classifier.fit(x_train, y_train)
    return classifier

def train_knn(x_train, y_train, n_neighbors=3):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train, y_train)
    return knn

def train_svm(x_train, y_train, kernel='rbf'):
    svm_clf = svm.SVC(kernel=kernel)
    svm_clf.fit(x_train, y_train)
    return svm_clf

def train_decision_tree(x_train, y_train):
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    return clf

def train_random_forest(x_train, y_train):
    rfc = RandomForestClassifier(bootstrap=True, max_depth=7, max_features=15, min_samples_leaf=3,
                                 min_samples_split=10, n_estimators=200, random_state=7)
    rfc.fit(x_train, y_train)
    return rfc
