import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as pyplot
from os import listdir
from os.path import isfile, join
from matplotlib import style

cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

x = cancer.data
y = cancer.target

total_size = len(y)
train_size = int(total_size * 0.7)
test_size = total_size - train_size

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
# x_train, x_test, y_train, y_test = x[:train_size], x[:test_size], y[:train_size], y[:test_size]


# print(x_train, y_train)
classes = ['malignant', 'benign']

clf = svm.SVC(kernel="linear",  C=0.85)
# clf = KNeighborsClassifier()

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)

print(acc)