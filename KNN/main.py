import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")

PREDICT = "class"

le = preprocessing.LabelEncoder()
labels_dict = dict()
for label in ["buying", "maint", "door", "persons", "lug_boot", "safety", PREDICT]:
    labels_dict[label] = le.fit_transform(list(data[label]))

x = list(zip(labels_dict["buying"], labels_dict["maint"], labels_dict["door"], labels_dict["persons"],
             labels_dict["lug_boot"], labels_dict["safety"]))
y = list(labels_dict[PREDICT])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=8)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)

names = ["unacc", "acc", "good", "vgood"]
for x in range(len(predicted)):
    if predicted[x] != y_test[x]:
        print(f"Predicted: {predicted[x]} Actual: {y_test[x]}")