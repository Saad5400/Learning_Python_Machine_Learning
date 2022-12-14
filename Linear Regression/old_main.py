import os
import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from os import listdir
from os.path import isfile, join
from matplotlib import style

data = pd.read_csv("edited.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "Medu", "Fedu", "traveltime", "famrel", "goout",
             "Dalc", "Walc", "age", "health", "sex", "famsup", "schoolsup", "paid", "activities", "nursery", "higher",
             "internet", "romantic"]]

predict = "G3"

x = np.array(data.drop(labels=[predict], axis=1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

loaded_model = pickle.load(open("model_0.8417468171207294_.sav", "rb"))

# for i in range(5):
#
#
#     linear = linear_model.LinearRegression()
#
#     linear.fit(x_train, y_train)
#     # loaded_model = pickle.load(open("model_0.9451700522994311_.sav", "rb"))
#
#     acc = linear.score(x, y)
#
#     print("Acc: ", acc)
#     print("Coefficient: ", linear.coef_)
#     print("Intercept: ", linear.intercept_)
#     if acc >= 0.85:
#         pickle.dump(linear, open(f"model_{acc}_.sav", "wb"))

# dir_path = "C:\\Users\\Family\\PycharmProjects\\tensorTutorial"
# onlyfiles = [f for f in listdir() if isfile(join(dir_path, f))]

predictions = loaded_model.predict(x_test)

for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])

p = "romantic"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()