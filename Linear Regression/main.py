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

data = pd.read_csv("edited_cars.csv", index_col="car_ID")
data = data[["fueltype", "aspiration", "doornumber", "drivewheel", "wheelbase", "carwidth", "carheight",
             "enginesize", "stroke", "peakrpm", "price"]]

"""important labels:
price
peakrpm
stroke
enginesize
carwidth
"""

predict = "price"

x = np.array(data.drop(labels=[predict], axis=1))
y = np.array(data[predict])

def create_model(print_data=True):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.5)

    model = linear_model.LinearRegression()

    model.fit(x_train, y_train)

    acc = model.score(x_test, y_test)

    col_list = []
    for col in data.columns:
        if str(col) == predict: continue
        col_list.append(str(col))

    if print_data:
        print("Acc: ", acc)
        print("Coefficient: \n[", end="")
        for i in range(x.shape[1]):
            print(f"{col_list[i]}: {round(model.coef_[i], 2)}, ", end="")
            if i % 4 == 0 and i != 0:
                print("")
        print("]")
        print("Intercept: ", model.intercept_)

    return model, acc

# predictions = model.predict(x_test)
#
# print("-------------------------")
#
# for i in range(len(predictions)):
#     print(predictions[i], y_test[i])

# for i in col_list:
#     p = i
#     style.use("ggplot")
#     pyplot.scatter(data[p], data[predict])
#     pyplot.xlabel(p)
#     pyplot.ylabel("Price")
#     pyplot.show()

best = 0
sum = 0
best_model = None

loop = 50
for i in range(loop):
    model, acc = create_model(print_data = False)
    acc = model.score(x, y)
    # sum += acc
    # if acc > best:
    print(acc)
    best = acc
    best_model = model

print(f"BEST: {best}")

loaded_model = pickle.load(open("carModel_0.895049687129837_.pickle", "rb"))
print(f"LOADED MODEL: {loaded_model.score(x, y)}")

# if best_model is not None:
#     pickle.dump(best_model, open(f"carModel_{best}_.pickle", "wb"))

# print(f"Average = {sum/loop}")
# AVG = 0.8