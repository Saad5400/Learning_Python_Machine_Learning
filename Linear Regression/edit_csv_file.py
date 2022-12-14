import pandas as pd

data = pd.read_csv("CarPrice_Assignment.csv", index_col="car_ID")

print(data)

convert_dict = {
    "gas": 1,
    "diesel": -1,
    "std": 1,
    "turbo": -1,
    "two": 2,
    "four": 4,
    "rwd": -1,
    "fwd": 1,
    "4wd": 4
}

for i in range(1, 205):
    for j in ["fueltype", "aspiration", "doornumber", "drivewheel"]:
        data.loc[i, j] = convert_dict[data.loc[i, j]]

print(data)
data.to_csv("edited_cars.csv")