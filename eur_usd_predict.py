import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




data=(pd.read_csv(r"C:\Users\ishar\Desktop\trading apps\EUR_USD Historical Data (1).csv"))
data["Date_parsed"] = pd.to_datetime(data["Date"], format="%m/%d/%y")
data["Date_int"] = data["Date_parsed"].dt.strftime("%Y%m%d").astype(float)


X = data[["Date_int"]].values
y = data["Price"].values
scaler = StandardScaler()
xscaled = scaler.fit_transform(X)

xtrain,xtest,ytrain,ytest=train_test_split(xscaled,y,test_size=0.2)


model=KNeighborsRegressor()
model.fit(xtrain,ytrain)
pre=model.predict(xtest)
acc=model.score(xtest,ytest)
print("accurasy ",acc)




dates = input("enter your date (e.g., 30/01/2025): ")


new_date_parsed = pd.to_datetime(dates, format="%d/%m/%Y")
new_date_int = float(new_date_parsed.strftime("%Y%m%d"))

X_new = np.array([[new_date_int]])


X_new_scaled = scaler.transform(X_new)


pre = model.predict(X_new_scaled)
print("Predicted Price:", pre[0])
