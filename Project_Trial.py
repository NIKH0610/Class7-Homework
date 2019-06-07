import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn
import seaborn as sns

from sklearn import datasets
boston = datasets.load_boston()
bs = pd.DataFrame(data=mp.c_[boston.data, boston.target], columns = boston.feature_names + ['target'])
mean_CRIM = bs["CRIM"].mean()

#Anything more than Mean in CRIM is considered to be high CRIME rate

print(mean_CRIM)

#Removing Rows which is less than mean CRIM index

CRIM_high = bs.drop(bs[bs.CRIM<mean_CRIM].index)
#print(CRIM_high)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(CRIM_high, CRIM_high["MEDV"], size = 0.20)

from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(x_train, y_train)
linear.predict(x_test)

