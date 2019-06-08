import pandas as pd
import sklearn
from sklearn.cluster import KMeans

from sklearn import datasets
boston = datasets.load_boston()
bs = pd.DataFrame(boston.data)
bs.columns = boston.feature_names
bs['Target'] = pd.Series(boston.target,index = bs.index)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(bs[['AGE', 'DIS']], bs['Target'], test_size = 0.35)
for i in range(0, 4):
    kme = KMeans(n_clusters=(3+i*2), random_state=0).fit(x_train)
    kme.predict(x_test)
    s = kme.score(x_test)
    print(s)


