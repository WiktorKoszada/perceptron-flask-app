import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from perceptron import Perceptron

iris = load_iris()
X = iris.data[:, [0, 2]]  
y = iris.target

y = np.where(y == 0, 1, -1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(ppn, f)