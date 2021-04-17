import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()
x=iris.data[:,(2,3)] 
y=(iris.target==0)
y=y.astype(np.int)
from sklearn.linear_model import Perceptron
per_clf=Perceptron(random_state=42)
per_clf.fit(x,y)
y_pred=per_clf.predict(x)
# print(y_pred)
from sklearn.metrics import accuracy_score
print(accuracy_score(y,y_pred))