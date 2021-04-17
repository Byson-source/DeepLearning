import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data=pd.read_csv('./dataset/cost_revenue_clean.csv')
x=pd.DataFrame(data,columns=['production_budget_usd'])
y=pd.DataFrame(data,columns=['worldwide_gross_usd'])

regression=LinearRegression()
regression.fit(x,y)
#傾き
print(regression.coef_)
#y切片
print(regression.intercept_)
plt.plot(x,regression.predict(x),color='red',linewidth=4)
plt.scatter(x,y,alpha=0.3)
print(regression.score(x,y))
plt.show()