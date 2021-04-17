import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
from sympy import symbols,diff
from sklearn.linear_model import LinearRegression
def f(x,y):
    r=3**(-x**2-y**2)
    return 1/(r+1)

x_4=np.linspace(start=-2,stop=2,num=200)
y_4=np.linspace(start=-2,stop=2,num=200)
x_4,y_4=np.meshgrid(x_4,y_4)
a,b=symbols('x,y')
# print(diff(f(a,b),a))
# print(diff(f(a,b),a).evalf(subs={a:2.1,b:5.0}))
# multiplier=0.1
# max_iter=200
# params=np.array([[1.8,1.0]])
# for n in range(max_iter):
#     gradient_x=diff(f(a,b),a).evalf(subs={a:params[n][0],b:params[n][1]})
#     gradient_y=diff(f(a,b),b).evalf(subs={a:params[n][0],b:params[n][1]})
#     gradients=np.array([gradient_x,gradient_y])
#     new_params=params[-1]-multiplier*gradients
#     new_params=new_params.reshape(1,2)
#     params=np.append(arr=params,values=new_params,axis=0)

#3dグラフの作製
# fig=plt.figure(figsize=[8,6])
# ax=fig.gca(projection='3d')
# ax.set_xlabel('x',fontsize=20)
# ax.set_ylabel('y',fontsize=20)
# ax.set_zlabel('f(x,y)',fontsize=20)
# ax.plot_surface(x_4,y_4,f(x_4,y_4),cmap=cm.coolwarm,alpha=0.4)
# ax.scatter(params[:,0],params[:,1],f(params[:,0],params[:,1]),
# s=50,color='red')
# plt.show()
x_5=np.arange(7,20)
y_5=np.arange(10,23)
x_5.reshape(1,2)
y_5.reshape(1,2)
regr=LinearRegression()
regr.fit(x_5,y_5)
# y_5=np.arrange(7)
print(x_5)