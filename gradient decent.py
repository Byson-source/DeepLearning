import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
from sympy import symbols,diff
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
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
x_5=np.array([0.1,1.2,2.4,3.2,4.1,5.7,6.5])
y_5=np.array([1.7,2.4,3.5,3.0,6.1,9.4,8.2])
# print(y_5[0])
# regr=LinearRegression()
# regr.fit(x_5,y_5)
# y_hat=0.847535148603+1.22272646378*x_5
def MSE(x,y):
    mse_calc=1/7*sum((x-y)**2)
    return mse_calc
# # print(MSE())
# th_0=np.linspace(start=-1,stop=3,num=5)
# th_1=np.linspace(start=-1,stop=3,num=5)
# plot_t0,plot_t1=np.meshgrid(th_0,th_1)
# # print(plot_t1)
# plot_cost=np.zeros((5,5))
# print(regr.intercept_)
# print(regr.coef_)
# plt.scatter(x_5,y_5,s=50)
# plt.plot(x_5,regr.coef_[0][0]*x_5+regr.intercept_[0])
# plt.show()
def grad(x,y,thetas):
    n=y.size 
    theta0_slope=(-2/n)*sum(y-thetas[0]-thetas[1]*x)
    theta1_slope=(-2/n)*sum((y-thetas[0]-thetas[1]*x)*x)
    return np.array([theta0_slope,theta1_slope])
multiplier=0.01
thetas=np.array([2.9,2.9])
A=np.array([1,2,3,4,5])
B=np.array([2,3,4,7,8])
plot_vals=thetas.reshape(1,2)
mse_vals=MSE(y_5,thetas[0]+thetas[1]*x_5)
for i in range(1000):
    thetas=thetas-multiplier*grad(x_5,y_5,thetas)
    plot_vals=np.append(arr=plot_vals,values=thetas.reshape(1,2),axis=0)
    mse_vals=np.append(arr=mse_vals,values=MSE(y_5,thetas[0]+thetas[1]*x_5))

fig=plt.figure(figsize=[8,6])
ax=fig.gca(projection='3d')
ax.set_xlabel('x',fontsize=20)
ax.set_ylabel('y',fontsize=20)
ax.set_zlabel('f(x,y)',fontsize=20)
ax.scatter(plot_vals[:,0],plot_vals[:,1],mse_vals,cmap=cm.coolwarm,alpha=0.4)
# ax.scatter(params[:,0],params[:,1],f(params[:,0],params[:,1]),
# s=50,color='red')
plt.show()

print(thetas[0])
print(thetas[1])

