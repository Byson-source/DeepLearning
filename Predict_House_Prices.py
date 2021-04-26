# from sklearn.datasets import load_boston
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# import numpy as np
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# import statsmodels.api as sm

# boston_dataset=load_boston()
# data=pd.DataFrame(data=boston_dataset.data,columns=boston_dataset.feature_names)
# data['PRICE']=boston_dataset.target
# sns.set()
# # plt.figure(figsize=(5,4))
# # plt.hist(data['PRICE'],bins=50,ec='black')
# # print(data['RAD'].value_counts())
# # sns.jointplot(x=data['PRICE'],y=data['RM'],color='orange',alpha=0.4)
# # sns.lmplot(x='PRICE',y='RM',data=data)
# # plt.xlabel('Price in 000s')
# # plt.ylabel('Nr. of Houses')
# # sns.pairplot(data,kind='reg',plot_kws={'line_kws':{'color':'cyan'}})
# # plt.show()
# # print(data.describe())

# #分布が対数っぽいのでlogる
# prices=np.log(data['PRICE'])

# # features=data.drop('PRICE',axis=1)
# # x_train,x_test,y_train,y_test=train_test_split(features,prices,
# # test_size=0.2,random_state=10)
# # regr=LinearRegression()
# # regr.fit(x_train,y_train)
# # # print('Intercept:',regr.intercept_)
# # df=pd.DataFrame(data=regr.coef_,index=x_train.columns,columns=['coef'])

# # print(regr.score(x_train,y_train))
# # print(regr.score(x_test,y_test))

# # print(data)
# # sns.distplot(data)
# # plt.show()

# ###########################################################################################
# #ここから評価手法
# # x_incl_const=sm.add_constant(x_train)#y切片が存在する時必ず必要

# # model=sm.OLS(y_train,x_incl_const)
# # results=model.fit()

# # print(results)

# #p値を求めるp値が0.005よりも小さい場合そのデータは信頼性がある．　
# # df=pd.DataFrame({'params':results.params,'p-value':round(results.pvalues,3)})

# # print(variance_inflation_factor(exog=x_incl_const.values,exog_idx=1)) #VIFを求めるための式 VIFとは10よりも小さかったらそのデータに信頼性があり，他のデータと同列に扱ってもよいということを指す
# # vif_list=[variance_inflation_factor(exog=x_incl_const.values,exog_idx=index) for index in range(len(x_incl_const.columns))]
# # vif_df=pd.DataFrame({'column':x_incl_const.columns,'VIF':vif_list})
# # print(vif_df)

# #BICについて．BICとはそのモデルがどれほど複雑であるかを数字で表す．当然数字が小さいほうがいい．
# #Original model with log prices and all features

# #'INDUS'を含んだ場合
# # x_incl_const=sm.add_constant(x_train)
# # model=sm.OLS(y_train,x_incl_const)
# # results=model.fit()
# # org_coef=pd.DataFrame({'coef':results.params,'p-value':round(results.pvalues,3)})
# # print(results.bic)
# # print(results.rsquared)

# # r-squared...https://ja.wikipedia.org/wiki/%E6%B1%BA%E5%AE%9A%E4%BF%82%E6%95%B0#:~:text=%E6%B1%BA%E5%AE%9A%E4%BF%82%E6%95%B0%20%EF%BC%88%20%E3%81%91%E3%81%A3%E3%81%A6%E3%81%84%E3%81%91%E3%81%84%E3%81%99%E3%81%86,%E5%B0%BA%E5%BA%A6%E3%81%A8%E3%81%97%E3%81%A6%E5%88%A9%E7%94%A8%E3%81%95%E3%82%8C%E3%82%8B%E3%80%82
# #要は高ければ高いほど良い

# #INDUSを除いた場合
# # x_incl_const=sm.add_constant(x_train)
# # x_incl_const=x_incl_const.drop(['INDUS'],axis=1)
# # model=sm.OLS(y_train,x_incl_const)
# # results=model.fit()
# # org_coef=pd.DataFrame({'coef':results.params,'p-value':round(results.pvalues,3)})
# # print(results.bic)
# # print(results.rsquared)

# #さらにAGEを除いた場合
# # x_incl_const=sm.add_constant(x_train)
# # x_incl_const=x_incl_const.drop(['INDUS','AGE'],axis=1)
# # model=sm.OLS(y_train,x_incl_const)
# # results=model.fit()
# # org_coef=pd.DataFrame({'coef':results.params,'p-value':round(results.pvalues,3)})
# # print(results.bic)
# # print(results.rsquared)

# #これら二つを除いた場合，よりモデルが簡潔になる．

# #Residuals
# #より均等にプロットされていると，上手く学習ができている．
# prices=np.log(data['PRICE'])
# # prices=data['PRICE']
# features=data.drop(['PRICE','INDUS','AGE'],axis=1)
# x_train,x_test,y_train,y_test=train_test_split(features,prices,
# test_size=0.2,random_state=10)
# x_incl_const=sm.add_constant(x_train)
# model=sm.OLS(y_train,x_incl_const)
# results=model.fit()

# # residuals=y_train-results.fittedvalues
# # print(residuals)

# # print(results.resid)

# #Graph of Actual vs Predicted Prices
# # corr=round(y_train.corr(results.fittedvalues),2)
# # plt.scatter(x=y_train,y=results.fittedvalues)
# # plt.xlabel('Teacher Data')
# # plt.ylabel('Predicted Data')
# # plt.show()

# # #logを解く
# # corr=round(y_train.corr(results.fittedvalues),2)
# # plt.scatter(x=np.e**y_train,y=np.e**results.fittedvalues)
# # plt.xlabel('Teacher Data')
# # plt.ylabel('Predicted Data')
# # plt.show()

# # #residualと予測値を比べる
# # corr=round(y_train.corr(results.fittedvalues),2)
# # plt.scatter(x=np.e**(y_train-results.fittedvalues),y=np.e**results.fittedvalues)
# # plt.xlabel('Teacher Data')
# # plt.ylabel('Predicted Data')
# # plt.show()

# # #Residualsの歪度を調べる
# # resid_mean=round(results.resid.mean(),3)
# # resid_skew=round(results.resid.skew(),3)
# # sns.distplot(results.resid,color='navy')
# # plt.show()

# #MSEについて(log version)
# reduced_log_mse=round(results.mse_resid,3)
# reduced_log_rsquared=round(results.rsquared,3)
# print(reduced_log_rsquared)

# #logらない場合
# prices=data['PRICE']
# features=data.drop(['PRICE'],axis=1)
# x_train,x_test,y_train,y_test=train_test_split(features,prices,test_size=0.2,random_state=10)
# x_incl_const=sm.add_constant(x_train)
# model=sm.OLS(y_train,x_incl_const)
# results=model.fit()
# reduced_unlog_mse=round(results.mse_resid,3)
# reduced_unlog_rsquared=round(results.rsquared,3)

# #Third Model かなり要素を除く
# prices=np.log(data['PRICE'])
# features=data.drop(['PRICE','INDUS','AGE','LSTAT','RM','NOX','CRIM'],axis=1)
# x_train,x_test,y_train,y_test=train_test_split(features,prices,test_size=0.2,random_state=10)
# x_incl_const=sm.add_constant(x_train)
# model=sm.OLS(y_train,x_incl_const)
# results=model.fit()
# multi_reduced_log_mse=round(results.mse_resid,3)
# multi_reduced_log_rsquared=round(results.rsquared,3)
# # print(reduced_log_rsquared)
# #Rsquaredが減退
# #MSEとＲ-squared両方を調べることが大切

# #全てをデータフレームにまとめる
# evaluation_df=pd.DataFrame({'R-squared':[reduced_log_rsquared,reduced_unlog_rsquared,multi_reduced_log_rsquared],
#                             'MSE':[reduced_log_mse,reduced_unlog_mse,multi_reduced_log_mse],
#                             'MRSE':np.sqrt([reduced_log_rsquared,reduced_unlog_rsquared,multi_reduced_log_rsquared])},
#                             index=['Reduced log Model','Normal Price model','Omitted var model' ])
# print(evaluation_df)

# #もし予測値が30000ドルだったら
# upper_band=2*np.sqrt(reduced_log_mse)+np.log(30)
# lower_band=-2*np.sqrt(reduced_log_mse )+np.log(30)

#############################################################################################################3
#Valuation Tool
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

boston_dataset=load_boston()
data=pd.DataFrame(data=boston_dataset.data,columns=boston_dataset.feature_names)
features=data.drop(['INDUS','AGE'],axis=1)

# print(boston_dataset.target)
log_prices=np.log(boston_dataset.target)
target=pd.DataFrame(log_prices,columns=['PRICE'])

CRIME_IDX=0
ZN_IDX=1
CHAS_IDX=2
RM_IDX=4
PTRATO_IDX=8
property_stats=np.ndarray(shape=(1,11))
# property_stats[0][CRIME_IDX]=features['CRIM'].mean()
# property_stats[0][ZN_IDX]=features['ZN'].mean()
# property_stats[0][CHAS_IDX]=features['CHAS'].mean()
property_stats=features.mean().values.reshape(1,11)

regr=LinearRegression().fit(features,target)
fitted_vals=regr.predict(features)
# print({'MSE':mean_squared_error(target, fitted_vals)})
# print({'RMSE':np.sqrt(mean_squared_error(target, fitted_vals))})

def get_log_estimate(nr_rooms,
                    student_per_classroom,
                    next_to_river=False,
                    high_confidence=True):
    property_stats[0][RM_IDX]=nr_rooms
    property_stats[0][PTRATO_IDX]=student_per_classroom

    if next_to_river:
        property_stats[0][CHAS_IDX]=1
    else:
        property_stats[0][CHAS_IDX]=0
    elif high_confidence:
        property_stats[0][CHAS_ID] 
    log_estimate=regr.predict(property_stats)
    return log_estimate

