resid_mean=round(results.resid.mean(),3)
resid_skew=round(results.resid.skew(),3)
sns.distplot(results.resid,color='navy')
plt.show()