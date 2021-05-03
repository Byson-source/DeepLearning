import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

VOCAB_SIZE=2500

TOKEN_SPAM_PROB_FILE='./dataset/prob-spam.csv'
TOKEN_NONSPAM_PROB_FILE='./dataset/prob-nonspam.csv'
TOKEN_PROB_ALL_FILE='./dataset/token-all-prob.csv'

TEST_FEATURE_MATRIX='./dataset/test-features.csv'
TEST_TARGET_FILE='./dataset/test-target.csv'
DATA_JSON_FILE='./dataset/email_text_data.json'


#load the data
# x_test=pd.read_csv(TEST_FEATURE_MATRIX)
# x_test=x_test.iloc[:,1:]
# x_test=x_test.values
# # print(x_test.head())
# y_test=pd.read_csv(TEST_TARGET_FILE)
# y_test=y_test.iloc[:,1:]
# y_test=y_test.values
# # print(y_test.head())
# # print(y_test.head())
# token_spam_prob=pd.read_csv(TOKEN_SPAM_PROB_FILE)
# token_spam_prob=token_spam_prob.iloc[:,1]
# token_spam_prob=token_spam_prob.values
# # print(type(token_spam_prob))
# token_nonspam_prob=pd.read_csv(TOKEN_NONSPAM_PROB_FILE)
# token_nonspam_prob=token_nonspam_prob.iloc[:,1]
# token_nonspam_prob=token_nonspam_prob.values
# token_all_prob=pd.read_csv(TOKEN_PROB_ALL_FILE)
# token_all_prob=token_all_prob.iloc[:,1]
# token_all_prob=token_all_prob.values

#Calculate the joint probability
# prob_spam=np.log(token_spam_prob['0'])
# print(prob_spam)
# PROB_SPAM=0.3116
# print(token_spam_prob.head())
# joint_log_spam=x_test.dot(np.log(token_spam_prob)-np.log(token_all_prob))+np.log(PROB_SPAM)
# joint_log_nonspam=x_test.dot(np.log(token_nonspam_prob)-np.log(token_all_prob))+np.log(1-PROB_SPAM)
# print(joint_log_spam)

#Checking for the higher joint probability
#P(Spam|X)>P(Ham|X) or otherwise?
# predict=[]
# for i in range(len(joint_log_spam)):
#     if joint_log_spam[i]>joint_log_nonspam[i]:
#         predict.append(1)
#     else:
#          predict.append(0)

# print(predict)

#Metrics and accuracy
# correct_values=[1 for i in range(len(predict)) if y_test[i]==predict[i]]
# print(len(correct_values))
# print(len(correct_values)/len(predict))

#Visualizing the results
#Chart styling info
# sns.set()
# yaxis_label='P(X|Spam)'
# xaxis_label='P(X|NonSpam)'
# plt.figure(figsize=(11,7))
# plt.xlabel(xaxis_label)
# plt.ylabel(yaxis_label)
# plt.scatter(joint_log_nonspam,joint_log_spam)
# plt.show()

# sns.set()
# linedata=np.linspace(start=-14000,stop=1,num=1000)
# summary_df=pd.DataFrame({yaxis_label:joint_log_spam,xaxis_label:joint_log_nonspam,'Actual Category':y_test.reshape(1725,)})
# sns.lmplot(x=xaxis_label,y=yaxis_label,data=summary_df,hue='Actual Category',fit_reg=False)
# plt.plot(linedata,linedata,color='black')
# plt.xlim([-90,10])
# plt.ylim([-200,10])
# plt.show()

#False Positive and False Negative
# true_pos=[]
# for i in range(len(predict)):
#     if predict[i]==1:
#         if predict[i]==y_test[i]:
#             true_pos.append('True')
#         else:
#             true_pos.append('False')
#     else:
#         true_pos.append('False')
# false_pos=[]
# for i in range(len(predict)):
#     if predict[i]==1:
#         if predict[i]!=y_test[i]:
#             false_pos.append('True')
#         else:
#             false_pos.append('False')
#     else:
#         false_pos.append('False')
# false_neg=[]
# for i in range(len(predict)):
#     if predict[i]==0:
#         if predict[i]!=y_test[i]:
#             false_neg.append('True')
#         else:
#             false_neg.append('False')
#     else:
#         false_neg.append('False')

# print(true_pos.count('True'))
# print(false_pos.count('True'))
# print(false_neg.count('True'))

#precision=True_Positive/(True_Positive+False_Positive)
#Recall=True_Positive/(True_Positive+False_Negative)
# precision_score=true_pos.count('True')/(true_pos.count('True')+false_pos.count('True'))
# recall_score=true_pos.count('True')/(true_pos.count('True')+false_neg.count('True'))
# print(round(precision_score,3))

#F-score and F1-score
#F-score=2*(Precision*Recall/(Precision+Recall))
#F-scoreはPrecisionとRecallの合わせ技

##############################################################################################################
#Scikit learningを使ってみる
data=pd.read_json(DATA_JSON_FILE)
# print(data.head())
data.sort_index(inplace=True)

vectorizer=CountVectorizer(stop_words='english')
all_features=vectorizer.fit_transform(data.MESSAGE)
# print(vectorizer.vocabulary_)

x_train,x_test,y_train,y_test=train_test_split(all_features,data.CATEGORY,
                                                test_size=0.3,random_state=88)

classifier=MultinomialNB()
classifier.fit(x_train,y_train)

prediction=classifier.predict(x_test)
accuracy=(prediction==y_test).sum()
print(accuracy/len(prediction))