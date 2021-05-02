import pandas as pd
import numpy as np

TRAINING_DATA_FILE='./dataset/train-data.csv'
TEST_DATA_FILE='./dataset/test-data.csv'
VOCAB_SIZE=2500
TOKEN_SPAM_PROB_FILE='./dataset/prob-spam.csv'
TOKEN_NONSPAM_PROB_FILE='./dataset/prob-nonspam.csv'
TOKEN_PROB_ALL_FILE='./dataset/token-all-prob.csv'
TEST_FEATURE_MATRIX='./dataset/test-features.csv'
TEST_TARGET_FILE='./dataset/test-target.csv'

training_data=pd.read_csv(TRAINING_DATA_FILE)
test_data=pd.read_csv(TEST_DATA_FILE)

def make_full_matrix(sparse_matrix,doc_idx=1,word_idx=2,cat_idx=3,freq_idx=4):
    column_names=['CATEGORY']+list(range(0,VOCAB_SIZE)) 
    doc_id_names=sparse_matrix.DOC_ID.value_counts().index
    full_matrix=pd.DataFrame(index=doc_id_names,columns=column_names)
    full_matrix.fillna(value=0,inplace=True)
    label_status=[sparse_matrix[sparse_matrix.DOC_ID==i].LABEL.iloc[0] for i in doc_id_names]
    full_matrix.CATEGORY=label_status

    for i in range(sparse_matrix.shape[0]):
        doc_nr=sparse_matrix.iloc[i,doc_idx]
        word_id=sparse_matrix.iloc[i,word_idx]
        full_matrix.loc[doc_nr,word_id]+=1
    return full_matrix
# full_train_data=make_full_matrix(training_data)

# #条件付確率
# prob_spam=len(full_train_data[full_train_data.CATEGORY==1].index)/full_train_data.shape[0]
# full_train_features=full_train_data.loc[:,full_train_data.columns!='CATEGORY']
# email_length=full_train_features.sum(axis=1)
# # print(email_length.head())
# total_wc=email_length.sum()

# #spamのメッセージを含むメールの単語の長さ
# only_spam_data=full_train_data.loc[full_train_data.CATEGORY==1]
# spam_features=only_spam_data.loc[:,only_spam_data.columns!='CATEGORY']
# spam_length=spam_features.sum(axis=1)
# Total_spam_length=spam_length.sum()

# #Nonspamのメッセージを含むメールの単語の長さ
# only_nonspam_data=full_train_data.loc[full_train_data.CATEGORY==0]
# nonspam_features=only_nonspam_data.loc[:,only_nonspam_data.columns!='CATEGORY']
# nonspam_length=nonspam_features.sum(axis=1)
# Total_nonspam_length=nonspam_length.sum()

# #Spamメールだけ抜き出してそれぞれ単語が何個出現したか
# #☟有用
# train_spam_tokens=full_train_features.loc[full_train_data.CATEGORY==1]
# summed_spam_tokens=train_spam_tokens.sum(axis=0)+1 

# #Nonspamの場合
# train_nonspam_tokens=full_train_features.loc[full_train_data.CATEGORY==0]
# summed_nonspam_tokens=train_nonspam_tokens.sum(axis=0)+1

# #条件付確率の計算
# #P(Token|Spam)
# prob_tokens_spam=summed_spam_tokens/(Total_spam_length+VOCAB_SIZE)
# # prob_tokens_spam.to_csv(TOKEN_SPAM_PROB_FILE)
# # print(prob_tokens_spam.sum())

# #P(Token|Ham)
# prob_tokens_nonspam=summed_nonspam_tokens/(Total_nonspam_length+VOCAB_SIZE)
# # prob_tokens_nonspam.to_csv(TOKEN_NONSPAM_PROB_FILE)

# #P(Token)
# prob_tokens_all=full_train_features.sum(axis=0)/full_train_features.sum(axis=0).sum()
# # prob_tokens_all.to_csv(TOKEN_PROB_ALL_FILE)

#Prepare test data
full_test_data=make_full_matrix(test_data)
test_feature_data=full_test_data.loc[:,full_test_data.columns!='CATEGORY']
test_feature_data.to_csv(TEST_TARGET_FILE)
print(full_test_data.head())
test_target_data=full_test_data.CATEGORY
test_target_data.to_csv(TEST_FEATURE_MATRIX)