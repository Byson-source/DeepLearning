FILE='./dataset/SpamData/01_Processing/practice_email.txt'
from os import walk
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import nltk
from nltk.stem import PorterStemmer,SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

SPAM_1_PATH='./dataset/SpamData/01_Processing/spam_assassin_corpus/spam_1'
SPAM_2_PATH='./dataset/SpamData/01_Processing/spam_assassin_corpus/spam_2'
EASY_NONSPAM_1_PATH='./dataset/SpamData/01_Processing/spam_assassin_corpus/easy_ham_1'
EASY_NONSPAM_2_PATH='./dataset/SpamData/01_Processing/spam_assassin_corpus/easy_ham_2'
DATA_JSON_FILE='./dataset/email_text_data.json'
WORD_ID_FILE='./dataset/word_by_id.csv'
SPAM_CAT=1
HAM_CAT=0
VOCAB_SIZE=2500

#情報の取得
# stream=open(FILE,encoding='latin-1')
# message=stream.read()
# stream.close()
# print(message)

#メール本文のみ取得
# stream=open(FILE,encoding='latin-1')
# is_body=False
# lines=[]
# for line in stream:
#     if is_body:
#         lines.append(line)
#     elif line=='\n':
#         is_body=True
# stream.close()
# email_body='\n'.join(lines)
# print(email_body)
# print(lines)

# def generate_squares(N):
#     for my_number in range(N):
#         yield my_number**2
#yieldは前のループの状態を覚えている

# for root,dirnames,filenames in walk('./dataset'):
    # print(root)
    # print(dirnames)
    # print(filenames)



# test=join('1','23')
# print(test)

def email_body_generator(path):
    for root,dirnames,filenames in walk(path):
        for file_name in filenames:
            filepath=join(root,file_name)
            stream=open(filepath,encoding='latin-1')
            is_body=False
            lines=[]
            for line in stream:
                if is_body:
                    lines.append(line)
                elif line=='\n':
                    is_body=True
            stream.close()
            email_body='\n'.join(lines)
            yield file_name,email_body

def df_from_directory(path,classification):
    rows=[]
    row_names=[]
    for file_name,email_body in email_body_generator(path):
        rows.append({'MESSAGE':email_body,'CATEGORY':classification})
        row_names.append(file_name)
    return pd.DataFrame(rows,index=row_names)

#HTMLタグを消した
def clean_message(message,stemmer=PorterStemmer(),
                    stop_words=set(stopwords.words('english'))):
    soup=BeautifulSoup(message,'html.parser')
    cleaned_text=soup.get_text()
    words=word_tokenize(cleaned_text.lower())
    filtered_words=[stemmer.stem(word) for word in words if word not in stop_words and word.isalpha()]
    
    return filtered_words
# print(clean_message(data.at[2,'MESSAGE']))

spam_emails=df_from_directory(SPAM_1_PATH,SPAM_CAT)
spam_emails=spam_emails.append(df_from_directory(SPAM_2_PATH,SPAM_CAT))
non_spam_emails=df_from_directory(EASY_NONSPAM_1_PATH,HAM_CAT)
non_spam_emails=non_spam_emails.append(df_from_directory(EASY_NONSPAM_2_PATH,HAM_CAT))
data=pd.concat([spam_emails,non_spam_emails])
# print(non_spam_emails.shape)

# print(data.shape)
# print(data.head())

#Clean the data
# print(data['MESSAGE'].isnull().values.any())

#Check if there are empty emails
# print((data.MESSAGE.str.len()==0).sum())

#The name of empty index
# print(data[data.MESSAGE.str.len()==0].index)

# print(data.index.get_loc('cmds'))

data=data.drop(['cmds'])
# print((data.MESSAGE.str.len()==0).any())

document_ids=range(0,len(data.index))
# print(type(document_ids))
data['DOC_ID']=document_ids
data['FILE_NAME']=data.index
data.set_index('DOC_ID',inplace=True)
# print(data.head())

#Save as a json file
# data.to_json(DATA_JSON_FILE)

#Pie Charts
# category_names=['Spam','Legit Mail']
# sizes=[data.CATEGORY.value_counts()[0],data.CATEGORY.value_counts()[1]]
# sns.set()
# plt.pie(sizes,labels=category_names,autopct='%1.1f%%')
# plt.show()

#Natural Language Processing
#Download the NLTK Resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('gutenberg')
# nltk.download('shakespeare')
# print(word_tokenize(msg.lower()))
stop_words=set(stopwords.words('english'))

# msg='All work and no play makes Jack a dull boy. To be or not to be.'
# stemmer=PorterStemmer()
# stemmer=SnowballStemmer('english')
# msg.lower()
# words=word_tokenize(msg.lower())
# filtered_words=[stemmer.stem(singleword) for singleword in words if singleword not in stop_words and singleword.isalpha()==True]
# soup=BeautifulSoup(data.at[2,'MESSAGE'],'html.parser')
# # print(soup.prettify())
# #HTMLタグを消す
# print(soup.get_text())


#1行目から三行目のメッセージを抜き出す．
# first_emails=data.MESSAGE.iloc[0:3]
#cleaned_messageを使用する
nested_list=data.MESSAGE.apply(clean_message)
# print(nested_list.head())
# print(nested_list)
#全ての単語を一つにまとめる
# flat_list=[word for words in nested_list for word in words]

#nest_listをスパムと非スパムにわける
# spam_id=data[data.CATEGORY==1].index
# ham_id=data[data.CATEGORY==0].index
# nested_list_ham=nested_list.loc[ham_id]
# nested_list_spam=nested_list.loc[spam_id]
# spam_number=[len(spam_collection_list) for spam_collection_list in nested_list_spam]
# ham_number=[len(ham_collection_list) for ham_collection_list in nested_list_ham]
# print(sum(ham_number))
# print(sum(spam_number))
# spam_words_list,ham_words_list=[],[]
# for spam_collection_list in nested_list_spam:
#     spam_words_list+=spam_collection_list
# for ham_collection_list in nested_list_ham:
#     ham_words_list+=ham_collection_list
# spam_words_collection=pd.Series(spam_words_list)
# word_counts=spam_words_collection.value_counts()
# print(word_counts[0:10])

# word_cloud=WordCloud().generate(nested_list_spam)
# plt.imshow(word_cloud)
# plt.show()

# example_corpus=nltk.corpus.gutenberg.words('melville-moby_dick.txt')
# print(len(example_corpus))

# word_list=[' '.join(word) for word in example_corpus]
# print(word_list)

# word_cloud=WordCloud().generate(word_list)
# plt.imshow(word_cloud)
# plt.show()

#メッセージを洗ってlist形式にする
stemmed_nested_list=data.MESSAGE.apply(clean_message)

length_of_message=[len(story) for story in stemmed_nested_list]
# print(stemmed_nested_list.iloc[length_of_message.index(max(length_of_message))])
# data['stemmed_nested_list']=data.MESSAGE.apply(clean_message)
# data['Length_of_the_Message']=len(data.stemmed_nested_list)
# print('Max_ID:',data[data.Length_of_the_Message==data.Length_of_the_Message.max()].index)

#全てのメッセージを足し合わせて単語だけのlistにする
flat_stemmed_list=[item for sublist in stemmed_nested_list for item in sublist]
#wordの種類ごとに分ける
unique_words=pd.Series(flat_stemmed_list).value_counts()
# print(unique_words.head())
frequent_words=unique_words[0:VOCAB_SIZE]
print(frequent_words.head(10))

# #Create Vocabulary Dataframe with a word_id
word_ids=list(range(0,VOCAB_SIZE))
vocab=pd.DataFrame({'VOCAB_WORD':frequent_words.index.values},index=word_ids)
vocab.index.name='WORD_ID'

# #Save the vocabulary as a csv file
# vocab.to_csv(WORD_ID_FILE,index_label=vocab.index.name,header='VOCAB_WORD')

#Generate Features and a SparseMatrix
word_column_df=pd.DataFrame.from_records(stemmed_nested_list.tolist())
x_train,x_test,y_train,y_test=train_test_split(word_column_df,data.CATEGORY,
test_size=0.3,random_state=42)

#Create a sparse matrix for the training data
word_index=pd.Index(vocab.VOCAB_WORD)
#'thu'のIndex取得
word_index.get_loc('thu')

def make_sparse_matrix(df,indexed_words,labels):
    """
    spamメールで使われる単語をデータフレームとして出力する
    """
    nr_rows=df.shape[0]
    nr_cols=df.shape[1]
    word_set=set(indexed_words)
    dict_list=[]
    for i in range(nr_rows):
        for j in range(nr_cols):
            word=df.iat[i,j]
            if word in word_set:
                doc_id=df.index[i]
                word_id=indexed_words.get_loc(word)
                category=labels[doc_id]
                item={'LABEL':category,'DOC_ID':doc_id,
                'OCCURENCE':1,'WORD_ID':word_id}
                dict_list.append(item)
    return pd.DataFrame(dict_list)

sparse_train_df=make_sparse_matrix(x_train,word_index,y_train)

#Combine occurences with the pandas grupby method
