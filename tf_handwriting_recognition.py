# if __name__ == '__main__':

from numpy.random import seed
seed(888)


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from time import strftime
# import keras
print(tf.__version__)
from tensorflow import set_random_seed
set_random_seed(404)
from PIL import Image
"""#Constant"""

X_TRAIN_PATH='./MNIST/digit_xtrain.csv'
X_TEST_PATH='./MNIST/digit_xtest.csv'
Y_TRAIN_PATH='./MNIST/digit_ytrain.csv'
Y_TEST_PATH='./MNIST/digit_ytest.csv'
IMAGE_WIDTH=28
IMAGE_HEIGHT=28
CHANNELS=1
NR_CLASSES=10
VALIDATION_SIZE=10000
TOTAL_INPUTS=IMAGE_WIDTH*IMAGE_HEIGHT*CHANNELS
LOGGING_PATH='tensorboard_mnist_digit_logs/'

"""#Get the data"""

y_train=np.loadtxt(Y_TRAIN_PATH,delimiter=',',dtype=int)
y_test=np.loadtxt(Y_TEST_PATH,delimiter=',',dtype=int)
x_train=np.loadtxt(X_TRAIN_PATH, delimiter=',',dtype=int)
x_test=np.loadtxt(X_TEST_PATH, delimiter=',',dtype=int)
print(x_test.shape)

"""#Data Preprocessing"""
"""#Rescale the feature"""
x_train_all,x_test_all=x_train/255,x_test/255
y_train=np.eye(NR_CLASSES)[y_train]
y_test=np.eye(NR_CLASSES)[y_test]
"""#Create validation dataset from training data"""
x_val=x_train[:VALIDATION_SIZE]
y_val=y_train[:VALIDATION_SIZE]
x_train=x_train[VALIDATION_SIZE:]
y_train=y_train[VALIDATION_SIZE:]

"""#Setup Tensorflow Graph"""
#nameはgraphのノードに名前をつけてあげるのに使う
X=tf.placeholder(tf.float32,shape=[None,TOTAL_INPUTS],name='X')
Y=tf.placeholder(tf.float32,shape=[None,NR_CLASSES],name='Y')

"""Hyperparameters"""
nr_epochs=30
lr=1e-4
n_hidden1=512
n_hidden2=64

"""Creating Network"""
"""TOTAL_INPUTSは画像一枚分の入力"""
def setup_layer(input,weight_dim,bias_dim,name):
    with tf.name_scope(name):
        initial_weight=tf.truncated_normal(shape=weight_dim,stddev=0.1,seed=42)
        w = tf.Variable(initial_value=initial_weight, name='w')
        initial_bias = tf.constant(value=0.0, shape=bias_dim)
        b = tf.Variable(initial_value=initial_bias, name='b')

        layer_in = tf.matmul(input, w) + b
        layer_out = tf.nn.relu(layer1_in)
        if name=='out':
            output=tf.nn.softmax(layer_in)
        else:
            output=tf..nn.relu(layer_in)
        #値の変遷をグラフ化する
        tf.summary.histogram('weights',w)
        tf.summary.histogram('biases',b)
        return output


#dropout layerを入れる
layer_1=setup_layer(input=X,weight_dim=[TOTAL_INPUTS,n_hidden1],bias_dim=[n_hidden1],name='first_hidden_layer')
layer_drop=tf.nn.dropout(layer_1,keep_prob=0.8,name='dropout_layer')
    # initial_w1=tf.truncated_normal(shape=[TOTAL_INPUTS,n_hidden1],stddev=0.1,seed=42)
    # w1=tf.Variable(initial_value=initial_w1,name='w1')
    # initial_b1=tf.constant(value=0.0,shape=[n_hidden1])
    # b1=tf.Variable(initial_value=initial_b1,name='b1')
    #
    # layer1_in=tf.matmul(X,w1)+b1
    # layer1_out=tf.nn.relu(layer1_in)

"""第二層目"""

layer_2=setup_layer(input=layer_drop,weight_dim=[layer_1.shape[0],n_hidden2],bias_dim=[n_hidden2],name='second_hidden_layer')
    # initial_w2 = tf.truncated_normal(shape=[layer1_out.shape[0], n_hidden2], stddev=0.1, seed=42)
    # w2 = tf.Variable(initial_value=initial_w2,name='w2')
    # initial_b2 = tf.constant(value=0.0, shape=[n_hidden2])
    # b2 = tf.Variable(initial_value=initial_b2,name='b2')
    #
    # layer2_in=tf.matmul(layer1_out,w2)+b2
    # layer2_out=tf.nn.relu(layer_2_in)
"""Third layer"""

setup_layer(input=X, weight_dim=[layer_2.shape[0], NR_CLASSES], bias_dim=[NR_CLASSES], name='out')
    # initial_w3 = tf.truncated_normal(shape=[layer2_out.shape[0], NR_CLASSES], stddev=0.1, seed=42)
    # w3 = tf.Variable(initial_value=initial_w3)
    # initial_b3 = tf.constant(value=0.0, shape=[NR_CLASSES])
    # b3 = tf.Variable(initial_value=initial_b3)
    # layer3_in = tf.matmul(layer2_out, w3) + b3
    # layer3_out = tf.nn.softmax(layer3_in)
model_name=f'{n_hidden1}-{n_hidden2} DO= LR{learning_rate} E{nr_epochs}'
"""Tensorboard"""
#Folder for tensorboard
folder_name=f'{model_name} at {strftime("%H:%M")}'
directory=os.path.join(LOGGING_PATH,folder_name)
try:
    os.mkdir(directory)
except OSError as exception:
    print(exception.strerror)

"""Loss.Optimisation & Metrics"""
"""Define loss function"""
"""tf.reduce_meanの引数のaxisはデフォルトではNoneとなっている。Noneである場合、すべての要素の平均値を求めている。"""
with tf.name_scope('loss_calc'):
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=layer3_output))

"""Defining optimizer"""
with tf.name_scope('optimizer')
    optimizer=tf.train.AdamOptimizer(learning_rate=lr)
    train_step=optimizer.minimize(loss)

"""Accuracy metric"""
with tf.name_scope('accuracy_calc'):
    correct_pred=tf.equal(tf.argmax(output,axis=1),tf.argmax(Y,axis=1))

    """The operation casts x (in case of Tensor) or x.values (in case of SparseTensor or IndexedSlices) to dtype.dtypeとはnumpyの要素"""
    accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#ここで、tensorboardで表示する変数を定義しておく
with tf.name_scope('performance'):
    tf.summary.scalar('accuracy',accuracy)
    tf.summary.scalar('cost',loss)

"""Check input images in Tensorboard"""
with tf.name_scope('show_image'):
    x_image=tf.reshape(X,[-1,28,28,1])
    tf.summary.image('image_input',x_image,max_outputs=4)

"""https://www.atmarkit.co.jp/ait/articles/1804/20/news131.html"""
"""Run session"""
sess=tf.Session()
#Initialize all the valiables
init=tf.glorot_normal_initializer()
sess.run(init)

#Set up filewriter and merge summaries
"""https://www.atmarkit.co.jp/ait/articles/1804/27/news151.html"""
#.merge_allはtensorboardで定義した変数をまとめて格納する。この場合単一だけど。。。
merged_summary=tf.summary.merge_all()
#Writes Summary protocol buffers to event files.
train_writer=tf.summary.FileWriter(directory+'/train')
train_writer.add_graph(sess.graph)
validation_writer=tf.summary.FileWriter(directory+'/validation')
"""Batching the data"""
size_of_batch=1000
num_examples=y_train.shape[0]
nr_iterations=int(num_examples/size_of_batch)
index_in_epoch=0

def next_batch(batch_size,data,labels):
    global num_examples
    global index_in_epoch
    start=index_in_epoch
    index_in_epoch+=batch_size
    if index_in_batch>num_examples:
        start=0
        index_in_epoch=batch_size
    end=index_in_epoch
    return data[start:end],labels[start:end]
"""Training loop"""
for epoch in range(nr_epochs):
    """training dataset"""
    for i in range(nr_iterations):
        batch_x,batch_y=next_batch(size_of_batch,x_train,y_train)
        feed_dictionary={X:batch_x,Y:batch_y}
        sess.run(train_step,feed_dict=feed_dictionary)
    s,batch_accuracy=sess.run(fetches=[merged_summary,accuracy],feed_dict=feed_dictionary)
    train_writer.add_summary(s,epoch)

    print(f'Epoch {epoch} \t|Training Accuracy={batch_accuracy}')
    """Validation"""
    summary=sess.run(fetches=merged_summary,feed_dict=feed_dictionary{X:x_val,Y:y_val})
    validation_writer.add_summary(summary,epoch)
print('Done training!')

#Make a prediction

#Testing and Evaluation
Image.open('MNIST/test_img.png')
bw=img.convert('L')
img_array=np.invert(bw)
# print(img_array.shape)
test_img=img_array.ravel()
#fetchは出力
prediction=sess.run(fetches=tf.argmax(output,axis=1),feed_dict={X:[test_img]})
print(f'Prediction for test image is {prediction}')
prediction=sess.run(fetches=accuracy,feed_dict={X:x_test,Y:y_test})
accuracy_count=(prediction==y_test).count('True')
accuracy_score=accuracy_count/len(y_test)

"""Reset for the next run"""
train_writer.close()
validation_writer.close()
sess.close()
tf.reset_default_graph()

