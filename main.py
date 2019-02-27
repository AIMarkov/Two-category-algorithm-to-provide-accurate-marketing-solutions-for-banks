import pandas as pd
from process import DATA
import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'#解决The TensorFlow library wasn't compiled to use SSE instructions, but these are available on your machine and could speed up CPU computations.
data=pd.read_csv("train_set.csv")
DATA_=DATA(data)
DATA_.processed()
# dataX,label=DATA_.samplpe(2)
# print(np.asarray(label))
# exit()



x=tf.placeholder("float32",shape=[None,16])
y_=tf.placeholder("float32",shape=[None,2])
#构造全连接神经模型
W1=tf.Variable(tf.random_normal([16,200]))
W2=tf.Variable(tf.random_normal([200,100]))
W3=tf.Variable(tf.random_normal([100,2]))
b1=tf.Variable(tf.zeros([200]))
b2=tf.Variable(tf.zeros([100]))
b3=tf.Variable(tf.zeros([2]))

out=tf.sigmoid(tf.matmul(x,W1)+b1)
out=tf.tanh(tf.matmul(out,W2)+b2)
out=tf.matmul(out,W3)+b3
out=tf.nn.sigmoid(out)


#损失函数和优化器
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(out),reduction_indices=[1]))
trainer=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_pre=tf.equal(tf.argmax(out,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_pre,"float"))
#初始化变量,放在最后注意一定
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
for i in range(100000):
    dataX,label=DATA_.samplpe(128)
    sess.run(trainer,feed_dict={x:np.asarray(dataX),y_:np.asarray(label)})
    print("accuracy:",sess.run(accuracy,feed_dict={x:np.asarray(dataX),y_:np.asarray(label)}))



