import pandas as pd

df=pd.read_csv('/home/santhosh/Desktop/ai_android/Bike_Sharing_Dataset/day.csv')
#print(df.head())
df=df.drop(['instant'],1).drop(['dteday'],1)
#print(df.head())

import numpy as np

features=np.array(df.drop(['cnt'],1)).astype(np.float32)
labels=np.array(df['cnt']).astype(np.float32)

from sklearn import cross_validation

x_test,x_train,y_test,y_train=cross_validation.train_test_split(features,labels)
#print(x_test,x_train,y_test,y_train)

x_test=np.array(x_test).astype(np.float32)
x_train=np.array(x_train).astype(np.float32)
y_test=np.array(y_test).astype(np.float32)
y_train=np.array(y_train).astype(np.float32)

print(x_test.shape,x_train.shape,y_test.shape,y_train.shape)

import tensorflow as tf

y=tf.placeholder(tf.float32)

m=tf.Variable(tf.truncated_normal(shape=[13,548]),dtype=tf.float32)
x=tf.placeholder(tf.float32)
b=tf.Variable(tf.constant(.1,shape=[1]),dtype=tf.float32)

mx_b=tf.matmul(x,m)+b
erroe=y-mx_b
loss=tf.reduce_mean(tf.square(y-mx_b))

training=tf.train.AdamOptimizer().minimize(loss)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(2000):
    print(i,sess.run([training,tf.reduce_sum(erroe)],feed_dict={x:x_test,y:y_test}))