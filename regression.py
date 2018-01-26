# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#sess  = tf.Session()
x = np.linspace(1,10,1000)
noise = np.random.normal(size = len(x))
y = 0.5*x + 4 + noise
x_df = pd.DataFrame(x,columns = ['input'])
y_df = pd.DataFrame(y,columns = ['output'])

data = pd.concat([x_df,y_df],axis = 1)

plt.scatter(x_df,y_df)
plt.show()

batch_size = 4
m = tf.Variable(0.4)
c = tf.Variable(3.0)
xph = tf.placeholder(tf.float32,[batch_size])
yph = tf.placeholder(tf.float32,[batch_size])
y_pred = m*xph + c
error = tf.reduce_sum(tf.square(yph-y_pred))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()
with tf.Session() as sess :
    
    sess.run(init)
    batches = 200
    for i in range(batches):
        rand_ind = np.random.randint(len(x),size = batch_size)
        feed = {xph : x[rand_ind],yph : y[rand_ind]}
        sess.run(train , feed_dict = feed)   
    writer = tf.summary.FileWriter("./file")
    writer.add_graph(sess.graph)
    mnew , cnew = sess.run([m,c])
        
print(mnew,cnew)
    