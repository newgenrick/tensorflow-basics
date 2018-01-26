# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sess  = tf.Session()
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
c = tf.Variable(3)
xph = tf.placeholder(tf.float32,[batch_size])
yph = tf.placeholder(tf.float32,[batch_size])