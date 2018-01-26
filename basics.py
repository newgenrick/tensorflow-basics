# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 12:07:42 2018

@author: Abhishek
"""
import tensorflow as tf

print(tf.__version__)
sess = tf.Session()
'''Tensorflow Constants'''
node1 = tf.constant('hello')
node2 = tf.constant(' world')
node3 = node1 + node2
sess.run(node3)

'''the following two lines of code makes a log file inside the current directory 
    inside a file folder which can be use by tensorboard to display the session
    graph after running the following code write the following command to cmd
    >>tensorboard --logdir full/filepath || then navigate to localhost:6006 
    then select graph'''
writer = tf.summary.FileWriter("./file")
writer.add_graph(sess.graph)



print("the type of node1 and node2 is " + str(type(node1)))
print("the type of node3 is "+str(type(node3)))
print(node1)
print(node2)
print(node3)

with tf.Session() as sess:
    print(sess.run(node3))
    
fillmat = tf.fill((3,3),5)
zeromat = tf.zeros((3,3))
onesmat = tf.ones((3,3))
randmat = tf.random_normal((3,3))

allmat = [fillmat,zeromat,onesmat,randmat]

with tf.Session() as sess:
    for mat in allmat:
        print(sess.run(mat))
        print('\n')
    
print(fillmat.get_shape())

with tf.Session() as sess:
    print(sess.run(tf.matmul(onesmat,randmat)))

'''every graph makes a default graph which can be changed by the programmer'''
print(tf.get_default_graph())
temp_graph = tf.Graph() #creates a new graph
default_graph = tf.get_default_graph()
with temp_graph.as_default():
    print(temp_graph is tf.get_default_graph())
print(temp_graph is tf.get_default_graph())


'''all the variables have to be initialised'''


sess =  tf.Session()
r = tf.ones((2,2))
print(r)
t = tf.Variable(initial_value = r)
print(t)
init = tf.global_variables_initializer()
sess.run(init)
sess.run(t)

'''placeholders'''
k = tf.placeholder(tf.float16)



