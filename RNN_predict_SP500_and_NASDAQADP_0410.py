#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 21:34:41 2018

@author: mengzhehuang
"""
import numpy as np
import os
import tensorflow as tf
# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
# To plot pretty figures
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


import pandas as pd

# Import data
data = pd.read_csv('data_stocks.csv')
# Drop date variable
data = data.drop(['DATE'], 1)
# Dimensions of dataset
#n = data.shape[0]
#p = data.shape[1]
# Make data a numpy array

x1 = data['SP500'].values
y = data['NASDAQ.ADP'].values
data = np.c_[x1, y]

n = 40000
data = data[n:,:]
#data = data.reshape((data.shape[0],2))

import numpy.random


plt.title("SP500", fontsize=14)
plt.plot(data)
plt.legend(loc="upper left")
plt.xlabel("Time")
plt.show()

reset_graph()

n_inputs = 2
n_steps = 30
n_neurons = 200
n_h_1 = 50
n_outputs = 2
#
#
def next_batch(num):    
    data_prev = data[num-n_steps:num,:]
    data_current = data[num-n_steps+1:num+1,:]
    return data_prev.reshape(-1, n_steps, n_outputs), data_current.reshape(-1, n_steps, n_outputs)


#indices_perm = np.random.permutation(data.shape[0] - n_steps) + n_steps
#for it in range(indices_perm.shape[0]):
#    i = indices_perm[it]
#    data_prev, data_current = next_batch(i)
# 
    

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])



cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

learning_rate = 0.001

stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs_1 = tf.layers.dense(stacked_rnn_outputs, n_h_1)
stacked_outputs_2 = tf.layers.dense(stacked_outputs_1, n_outputs)
outputs = tf.reshape(stacked_outputs_2, [-1, n_steps, n_outputs])

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_iterations = 2
batch_size = 1

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        indices_perm = np.random.permutation(data.shape[0] - n_steps) + n_steps
        for it in range(indices_perm.shape[0]):
            i = indices_perm[it]
            X_batch, y_batch = next_batch(i)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if it % 100 == 0:
                mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
                print(it, "\tMSE:", mse)
    saver.save(sess, "./my_time_series_NASDAQ_ADP_0410")

with tf.Session() as sess:                          # not shown in the book
    saver.restore(sess, "./my_time_series_NASDAQ_ADP_0410")   # not shown
    indices_perm = np.random.permutation(data.shape[0] - n_steps) + n_steps
#    index = indices_perm[0]
#    t_new = t[index-n_steps+1:index+1]
    X_new, y_test = next_batch(i)
#    X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
    y_pred = sess.run(outputs, feed_dict={X: X_new})


plt.plot(y_test[0,:,0], "b-", markersize=10, label="groundtruth")
plt.plot(y_pred[0,:,0], "r-", markersize=10, label="prediction")
plt.xlabel("Time")
plt.ylabel("x1")
plt.legend(loc="upper right")
plt.show()

plt.plot(y_test[0,:,1], "b-", markersize=10, label="groundtruth")
plt.plot(y_pred[0,:,1], "r-", markersize=10, label="prediction")
plt.xlabel("Time")
plt.ylabel("x1")
plt.legend(loc="upper right")
plt.show()