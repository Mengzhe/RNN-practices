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
# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rnn"
def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

t_min, t_max = 0, 30
resolution = 0.1
t = np.linspace(t_min, t_max, int((t_max - t_min) / resolution))

def time_series_x1(t):
    return np.cos(t)/3 + np.sin(t) + 0.5

def time_series_x2(t):
    return np.sin(t*2) + np.cos(t*5)/6 + 1

import numpy.random
def time_series_ys(t,x1,x2):
    temp1 = x1*x1
    temp2 = x2*x2
    temp3 = 2*x1*x2
    temp4 = np.sin(t) - 0.5
    return  temp1 + temp2 + temp3 + temp4 + numpy.random.rand(1,x1.shape[0])[0]

x1 = time_series_x1(t)
x2 = time_series_x2(t)
ys = time_series_ys(t,x1,x2) 
data = np.c_[x1, x2, ys]


plt.title("ys", fontsize=14)
plt.plot(t, ys)
plt.legend(loc="upper left")
plt.xlabel("Time")
plt.show()

reset_graph()

n_inputs = 3
n_steps = 10
n_neurons = 100
n_h_1 = 50
n_outputs = 3


def next_batch(num):    
    data_prev = data[num-n_steps:num,:]
    data_current = data[num-n_steps+1:num+1,:]
#    return x1x2ys[:, :-1].reshape(-1, n_steps, 3), x1x2ys[:, 1:].reshape(-1, n_steps, 3)
    return data_prev.reshape(-1, n_steps, 3), data_current.reshape(-1, n_steps, 3)

indices_perm = np.random.permutation(int((t_max - t_min) / resolution) - n_steps) + n_steps
for it in range(indices_perm.shape[0]):
    i = indices_perm[it]
    data_prev, data_current = next_batch(i)
 
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

n_iterations = 30
batch_size = 1

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        indices_perm = np.random.permutation(int((t_max - t_min) / resolution) - n_steps) + n_steps
        for it in range(indices_perm.shape[0]):
            i = indices_perm[it]
            X_batch, y_batch = next_batch(i)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if it % 100 == 0:
                mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
                print(iteration, "\tMSE:", mse)
    saver.save(sess, "./my_time_series_model_0408_02")

with tf.Session() as sess:                          # not shown in the book
    saver.restore(sess, "./my_time_series_model_0408_02")   # not shown
    indices_perm = np.random.permutation(int((t_max - t_min) / resolution) - n_steps) + n_steps
    index = indices_perm[0]
    t_new = t[index-n_steps+1:index+1]
    X_new, y_test = next_batch(i)
#    X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
    y_pred = sess.run(outputs, feed_dict={X: X_new})


plt.plot(t_new, y_test[0,:,0], "b-", markersize=10, label="groundtruth")
plt.plot(t_new, y_pred[0,:,0], "r-", markersize=10, label="prediction")
plt.xlabel("Time")
plt.ylabel("x1")
plt.legend(loc="upper right")
plt.show()

plt.plot(t_new, y_test[0,:,1], "b-", markersize=10, label="groundtruth")
plt.plot(t_new, y_pred[0,:,1], "r-", markersize=10, label="prediction")
plt.xlabel("Time")
plt.ylabel("x2")
plt.legend(loc="upper right")
plt.show()


plt.plot(t_new, y_test[0,:,2], "b-", markersize=10, label="groundtruth")
plt.plot(t_new, y_pred[0,:,2], "r-", markersize=10, label="prediction")
plt.xlabel("Time")
plt.ylabel("y")
plt.legend(loc="upper right")
plt.show()