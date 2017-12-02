# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random
import math
from sklearn.model_selection import train_test_split
import tensorflow as tf
sess = tf.Session()
seed = 1733715
random.seed(seed)

def franke(x1, x2):
    """
    The true function we have to approximate
    """
    return(
    .75 * math.exp(-(9 * x1 - 2) ** 2 / 4.0 - (9 * x2 - 2) ** 2 / 4.0) +
    .75 * math.exp(-(9 * x1 + 1) ** 2 / 49.0 - (9 * x2 + 1) / 10.0) +
    .5 * math.exp(-(9 * x1 - 7) ** 2 / 4.0 - (9 * x2 - 3) ** 2 / 4.0) -
    .2 * math.exp(-(9 * x1 - 4) ** 2 - (9 * x2 - 7) ** 2)
  )

def generateTrainTestSet():
    """
    Generate 100 datapoints, randomly chosen in the square [0,1]x[0,1], with output given by the franke function plus a uniform random noise
    """
    x1 = np.array([random.uniform(0,1) for i in range(100)])
    x2 = np.array([random.uniform(0,1) for i in range(100)])
    y = np.array([franke(x1[i],x2[i]) + random.uniform(-0.1, 0.1) for i in range(100)])
    X = pd.DataFrame(data = {'x1':x1, 'x2': x2}).values

    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 1733715)

    return x_train, x_test, y_train.reshape((-1, 1)), y_test.reshape((-1, 1))

def ELtrainMLP(x_train, y_train, x_test, y_test, N, rho, learning_rate, max_iter, verbose = False):
    """
    Train an N neuron shallow Multilayer Perceptron on the train set
    (x_train, y_train), optimization performed on regularized loss
    with regularization parameter rho; number of iteration of the gradient
    descent optiizer equal to max_iter
    Returns the fitted MLP together with final loss on training set and
    optimized parameters
    """
    sess = tf.Session()

    # Initialization of model parameters
    w = tf.Variable(tf.truncated_normal(shape = [N, 2], seed = seed), trainable = False)
    b = tf.Variable(tf.truncated_normal(shape = [N, 1], seed = seed), trainable = False)
    v = tf.Variable(tf.truncated_normal(shape = [N, 1], seed = seed))

    # Placeholders for train data
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    # Output of the hidden layer
    hidden_output = tf.tanh(tf.matmul(w, tf.transpose(x)) - b)

    # Output of the network
    f_out = tf.matmul(tf.transpose(hidden_output), v)

    omega = tf.concat(values = [w,b,v], axis = 1) # Just to calculate easily the norm in the regularized term of the loss

    squared_loss = (1/2)*tf.reduce_mean(tf.squared_difference(f_out, y))
    regularizer = rho*tf.square(tf.norm(omega))/2

    loss = squared_loss + regularizer

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    # Initialize all the tf variables
    init = tf.global_variables_initializer()
    sess.run(init)

    #prev_loss = 100.0
    gradient_norm = 1
    i = 0
    while i < max_iter:
        #for i in range(max_iter):
        sess.run(train, {x: x_train, y: y_train})

        curr_loss = sess.run(loss, {x: x_train, y: y_train})

        if verbose == True and (i+1)%(max_iter/1000) == 0:
            print("\r%3d%% Training MLP, current loss: %0.8f" %((i+1)/max_iter*100, curr_loss), end = '')
#        if abs(prev_loss - curr_loss) < epsilon:
#            break
        #prev_loss = curr_loss
        i += 1
    if verbose: print('')
    print('Number of iterations: %d' %(i+1))
    opt_W, opt_b, opt_v, test_loss = sess.run([w, b, v, squared_loss], {x: x_test, y: y_test})
    train_loss = sess.run(loss, {x: x_train, y: y_train})
    sess.close()
    return opt_W, opt_b, opt_v, test_loss, train_loss, i+1


def makeMLP(w, b, v):
    def MLP(x_new):
        sess = tf.Session()
        X = tf.placeholder(tf.float32)
        hidden_output = tf.tanh(tf.matmul(w, tf.transpose(X)) - b) # Output of the hidden layer
        f_out = tf.matmul(tf.transpose(v), hidden_output) # Output of the network
        output = sess.run(f_out, {X: x_new})
        sess.close()
        return output
    return MLP



def compute_loss(y_h, y_t):
    sess = tf.Session()

    y_hat = tf.placeholder(dtype = tf.float32)
    y_true = tf.placeholder(dtype = tf.float32)

    loss = 1/2*tf.reduce_mean(tf.squared_difference(y_hat, y_true))

    output = sess.run(loss, {y_hat : y_h, y_true: y_t})
    sess.close()
    return output