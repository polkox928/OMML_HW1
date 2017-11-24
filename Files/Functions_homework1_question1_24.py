# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random
import math
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
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
    
    return x_train, x_test, y_train, y_test

def trainMLP(x_train, y_train, N, rho, max_iter):
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
    w = tf.Variable(tf.truncated_normal(shape = [N, 2], seed = seed))
    b = tf.Variable(tf.truncated_normal(shape = [N, 1], seed = seed))
    v = tf.Variable(tf.truncated_normal(shape = [N, 1], seed = seed))
    
    # Placeholders for train data
    x = tf.placeholder(shape = x_train.shape, dtype = tf.float32)
    y = tf.placeholder(tf.float32)

    
    hidden_output = tf.tanh(tf.matmul(w, tf.transpose(x)) - b) # Output of the hidden layer
    f_out = tf.matmul(tf.transpose(v), hidden_output) # Output of the netword

    P = len(x_train)
    omega = tf.concat(values = [w,b,v], axis = 1) # Just to calculate easily the norm in the regularized term of the loss

    squared_loss = 1/(2*P)*tf.reduce_sum(tf.squared_difference(f_out, y))
    regularizer = rho*tf.square(tf.norm(omega))/2

    loss = squared_loss + regularizer

    optimizer = tf.train.GradientDescentOptimizer(0.005)
    train = optimizer.minimize(loss)
    # Initialize all the tf variables
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(max_iter):
        sess.run(train, {x: x_train, y: y_train})
        if (i+1) %(max_iter/100) == 0:
            curr_loss = sess.run(loss, {x: x_train, y: y_train})
            
            print("\r%3d%% Training MLP, current loss on training set: %0.8f" %((i+1)/max_iter*100, curr_loss), end = '')
    
    opt_W, opt_b, opt_v, loss_value = sess.run([w, b, v, loss], {x: x_train, y: y_train})
    
    def makeMLP(x_new):
        #session = tf.Session()
        X = tf.placeholder(tf.float32)
        hidden_output = tf.tanh(tf.matmul(opt_W, tf.transpose(X)) - opt_b) # Output of the hidden layer
        f_out = tf.matmul(tf.transpose(opt_v), hidden_output) # Output of the network
        output = sess.run(f_out, {X: x_new})
        return output
    
    return makeMLP, w, b, v, loss_value


def grid_search_Nrho(N_values, rho_values, x_train, y_train, max_iter = 10000):
    grid = dict()
    for N in N_values:
        for rho in rho_values:
            print('\n\nN: %d   rho: %e' %(N, rho))
            grid[(N,rho)] = trainMLP(x_train, y_train, N, rho, max_iter)[4]
    return grid

