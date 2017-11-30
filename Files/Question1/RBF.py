import tensorflow as tf
import numpy as np
import pandas as pd
import random
import math
sess = tf.Session()
seed = 1733715
random.seed(seed)
from sklearn.model_selection import train_test_split

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



def trainRBF(x_train, y_train, N, rho, sigma, max_iter=1000, verbose = False):
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
    v = tf.Variable(tf.truncated_normal(shape = [N, 1], seed = seed, dtype = tf.float64))
    c = tf.Variable(tf.truncated_normal(shape = [N, 2], seed = seed, dtype = tf.float64))
    # Placeholders for train data
    X = tf.placeholder(tf.float64)
    y = tf.placeholder(tf.float64)


    init = tf.global_variables_initializer()
    sess.run(init)

    P = x_train.shape[0]

    norma = np.zeros(shape = (P, N))

    C = sess.run(c)
    for i in range(P):
        for j in range(N):
            x = x_train[i]
            Cj = C[j]
            norma[i, j] = np.linalg.norm(x-Cj)



    hidden_output = tf.exp(-tf.pow(norma/sigma,2))
    #tf.tanh(tf.matmul(w, tf.transpose(x)) - b) # Output of the hidden layer
    f_out = tf.matmul(hidden_output,v) # Output of the netword

    #P = len(x_train)
    omega = tf.concat(values = [c,v], axis = 1) # Just to calculate easily the norm in the regularized term of the loss

    squared_loss = 1/(2)*tf.reduce_mean(tf.squared_difference(f_out, y))
    regularizer = rho*tf.square(tf.norm(omega))/2

    loss = squared_loss + regularizer

    optimizer = tf.train.GradientDescentOptimizer(0.005)
    train = optimizer.minimize(loss)
    # Initialize all the tf variables


    for i in range(max_iter):
        sess.run(train, {X: x_train, y: y_train})
        if (i+1) %(max_iter/100) == 0 and verbose == True:
            curr_loss = sess.run(loss, {X: x_train, y: y_train})
            print("\r%3d%% Training RBF, current loss on training set: %0.8f" %((i+1)/max_iter*100, curr_loss), end = '')

    opt_c, opt_v, loss_value = sess.run([c, v, loss], {X: x_train, y: y_train})
    return opt_c, opt_v, loss_value


def makeRBF(c, v,sigma):
    def RBF(x_new):
        sess = tf.Session()
        X = tf.placeholder(tf.float32)


        P = x_new.shape[0]
        N = c.shape[0]
        norma = tf.zeros(shape=(P, N))
        for i in range(len(X)):
            for j in range(len(c)):
                x = X[i]
                C = c[j]
                norma[i,j] = tf.norm(x-C)

        hidden_output = tf.exp((-norma/sigma)**2) # Output of the hidden layer
        f_out = tf.matmul(hidden_output, v) # Output of the network
        output = sess.run(f_out, {X: x_new})
        return output[0]
    return RBF



def compute_loss(y_h, y_t):
    sess = tf.Session()
    #P = len(y_t)
    y_hat = tf.placeholder(dtype = tf.float32)
    y_true = tf.placeholder(dtype = tf.float32)

    loss = 1/(2)*tf.reduce_mean(tf.squared_difference(y_hat, y_true))

    output = sess.run(loss, {y_hat : y_h, y_true: y_t})

    return output


def grid_search_NrhoSigma(N_values, rho_values, sigma_values, x_train, y_train, max_iter = 10000):
    grid = dict()
    for N in N_values:
        for rho in rho_values:
            for sigma in sigma_values:
                print('\nN: %d   rho: %0.1e  sigma: %0.1f' %(N, rho, sigma))
                grid[(N, rho, sigma)] = trainRBF(x_train, y_train, N, rho, sigma, max_iter)[2]
    return grid
