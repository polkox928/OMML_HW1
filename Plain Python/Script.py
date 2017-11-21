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

############

def franke(x1, x2):
  return (
    .75 * math.exp(-(9 * x1 - 2) * 2 / 4.0 - (9 * x2 - 2) * 2 / 4.0) +
    .75 * math.exp(-(9 * x1 + 1) ** 2 / 49.0 - (9 * x2 + 1) / 10.0) +
    .5 * math.exp(-(9 * x1 - 7) * 2 / 4.0 - (9 * x2 - 3) * 2 / 4.0) -
    .2 * math.exp(-(9 * x1 - 4) * 2 - (9 * x2 - 7) * 2)
  )


############

x1 = np.array([random.uniform(0,1) for i in range(100)])
x2 = np.array([random.uniform(0,1) for i in range(100)])
y = np.array([franke(x1[i],x2[i]) + random.uniform(-0.1, 0.1) for i in range(100)])

#######
X = pd.DataFrame(data = {'x1':x1, 'x2': x2}).values
######

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 1733715)
#print("Number of Samples:\n Train Set: %d\n Test Set: %d" %(len(x_train), len(x_test)))
#######
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.scatter(x1, x2, y)
#plt.show()
####################
# Initialization of hyperparameters
N = 3 # Number of neurons in hidden layer
sigma = 1 # Parameter of hyperbolic tangent (activation function) In tensorflow it is 1
rho = 1 # Learnign rate

# Initialization of model parameters
w = tf.Variable(tf.truncated_normal(shape = [N, 2], seed = seed))
b = tf.Variable(tf.truncated_normal(shape = [N, 1], seed = seed))
v = tf.Variable(tf.truncated_normal(shape = [N, 1], seed = seed))

# Placeholder for input data
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

init = tf.global_variables_initializer()
sess.run(init)

hidden_output = tf.tanh(tf.matmul(w, tf.transpose(x)) - b)
#print(sess.run(hidden_output, {x: [[1., 1.]]}))
output = tf.matmul(tf.transpose(v), hidden_output)

y_hat = sess.run(output, {x: X})[0]
#y_hat

##########
#x = tf.truncated_normal(shape = [1,2])
#print(sess.run(x), tf.shape(x))
hidden_output = tf.tanh(tf.matmul(w, tf.transpose(x)))
xp = [1., 1.]
#print(sess.run(output, {x: [xp]}))
# x and w must be tensors of rank (number of dimensions) 2

# checking shapes
a = tf.shape([[1., 1.]])
sess.run(a)
############
#import tensorflow as tf

class MLP:
    def __init__(self):
        self.hyperparameters = False
        self.fitted = False
        return

    def set_hyperparameters(self, N, rho, sigma = 1):
        self.N = N
        self.rho = rho
        self.sigma = sigma

        self.hyperparameters = True
        return

    def f(self, X, sigma = 1, w = None, b = None, v = None):
        if (not w) and (not b) and (not v):
            print('Please insert model parameters')
            return

        #if len(X.shape) < 2: # Corrects dimension of vectors
         #   X = [X]

        # Placeholder for input data
        x = tf.placeholder(tf.float32)

        hidden_output = tf.tanh(tf.matmul(w, tf.transpose(x)) - b)

        output = tf.matmul(tf.transpose(v), hidden_output)

        return (output, {x: X},)[0]

    ''' def loss(self, X, y, rho, w, b, v, sigma = 1):
        P = len(X)
        omega = tf.concat(values = [w,b,v], axis = 1)

        squared_loss = 1/(2*P)*tf.reduce_sum(tf.squared_difference(self.f(X, sigma, w, b,v), y))
        regularizer = rho*tf.square(tf.norm(omega))

        return sess.run(squared_loss + regularizer)
    '''

    def train(self, x_train, y_train, N = 2, rho = 0.5, sigma = 1, max_iter = 1000, fit = True):
        sess = tf.Session()
        w = tf.Variable(tf.truncated_normal(shape = [N, 2], seed = seed))
        b = tf.Variable(tf.truncated_normal(shape = [N, 1], seed = seed))
        v = tf.Variable(tf.truncated_normal(shape = [N, 1], seed = seed))

        x = tf.placeholder(shape = x_train.shape, dtype = tf.float32)
        y = tf.placeholder(tf.float32)

        hidden_output = tf.tanh(tf.matmul(w, tf.transpose(x)) - b)

        f_out = tf.matmul(tf.transpose(v), hidden_output)

        P = len(x_train)
        omega = tf.concat(values = [w,b,v], axis = 1)

        squared_loss = 1/(2*P)*tf.reduce_sum(tf.squared_difference(f_out, y))
        regularizer = rho*tf.square(tf.norm(omega))

        loss = squared_loss + regularizer

        optimizer = tf.train.GradientDescentOptimizer(0.5)
        train = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(1000):
            sess.run(train, {x: x_train, y: y_train})
            if i %(max_iter/10) == 0:
                # evaluate training accuracy
                curr_W, curr_b, curr_v, curr_loss = sess.run([w, b, v, loss], {x: x_train, y: y_train})
                print("loss: %s"%curr_loss)
        return



        if fit:
            self.fitted = True
            # Set hyperparameters
            self.N = N
            self.rho = rho
            self.sigma = sigma

            # Set model parameters
            self.w = w
            self.b = b
            self.v = v

        if not fit:
            # Returns a dictionary with model parameters values
            return {'w': w,
                    'b': b,
                    'v': v}
    def predict(self, x_new):
        '''To use predict, we must first fit the model'''
        if not self.fitted:
            pass
#print(x1)

################EXACUTiON
print(train(x_train,y_train))
