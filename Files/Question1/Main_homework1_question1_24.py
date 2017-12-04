# -*- coding: utf-8 -*-
#%%

import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from mpl_toolkits.mplot3d import Axes3D

from Functions_homework1_question1_24 import *

# Optimal hyperparameters found via gridsearch
max_iter = 100000
learning_rate = 0.0005
N_opt = 5
rho_opt = 1e-5

# Generate data points
x_train, x_test, y_train, y_test= generateTrainTestSet()


start = time.time()
# Training of the MLP
print('')
w_opt, b_opt, v_opt, test_loss, train_loss, iterations = trainMLP(x_train, y_train,
                                                                  x_test, y_test,
                                                                  N_opt, rho_opt,
                                                                  learning_rate = learning_rate,
                                                                  max_iter = max_iter , epsilon = 1e-8,
                                                                  verbose = True)
print('Training time: %0.f seconds' %(time.time()-start))
MLP = makeMLP(w_opt, b_opt, v_opt)

print('\nLoss on train set: %0.8f \nLoss on test set: %0.8f' %(train_loss, test_loss))

# Plot
grid_X = np.array([[i, j] for i in np.arange(0, 1.0, 0.01) for j in np.arange(0, 1.0, 0.01)])

grid_franke = np.array([franke(x[0],x[1]) for x in grid_X]).reshape((100,100))
grid_Z = MLP(grid_X).reshape((100,100))

fig = plt.figure()
ax = fig.gca(projection='3d')

x = np.array(np.arange(0,1.0, 0.01))
X, Y = np.meshgrid(x, x)

# Plot the surface.
surf = ax.plot_surface(X, Y, grid_Z, cmap=cm.Reds,
                       linewidth=0, antialiased=False, label = 'Approximating function')
surf2 = ax.plot_surface(X, Y, grid_franke, cmap=cm.Blues,
                    linewidth=0, antialiased=False, label = 'Franke function')
# Add the legend
fake2Dline = mpl.lines.Line2D([0],[0], linestyle="none", c='b', marker = 'o')
fake2Dline1 = mpl.lines.Line2D([0],[0], linestyle="none", c='r', marker = 'o')
ax.legend([fake2Dline, fake2Dline1], ["Franke's function", 'Approximating function'], numpoints = 1)
# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

#plt.savefig('MLP%s.png' %time.asctime())
#plt.show()
# Prediction vs True value (Test set)
plt.figure()
plt.scatter(y_test, MLP(x_test))
plt.title('Predictions vs True value')
plt.xlabel('Y Test-set')
plt.ylabel('Predictions on Test-set')
plt.savefig('predMLP%s.png' %time.asctime())
