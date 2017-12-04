# -*- coding: utf-8 -*-
#%%
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from mpl_toolkits.mplot3d import Axes3D

from RBF import *

max_iter = 20000
epsilon = 1e-5
learning_rate = 0.0005

x_train, x_test, y_train, y_test = generateTrainTestSet()

sigma_opt = 0.3
N_opt = 10
rho_opt = 1e-5

start = time.time()
c_opt, v_opt, test_loss, train_loss = trainRBF(x_train, y_train,
                                               x_test, y_test,
                                               N_opt, rho_opt, sigma_opt,
                                               learning_rate = learning_rate,
                                               max_iter = max_iter,
                                               epsilon = epsilon,
                                               verbose = True)
print('\nTraining time: %d seconds'%(time.time()-start))
RBF = makeRBF(c_opt, v_opt, sigma_opt)

print('\nLoss on train set: %0.8f \nLoss on test set: %0.8f' %(train_loss, test_loss))

# Plot
grid_X = np.array([[i, j] for i in np.arange(0, 1.0, 0.01) for j in np.arange(0, 1.0, 0.01)])

grid_franke = np.array([franke(x[0], x[1]) for x in grid_X]).reshape((100,100))
grid_Y = RBF(grid_X).reshape((100,100))

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.

x1 = np.array(np.arange(0,1.0, 0.01))
X, Y = np.meshgrid(x1, x1)


# Plot the surface.
surf = ax.plot_surface(X, Y, grid_Y, cmap = cm.Reds, linewidth=0, antialiased=False)
surf2 = ax.plot_surface(X, Y, grid_franke, cmap = cm.Blues, linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add the legend
fake2Dline = mpl.lines.Line2D([0],[0], linestyle="none", c='b', marker = 'o')
fake2Dline1 = mpl.lines.Line2D([0],[0], linestyle="none", c='r', marker = 'o')
ax.legend([fake2Dline, fake2Dline1], ["Franke's function", 'Approximating function'], numpoints = 1)
# Save the figure to a .png
plt.savefig('RBF%s.png'%time.asctime())
