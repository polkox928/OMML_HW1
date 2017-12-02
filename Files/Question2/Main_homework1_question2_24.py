# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from Functions_homework1_question2_24 import *
learning_rate = 0.05
max_iter = 100000
#%%
x_train, x_test, y_train, y_test= generateTrainTestSet()

N_opt = 20
rho_opt =  1.0e-5

w, b, v_opt, test_loss, train_loss, iterations = ELtrainMLP(x_train, y_train, x_test, y_test, N_opt, rho_opt, learning_rate = learning_rate,  max_iter = max_iter)

MLP = makeMLP(w, b, v_opt)

print('\nLoss on train set: %0.8f \nLoss on test set: %0.8f' %(train_loss, compute_loss(MLP(x_test), y_test)))

#%%
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

grid_X = np.array([[i, j] for i in np.arange(0, 1.0, 0.01) for j in np.arange(0, 1.0, 0.01)])

grid_franke = np.array([franke(x[0],x[1]) for x in grid_X]).reshape((100,100))
grid_Z = MLP(grid_X).reshape((100,100))

fig = plt.figure()
ax = fig.gca(projection='3d')

#ax.scatter(x_test[ : , 0], x_test[ : , 1], y_test)
# Make data.

x = np.array(np.arange(0,1.0, 0.01))
X, Y = np.meshgrid(x, x)


# Plot the surface.
surf = ax.plot_surface(X, Y, grid_Z, cmap=cm.Reds,
                       linewidth=0, antialiased=False, label = 'Approximating function')
surf2 = ax.plot_surface(X, Y, grid_franke, cmap=cm.Blues,
                    linewidth=0, antialiased=False, label = 'Franke function')
#ax.legend(loc = 'best')

# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
plt.show()

plt.figure()
plt.scatter(y_test, MLP(x_test))
plt.show()
