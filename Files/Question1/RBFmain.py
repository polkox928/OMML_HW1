# -*- coding: utf-8 -*-
#%%
import matplotlib.pyplot as plt
import math
import numpy as np
from RBF import *
max_iter = 1000
#%%
x_train, x_test, y_train, y_test = generateTrainTestSet()

grid = grid_search_NrhoSigma([4, 10, 30], [1e-4, 1e-5],[0.1, 1.0, 2.0], x_train, y_train, x_test, y_test, max_iter = max_iter, verbose = True)

min_loss = min(grid.values())
opt_hyp = [nrhosigma for nrhosigma, loss in grid.items() if loss == min_loss][0]
N_opt, rho_opt, sigma_opt = opt_hyp

print('\nMin Loss: %0.5f' %min_loss)
print('\nOptimal hyperparameter: (%d, %0.1e, %0.2f)' %opt_hyp)

#%%
c_opt, v_opt, test_loss, val_loss = trainRBF(x_train, y_train, x_test, y_test, N_opt, rho_opt, sigma_opt, max_iter = max_iter*10, verbose = True)

RBF = makeRBF(c_opt, v_opt, sigma_opt)

print('\nLoss on train set: %0.8f \nLoss on test set: %0.8f' %(val_loss, compute_loss(RBF(x_test), y_test)))

##%%

# Plot


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

grid_X = [[i, j] for i in np.arange(0, 1.0, 0.01) for j in np.arange(0, 1.0, 0.01)]

grid_franke = np.array([franke(x[0],x[1]) for x in grid_X]).reshape((100,100))
grid_Y = RBF(grid_X).reshape((100,100))

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.

x1 = np.array(np.arange(0,1.0, 0.01))
X, Y = np.meshgrid(x1, x1)


# Plot the surface.
surf = ax.plot_surface(X, Y, grid_Y, cmap=cm.Reds,
                       linewidth=0, antialiased=False)
surf2 = ax.plot_surface(X, Y, grid_franke, cmap=cm.Blues,
                    linewidth=0, antialiased=False)

# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.


plt.show()
