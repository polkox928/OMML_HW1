# -*- coding: utf-8 -*-
#%%
import matplotlib.pyplot as plt
import numpy as np
import time
from Functions_homework1_question1_24 import *
max_iter = 10000
learning_rate = 0.05
#%%
x_train, x_test, y_train, y_test= generateTrainTestSet()

grid = grid_search_Nrho([5, 20], [1e-3, 1e-4, 1e-5], x_train, y_train, x_test, y_test,learning_rate = learning_rate, epsilon = 1e-8, max_iter = max_iter, verbose = True)

min_loss = min(grid.values())
opt_hyp = [nrho for nrho, loss in grid.items() if loss == min_loss][0]
N_opt, rho_opt = opt_hyp # N = 5, rho =  1.0e-5

print('\nMin Loss: %0.4f' %min_loss)
print('\nOptimal hyperparameter: (%d, %0.1e)' %opt_hyp)

#%%
start = time.time()
w_opt, b_opt, v_opt, test_loss, train_loss, iterations = trainMLP(x_train, y_train,
                                                                  x_test, y_test,
                                                                  N_opt, rho_opt,
                                                                  learning_rate = learning_rate,
                                                                  max_iter = max_iter , epsilon = -1e-8,
                                                                  verbose = True)
print('Training time: %0.f seconds' %(time.time()-start))
MLP = makeMLP(w_opt, b_opt, v_opt)

print('\nLoss on train set: %0.8f \nLoss on test set: %0.8f' %(train_loss, compute_loss(MLP(x_test), y_test)))


#%%
# Plot


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
#%%
#train_losses = []
#test_losses = []
#Ns = range(70, 170, 5)
#for N in Ns:
#    print('N: %d' %N)
#    te, tr = trainMLP(x_train, y_train, x_test, y_test, N, rho = 1e-5, max_iter = max_iter, epsilon = 1e-8)[3:5]
#    train_losses.append(tr)
#    test_losses.append(te)
#
#train_losses = np.array(train_losses)
#test_losses = np.array(test_losses)
#
#fig2 = plt.figure()
#plt.plot(Ns, train_losses, '-', label = 'Train losses')
#plt.plot(Ns, test_losses, '-', label = 'Test losses')
#plt.plot(Ns, abs(train_losses - test_losses), label = 'Loss difference')
#plt.legend(loc = 'best')
#plt.title('Number of maximum iterations = 1000,\nrho = 1e-5')
#
#plt.show()
#
