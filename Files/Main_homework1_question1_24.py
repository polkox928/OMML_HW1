# -*- coding: utf-8 -*-
#%%
import matplotlib.pyplot as plt
import numpy as np
import time
from Functions_homework1_question1_24 import *
max_iter = 1000
#%%
x_train, x_test, y_train, y_test= generateTrainTestSet()

grid = grid_search_Nrho([10, 20, 50, 100], [1e-3, 1e-4, 1e-5], x_train, y_train, x_test, y_test, epsilon = 1e-8, max_iter = max_iter)

min_loss = min(grid.values())
opt_hyp = [nrho for nrho, loss in grid.items() if loss == min_loss][0]
N_opt, rho_opt = opt_hyp # N = 5, rho =  1.0e-5

print('\nMin Loss: %0.4f' %min_loss)
print('\nOptimal hyperparameter: (%d, %0.1e)' %opt_hyp)

#%%
start = time.time()
w_opt, b_opt, v_opt, test_loss, train_loss, iterations = trainMLP(x_train, y_train, x_test, y_test, N_opt, rho_opt, max_iter = max_iter, epsilon = 1e-8, verbose = True)
print('Training time: %0.f seconds' %(time.time()-start))
MLP = makeMLP(w_opt, b_opt, v_opt)

print('\nLoss on train set: %0.8f \nLoss on test set: %0.8f' %(train_loss, compute_loss(MLP(x_test), y_test)))

#%%
# =============================================================================
# # Overfitting
# N = N_opt
# rho = rho_opt
# max_iter = 1000
# sess = tf.Session()
#
# # Initialization of model parameters
# w = tf.Variable(tf.truncated_normal(shape = [N, 2], seed = seed))
# b = tf.Variable(tf.truncated_normal(shape = [N, 1], seed = seed))
# v = tf.Variable(tf.truncated_normal(shape = [N, 1], seed = seed))
#
# # Placeholders for train data
# x = tf.placeholder(shape = x_train.shape, dtype = tf.float32)
# y = tf.placeholder(tf.float32)
# x_t = tf.placeholder(tf.float32)
# y_t = y_test
#
# hidden_output = tf.tanh(tf.matmul(w, tf.transpose(x)) - b) # Output of the hidden layer
# f_out = tf.matmul(tf.transpose(v), hidden_output) # Output of the netword
#
#
# omega = tf.concat(values = [w,b,v], axis = 1) # Just to calculate easily the norm in the regularized term of the loss
#
# squared_loss = 1/2*tf.reduce_mean(tf.squared_difference(f_out, y))
# regularizer = rho*tf.square(tf.norm(omega))/2
#
# loss = squared_loss + regularizer
#
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)
#
# hidden_output_t = tf.tanh(tf.matmul(w, tf.transpose(x_t)) - b)
# f_out_t = tf.matmul(tf.transpose(v), hidden_output_t)
#
#
# loss_t = 1/2*tf.reduce_mean(tf.squared_difference(f_out_t, y_t))
#
# # Initialize all the tf variables
# init = tf.global_variables_initializer()
# sess.run(init)
# train_loss = []
# test_loss = []
# iterations = []
# for i in range(max_iter):
#     sess.run(train, {x: x_train, y: y_train})
#     if (i+1) %(max_iter/100) == 0:
#         tr_loss, te_loss = sess.run([loss, loss_t], {x: x_train, x_t: x_test, y: y_train})
#         train_loss.append(tr_loss)
#         test_loss.append(te_loss)
#         iterations.append(i+1)
#         if abs(prev_loss - curr_loss) < epsilon or curr_loss > prev_loss:
#                 break
#             prev_loss = curr_loss
#
#
#
# plt.plot(iterations, train_loss)
# plt.plot(iterations, test_loss)
# plt.show()
# =============================================================================
#%%
# Plot


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

grid_X = [[i, j] for i in np.arange(0, 1.0, 0.01) for j in np.arange(0, 1.0, 0.01)]

grid_franke = np.array([franke(x[0],x[1]) for x in grid_X]).reshape((100,100))
grid_Y = MLP(grid_X).reshape((100,100))

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.

x = np.array(np.arange(0,1.0, 0.01))
X, Y = np.meshgrid(x, x)


# Plot the surface.
surf = ax.plot_surface(X, Y, grid_Y, cmap=cm.Reds,
                       linewidth=0, antialiased=False)
surf2 = ax.plot_surface(X, Y, grid_franke, cmap=cm.Blues,
                    linewidth=0, antialiased=False)

# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

plt.show()
#%%
train_losses = []
test_losses = []
Ns = range(70, 170, 5)
for N in Ns:
    print('N: %d' %N)
    te, tr = trainMLP(x_train, y_train, x_test, y_test, N, rho = 1e-5, max_iter = max_iter, epsilon = 1e-8)[3:5]
    train_losses.append(tr)
    test_losses.append(te)

train_losses = np.array(train_losses)
test_losses = np.array(test_losses)

fig2 = plt.figure()
plt.plot(Ns, train_losses, '-', label = 'Train losses')
plt.plot(Ns, test_losses, '-', label = 'Test losses')
plt.plot(Ns, abs(train_losses - test_losses), label = 'Loss difference')
plt.legend(loc = 'best')
plt.title('Number of maximum iterations = 1000,\nrho = 1e-5')

plt.show()

