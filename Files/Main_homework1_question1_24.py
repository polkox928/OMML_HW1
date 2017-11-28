# -*- coding: utf-8 -*-
#%%
import matplotlib.pyplot as plt
import numpy as np
from Functions_homework1_question1_24 import *
#%%
x_train, x_test, x_val, y_train, y_test, y_val = generateTrainTestSet()

grid = grid_search_Nrho([2, 5, 10, 20, 50], [1e-3, 1e-4, 1e-5], x_train, y_train, x_val, y_val, max_iter = 1000)

min_loss = min(grid.values())
opt_hyp = [nrho for nrho, loss in grid.items() if loss == min_loss][0]
N_opt, rho_opt = opt_hyp

print('\nMin Loss: %0.4f' %min_loss)
print('\nOptimal hyperparameter: (%d, %0.1e)' %opt_hyp)

#%%
w_opt, b_opt, v_opt, val_loss = trainMLP(x_train, y_train, x_val, y_val, N_opt, rho_opt, max_iter = 100000)

MLP = makeMLP(w_opt, b_opt, v_opt)

print('\nLoss on train set: %0.8f \nLoss on test set: %0.8f' %(val_loss, compute_loss(MLP(x_test), y_test)))

#%%
# Overfitting
N = N_opt
rho = rho_opt
max_iter = 10000
sess = tf.Session()

# Initialization of model parameters
w = tf.Variable(tf.truncated_normal(shape = [N, 2], seed = seed))
b = tf.Variable(tf.truncated_normal(shape = [N, 1], seed = seed))
v = tf.Variable(tf.truncated_normal(shape = [N, 1], seed = seed))

# Placeholders for train data
x = tf.placeholder(shape = x_train.shape, dtype = tf.float32)
y = tf.placeholder(tf.float32)
x_t = tf.placeholder(tf.float32)
y_t = y_test

hidden_output = tf.tanh(tf.matmul(w, tf.transpose(x)) - b) # Output of the hidden layer
f_out = tf.matmul(tf.transpose(v), hidden_output) # Output of the netword

P = len(x_train)
omega = tf.concat(values = [w,b,v], axis = 1) # Just to calculate easily the norm in the regularized term of the loss

squared_loss = 1/(2*P)*tf.reduce_sum(tf.squared_difference(f_out, y))
regularizer = rho*tf.square(tf.norm(omega))/2

loss = squared_loss + regularizer

optimizer = tf.train.GradientDescentOptimizer(0.005)
train = optimizer.minimize(loss)

hidden_output_t = tf.tanh(tf.matmul(w, tf.transpose(x_t)) - b)
f_out_t = tf.matmul(tf.transpose(v), hidden_output_t)

L = len(x_train)

loss_t = 1/(2*L)*tf.reduce_sum(tf.squared_difference(f_out_t, y_t))

# Initialize all the tf variables
init = tf.global_variables_initializer()
sess.run(init)
train_loss = []
test_loss = []
iterations = []
for i in range(max_iter):
    sess.run(train, {x: x_train, y: y_train})
    if (i+1) %(max_iter/100) == 0:
        tr_loss, te_loss = sess.run([loss, loss_t], {x: x_train, x_t: x_test, y: y_train})
        train_loss.append(tr_loss)
        test_loss.append(te_loss)
        iterations.append(i+1)

plt.figure()
plt.plot(iterations, train_loss)
plt.plot(iterations, test_loss)
plt.show()
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
fig.colorbar(surf, shrink=0.5, aspect=5)
fig.colorbar(surf2, shrink=0.5, aspect=5)

plt.show()

#%%


