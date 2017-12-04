import numpy as np
from Functions_homework1_question1_24 import *

max_iter = 10000
learning_rate = 0.005

x_train, x_test, y_train, y_test = generateTrainTestSet()

grid = grid_search_Nrho([2, 5, 20, 50], [1e-3, 1e-4, 1e-5], x_train, y_train, x_test, y_test,learning_rate = learning_rate, epsilon = 1e-8, max_iter = max_iter, verbose = True)

min_loss = min(grid.values())
opt_hyp = [nrho for nrho, loss in grid.items() if loss == min_loss][0]
N_opt, rho_opt = opt_hyp # N = 5, rho =  1.0e-5

print('\nMin Loss: %0.4f' %min_loss)
print('\nOptimal hyperparameter: (%d, %0.1e)' %opt_hyp)