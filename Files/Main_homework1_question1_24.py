# -*- coding: utf-8 -*-
#%%
import matplotlib.pyplot as plt

from Functions_homework1_question1_24 import *

x_train, x_test, y_train, y_test = generateTrainTestSet()

#losses = [loss for loss in trainMLP(x_train, y_train, 5, 1e-5, 1000)]

#plt.plot(losses)
#plt.show()

#mlp, w, b, v, loss = trainMLP(x_train, y_train, 5, 1e-5, 1000)
# print(mlp(x_test))

grid = grid_search_Nrho([2, 5, 10, 20], [1e-1, 1e-3, 1e-5], x_train, y_train, max_iter = 10000)

min_loss = min(grid.values())
opt_hyp = [nrho for nrho, loss in grid.items() if loss == min_loss][0]


