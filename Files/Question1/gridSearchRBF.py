#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 18:30:03 2017

@author: paolograniero
"""
import numpy as np
from RBF import *


max_iter = 12500

x_train, x_test, y_train, y_test = generateTrainTestSet()

grid = grid_search_NrhoSigma([4, 10, 30], [1e-4, 1e-5],[0.3, 1.0, 2.0], x_train, y_train, x_test, y_test, max_iter = max_iter, verbose = True)

min_loss = min(grid.values())
opt_hyp = [nrhosigma for nrhosigma, loss in grid.items() if loss == min_loss][0]
N_opt, rho_opt, sigma_opt = opt_hyp

print('\nMin Loss: %0.5f' %min_loss)
print('\nOptimal hyperparameter: (%d, %0.1e, %0.2f)' %opt_hyp)
