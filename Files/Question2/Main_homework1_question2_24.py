# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from Functions_homework1_question2_24 import *
#%%
x_train, x_test, y_train, y_test= generateTrainTestSet()

N_opt = 5
rho_opt =  1.0e-5

w, b, v_opt, test_loss, train_loss = ELtrainMLP(x_train, y_train, x_test, y_test, N_opt, rho_opt, max_iter = 5000)

MLP = makeMLP(w, b, v_opt)

print('\nLoss on train set: %0.8f \nLoss on test set: %0.8f' %(train_loss, compute_loss(MLP(x_test), y_test)))
