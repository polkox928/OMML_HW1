# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import library as lib
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
random.seed(1733715)


x1 = np.array([random.uniform(0,1) for i in range(100)])
x2 = np.array([random.uniform(0,1) for i in range(100)])
y = np.array([lib.franke(x1[i],x2[i]) + random.uniform(-0.1, 0.1) for i in range(100)])

dataset = pd.DataFrame(data = {'x1':x1, 'x2': x2, 'y': y})
print(dataset.head())

x_train, x_test, y_train, y_test = train_test_split(dataset[['x1','x2']], 
                                                    dataset['y'], 
                                                    test_size = 0.3, 
                                                    random_state = 1733715)
print("Number of Samples:\n Train Set: %d\n Test Set: %d" %(len(x_train), len(x_test))) 

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x1, x2, y)
plt.show()