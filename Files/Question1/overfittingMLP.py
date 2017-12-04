import time

import matplotlib.pyplot as plt
import numpy as np

from Functions_homework1_question1_24 import *

x_train, x_test, y_train, y_test= generateTrainTestSet()

train_losses = []
test_losses = []
Ns = range(1, 205, 5)
for N in Ns:
   print('N: %d' %N)
   te, tr = trainMLP(x_train, y_train, x_test, y_test, N, rho = 1e-5, max_iter = 5000, epsilon = 1e-8)[3:5]
   train_losses.append(tr)
   test_losses.append(te)

train_losses = np.array(train_losses)
test_losses = np.array(test_losses)

fig2 = plt.figure()
plt.plot(Ns, train_losses, '-', label = 'Train losses')
plt.plot(Ns, test_losses, '-', label = 'Test losses')
plt.plot(Ns, abs(train_losses - test_losses), label = 'Loss difference')
plt.xlabel('Number of Neurons')
plt.ylabel('Loss value')
plt.legend(loc = 'best')
plt.title('Number of maximum iterations = 1000,\nrho = 1e-5')
plt.savefig('overfitting%s.png' %time.asctime())
plt.show()