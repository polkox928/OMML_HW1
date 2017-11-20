import numpy as np
import time
import Functions_homework1_question1_99 as f


# THIS IS AN EXAMPLE OF A "MAIN" FILE FOR THE HOMEWORKS


# some parameters must be initialized in the Main file

nNeurons = 10
w0 = 0
b0 = 0
v0 = 0


# then the calculations are made using the methods written in the Functions file

x_train,y_train,x_test,y_test = f.generateTrainingTestSets()

w_opt,b_opt,v_opt,numFunctionEvaluations,numGradientEvaluations,trainingComputingTime = f.trainMLP(x_train,y_train,nNeurons,w0,b0,v0)

MSE = f.computeMeanSquaredError(w_opt,b_opt,v_opt,x_test,y_test)


# finally the outputs are written in the output file
# IMPORTANT: the "output_homework1_teamNumber" file must be unique and must contain the outputs of all the 3 questions of the homework

output = open("output_homework1_99.txt","a")
output.write("This is homework 1: question 1")
output.write("\nTest MSE," + "%f" % MSE)
output.write("\nTraining computing time," + "%f" % trainingComputingTime)
output.write("\nFunction evaluations," + "%i" % numFunctionEvaluations)
output.write("\nGradient evaluations," + "%i" % numGradientEvaluations)
output.close()