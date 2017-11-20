import numpy as np
import time

# THIS IS AN EXAMPLE OF A "FUNCTIONS" FILE FOR THE HOMEWORKS


def generateTrainingTestSets():
    # ... here you will use the evaluateFrankFunction() method
    x_train = 1
    y_train = 1
    x_test = 2
    y_test = 2
    return x_train,y_train,x_test,y_test

def trainMLP(x_train,y_train,numNeurons,w0,b0,v0):
    # ...
    w_opt = 1
    b_opt = 1
    v_opt = 1
    numFunctionEvaluations = 100
    numGradientEvaluations = 50
    trainingComputingTime = 12.3
    return w_opt,b_opt,v_opt,numFunctionEvaluations,numGradientEvaluations,trainingComputingTime

def computeMLPOutput(w,b,v,x):
    return v * (w*x + b)

def computeMeanSquaredError(w,b,v,x,y):
    return (computeMLPOutput(w,b,v,x) - y) ** 2

def evaluateFrankeFunction(x):
    # ... real calculations to evaluate its output
    return x