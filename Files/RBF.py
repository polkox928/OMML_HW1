def trainMLP(x_train, y_train, N, sigma, max_iter, verbose = False):
    """
    Train an N neuron shallow Multilayer Perceptron on the train set
    (x_train, y_train), optimization performed on regularized loss
    with regularization parameter rho; number of iteration of the gradient
    descent optiizer equal to max_iter
    Returns the fitted MLP together with final loss on training set and
    optimized parameters
    """
    sess = tf.Session()
    # Initialization of model parameters
    v = tf.Variable(tf.truncated_normal(shape = [N, 1], seed = seed))
    c = tf.Variable(tf.truncated_normale(shape = [N, 2], seed = seed))
    # Placeholders for train data
    x = tf.placeholder(shape = x_train.shape, dtype = tf.float32)
    y = tf.placeholder(tf.float32)


    hidden_output = exp((-(np.linalg.norm(x-c))/sigma)**2)
    #tf.tanh(tf.matmul(w, tf.transpose(x)) - b) # Output of the hidden layer
    f_out = tf.matmul(tf.transpose(v), hidden_output) # Output of the netword

    P = len(x_train)
    omega = tf.concat(values = [c,v], axis = 1) # Just to calculate easily the norm in the regularized term of the loss

    squared_loss = 1/(2)*tf.reduce_mean(tf.squared_difference(f_out, y))
    regularizer = rho*tf.square(tf.norm(omega))/2

    loss = squared_loss + regularizer

    optimizer = tf.train.GradientDescentOptimizer(0.005)
    train = optimizer.minimize(loss)
    # Initialize all the tf variables
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(max_iter):
        sess.run(train, {x: x_train, y: y_train})
        if (i+1) %(max_iter/100) == 0 and verbose == True:
            curr_loss = sess.run(loss, {x: x_train, y: y_train})
            print("\r%3d%% Training RBF, current loss on training set: %0.8f" %((i+1)/max_iter*100, curr_loss), end = '')

    opt_c, opt_v, loss_value = sess.run([c, v, loss], {x: x_train, y: y_train})
    return opt_c, opt_v, loss_value


def makeRBF(c, v):
    def RBF(x_new):
        sess = tf.Session()
        X = tf.placeholder(tf.float32)
        hidden_output = exp((-(np.linalg.norm(x-c))/sigma)**2) # Output of the hidden layer
        f_out = tf.matmul(tf.transpose(v), hidden_output) # Output of the network
        output = sess.run(f_out, {X: x_new})
        return output[0]
    return RBF



def compute_loss(y_h, y_t):
    sess = tf.Session()
    P = len(y_t)
    y_hat = tf.placeholder(dtype = tf.float32)
    y_true = tf.placeholder(dtype = tf.float32)

    loss = 1/(2)*tf.reduce_mean(tf.squared_difference(y_hat, y_true))

    output = sess.run(loss, {y_hat : y_h, y_true: y_t})

    return output


def grid_search_NrhoSigma(N_values, rho_values, sigma_values, x_train, y_train, max_iter = 10000):
    grid = dict()
    for N in N_values:
        for rho in rho_values:
            for sigma in sigma_values:
                print('\nN: %d   rho: %0.1e  sigma: %0.1f' %(N, rho, sigma))
                grid[(N, rho, sigma)] = trainRBF(x_train, y_train, N, rho, max_iter)[3]
    return grid

