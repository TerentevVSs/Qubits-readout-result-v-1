import numpy as np
import matplotlib.pyplot as plt
import csv

param = 0


def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions
    of each layer in our network
    Returns:
    parameters -- python dictionary containing your parameters
    "W1", "b1", ..., "WL", "bL": Wl -- weight matrix of shape
    (layer_dims[l], layer_dims[l-1])
    bl -- bias vector of shape (layer_dims[l], 1)
    """
    global param
    parameters = {}
    L = len(layer_dims)  # number of layers in the network
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],
                                                   layer_dims[
                                                       l - 1]) * 0.01 * np.sqrt(
            2 / layers_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    param = parameters
    return parameters

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.
    Arguments:
    A -- activations from previous layer (or input data): (size of
    previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer,
    size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer,1)
    Returns:
    Z -- the input of the activation function, also called
    pre-activation parameter
    cache -- a python tuple containing "A", "W" and "b" ; stored for
    computing the backward pass efficiently
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def sigmoid(Z):
    """
    Numpy sigmoid activation implementation
    Arguments:
    Z - numpy array of any shape
    Returns:
    A - output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z):
    """
    Numpy Relu activation implementation
    Arguments:
    Z - Output of the linear layer, of any shape
    Returns:
    A - Post-activation parameter, of the same shape as Z
    cache - a python dictionary containing "A"; stored for computing
    the backward pass efficiently
    """
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer
    Arguments:
    A_prev -- activations from previous layer (or input data): (size of
    previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer,
    size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer,1)
    activation -- the activation to be used in this layer, stored as a
    text string: "sigmoid" or "relu"
    Returns:
    A -- the output of the activation function, also called the
    post-activation value
    cache -- a python tuple containing "linear_cache" and
    "activation_cache";
    stored for computing the backward pass efficiently
    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    else:
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->
    SIGMOID computation
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing: every cache of
    linear_activation_forward()
    (there are L-1 of them, indexed from 0 to L-1)
    """
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network
    # Implement [LINEAR -> RELU]*(L-1).Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev,
                                             parameters['W' + str(l)],
                                             parameters['b' + str(l)],
                                             "relu")
        caches.append(cache)
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)],
                                          parameters['b' + str(L)],
                                          "sigmoid")
    caches.append(cache)
    return AL, caches


def compute_cost(AL, Y, parameters, lambd):
    """
    Implement the cost function defined.
    Arguments:
    AL -- probability vector corresponding to your label predictions,
    shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if |0>,
    1 if |1>), shape (1, number of examples)
    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]
    # Compute loss from AL and y.
    cost = -1 / m * np.sum(
        Y * np.log(AL + 10 ** (-10)) + (1 - Y) * np.log(
            1 + 10 ** (-10) - AL))
    k = 1 / m * lambd / 2
    L = len(parameters) // 2  # number of layers in the neural network
    for l in range(1, L):
        cost = cost + k * np.sum(np.square(parameters['W' + str(l)]))
    cost = np.squeeze(cost)
    return cost


def linear_backward(dZ, cache, lambd):
    """
    Implement the linear portion of backward propagation for
    a single layer (layer l)
    Arguments:
    dZ -- Gradient of the cost with respect to the linear output
    (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward
    propagation in the current layer
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation
    (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l),
    same shape as W
    db -- Gradient of the cost with respect to b (current layer l),
    same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1 / m * np.dot(dZ, A_prev.T) + lambd / m * W
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def relu_backward(dA, cache):
    """
    The backward propagation for a single RELU unit.
    Arguments:
    dA - post-activation gradient, of any shape
    cache - 'Z' where we store for computing backward propagation
    efficiently
    Returns:
    dZ - Gradient of the cost with respect to Z
    """
    Z = cache
    # just converting dz to a correct object.
    dZ = np.array(dA, copy=True)
    # When z <= 0, we should set dz to 0 as well.
    dZ[Z <= 0] = 0
    return dZ


def sigmoid_backward(dA, cache):
    """
    The backward propagation for a single SIGMOID unit.
    Arguments:
    dA - post-activation gradient, of any shape
    cache - 'Z' where we store for computing backward propagation
    efficiently
    Returns:
    dZ - Gradient of the cost with respect to Z
    """
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


def linear_activation_backward(dA, cache, activation, lambd):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store
    for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as
    a text string: "sigmoid" or "relu"
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation
    (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l),
    same shape as W
    db -- Gradient of the cost with respect to b (current layer l),
    same shape as b
    """
    dA_prev = []
    dW = []
    db = []
    if activation == "relu":
        dZ = relu_backward(dA, cache[1])
        dA_prev, dW, db = linear_backward(dZ, cache[0], lambd)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, cache[1])
        dA_prev, dW, db = linear_backward(dZ, cache[0], lambd)
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches, lambd):
    grads = {}
    # the number of layers
    L = len(caches)
    # after this line, Y is the same shape as AL
    Y = Y.reshape(AL.shape)
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL + 10 ** (-10)) - np.divide(1 - Y,
                                                        1 + 10 ** (
                                                            -10) - AL))
    # Lth layer (SIGMOID -> LINEAR) gradients.
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads[
        "db" + str(L)] = linear_activation_backward(dAL, current_cache,
                                                    "sigmoid", lambd)
    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            grads["dA" + str(l + 1)], current_cache, "relu", lambd)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of
    L_model_backward
    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """
    global param
    L = len(parameters) // 2  # number of layers in the neural network
    # Update rule for each parameter.
    for l in range(L):
        parameters["W" + str(l + 1)] -= grads["dW" + str(
            l + 1)] * learning_rate
        parameters["b" + str(l + 1)] -= grads["db" + str(
            l + 1)] * learning_rate
    param = parameters
    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075,
                  num_iterations=3000,
                  print_cost=False,
                  lambd=0,
                  rate="start"):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->
    SIGMOID.
    Arguments:
    X -- data, numpy array of shape (2, number of examples)
    Y -- true "label" vector (containing 0 if |0>, 1 if |1>),
    of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size,
    of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    Returns:
    parameters -- parameters learnt by the model.
    They can then be used to predict.
    """
    global param
    costs = []  # keep track of cost
    if rate == "start":
        parameters = initialize_parameters_deep(layers_dims)
    else:
        parameters = param
    # gradient descent
    for i in range(0, num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) ->
        # LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        # Compute cost.
        cost = compute_cost(AL, Y, parameters, lambd)
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches, lambd)
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: " % i, cost)
        if print_cost and i % 100 == 0:
            costs.append(cost)
    # plot the cost
    """
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    """
    param = parameters
    return parameters


def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """
    m = X.shape[1]
    p = np.zeros((1, m))
    # Forward propagation
    probabilities, caches = L_model_forward(X, parameters)
    # convert probabilities to 0/1 predictions
    for i in range(0, probabilities.shape[1]):
        if probabilities[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print("Accuracy: " + str(np.sum((p == y) / m)))
    return p


def save_parameter(parameter_name, parameters):
    """
    Сохраняет параметр нейросети в текстовый файл
    Args:
        parameter_name: сохраняемый параметр нейросети
        parameters: переменная из которой сохраняется параметр
    Returns:
        Создает файл "parameter parameter_name.txt" с параметром
    """
    with open("parameter %s.txt" % parameter_name, "w") as file:
        A = parameters[parameter_name]
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                file.write(str(A[i][j]) + ' ')
            file.write('\n')


def save_parameters(parameters):
    """
    Сохраняет все параметры нейросети каждый в свой текстовый файл
    """
    save_parameter("W1", parameters)
    save_parameter("W2", parameters)
    save_parameter("W3", parameters)
    #save_parameter("W4", parameters)
    #save_parameter("W5", parameters)
    save_parameter("b1", parameters)
    save_parameter("b2", parameters)
    save_parameter("b3", parameters)
    #save_parameter("b4", parameters)
    #save_parameter("b5", parameters)


def get_parameter(parameter_name):
    """
    Получает из текстового файла "parameter parameter_name.txt"
    параметер нейросети parameter_name
    Args:
        parameter_name: получаемый из текстового файла параметер
    Returns:
        parameter: полученный из текстового файла параметр нейросети
    """
    with open('parameter %s.txt' % parameter_name, "r") as file:
        count = 0
        for line in file:
            line = line.split(' ')
            line = line[:len(line) - 1]
            for i in range(len(line)):
                line[i] = float(line[i])
            line = np.array(line)
            if count == 0:
                len_line = len(line)
                data = np.array(line).reshape((1, len_line))
                count = 1
            else:
                data = np.concatenate(
                    (data, line.reshape((1, len_line))))
    parameter = data
    return parameter


def get_parameters():
    """
    Получает параметры нейросети каждый из своего текстового файла
    Returns:
        parameters: словарь из параметров нейросети
    """
    W1 = get_parameter("W1")
    W2 = get_parameter("W2")
    W3 = get_parameter("W3")
    #W4 = get_parameter("W4")
    #W5 = get_parameter("W5")
    b1 = get_parameter("b1")
    b2 = get_parameter("b2")
    b3 = get_parameter("b3")
    #b4 = get_parameter("b4")
    #b5 = get_parameter("b5")
    parameters = {"W1": W1, "W2": W2, "W3": W3, #"W4": W4, #"W5": W5,
                  "b1": b1, "b2": b2, "b3": b3, #"b4": b4, #"b5": b5
                  }
    return parameters


constant_train = 72862
train_x = [[0] * constant_train, [0] * constant_train]
train_y = [[0] * constant_train]
constant_test = 8096
test_x = [[0] * constant_test, [0] * constant_test]
test_y = [[0] * constant_test]
FILENAME = "Training examples.csv"
with open(FILENAME, "r", newline="") as file:
    reader = csv.reader(file)
    i = 0
    k = 0
    for row in reader:
        if i % 10 == 0 and i < 80958:
            test_x[0][int(i / 10)] = float(row[0])
            test_x[1][int(i / 10)] = float(row[1])
            test_y[0][int(i / 10)] = float(row[2])
            i += 1
            k += 1
        elif i < 80958:
            train_x[0][i - k] = float(row[0])
            train_x[1][i - k] = float(row[1])
            train_y[0][i - k] = float(row[2])
            i += 1
# Normalization of training data
mu0 = 0
mu1 = 0
sigma0 = 0
sigma1 = 0
# Normalization of test data
"""
for i in range(constant_train):
    mu0 += train_x[0][i]
    mu1 += train_x[1][i]
    sigma0 = train_x[0][i] ** 2
    sigma1 = train_x[1][i] ** 2
mu0 = mu0 / constant_train
mu1 = mu1 / constant_train
sigma0 /= constant_train
sigma1 /= constant_train
for i in range(constant_train):
    train_x[0][i] -= mu0
    train_x[1][i] -= mu1
    train_x[0][i] = train_x[0][i] / sigma0 ** 0.5
    train_x[1][i] = train_x[1][i] / sigma1 ** 0.5
for i in range(constant_test):
    test_x[0][i] -= mu0
    test_x[1][i] -= mu1
    test_x[0][i] = test_x[0][i] / sigma0 ** 0.5
    test_x[1][i] = test_x[1][i] / sigma1 ** 0.5
"""
test_x = np.array(test_x)
test_y = np.array(test_y)

train_x = np.array(train_x)
train_y = np.array(train_y)
# Setting the number of neurons in each layer

"""
layers_dims = [2, 10, 8, 6, 1]
parameters = L_layer_model(train_x, train_y, layers_dims,
                           num_iterations=10000, learning_rate=2,
                           print_cost=True, lambd=0)
"""
layers_dims = [2, 8, 6, 1]
parameters = get_parameters()
param = parameters
parameters = L_layer_model(train_x, train_y, layers_dims,
                           num_iterations=5000, learning_rate=0.5,
                           print_cost=True, lambd=0, rate="continue")

print("Training examples:")
predict_train = predict(train_x, train_y, parameters)
# Test
print("Test examples:")
predict_test = predict(test_x, test_y, parameters)
"""
real_zeros = []
real_ones = []
image_zeros = []
image_ones = []
for i in range(constant_train):
    if train_y[0][i] == 0:
        real_zeros.append(train_x[0][i])
        image_zeros.append(train_x[1][i])
    else:
        real_ones.append(train_x[0][i])
        image_ones.append(train_x[1][i])
plt.plot(real_zeros, image_zeros, 'ro')
plt.show()
plt.plot(real_ones, image_ones, 'go')
plt.show()
plt.plot(real_zeros, image_zeros, 'ro')
plt.plot(real_ones, image_ones, 'go')
plt.show()
"""
save_parameters(parameters)
"""
parameters = get_parameters()
print("Test examples2:")
predict_test = predict(test_x, test_y, parameters)
"""
