import numpy as np

class LinearLayer:
    def __init__(self, in_features, out_features):
        self.weights = np.random.rand(in_features, out_features)
        self.biases = np.random.randn(out_features)
        self.weight_grad = None
        self.bias_grad = None

    def forward(self, x):
        x = np.matmul(self.weights, x)
        x = x + self.biases

class Softmax:
    def __init__(self):
        self.grad = None

    def forward(self, x):
        exp_matrix = np.exp(x)

        return exp_matrix / np.sum(exp_matrix)

    def backward(self, x, grad):
        self.grad = np.matmul(self.__jacobian(x), grad)

    def __jacobian(self, x):
        x_new_axis = x[:, np.newaxis]
        return -(np.ones((x.shape[0], x.shape[0])) * x_new_axis * x_new_axis.transpose() - np.diag(x))

class MSELoss:
    def __init__(self):
        self.grad = None

    def forward(self, x, expected):
        loss = x - expected
        loss = np.square(loss)

        return np.mean(loss)
    
    def backward(self, x, expected):
        self.grad = (2 / len(expected)) * (x - expected)

class CrossEntropyLoss:
    def __init__(self):
        self.grad = None

    def forward(self, x, expected):
        log_x = np.log(x)

        return -np.sum(log_x * expected)

    def backward(self, x, expected):
        self.grad = -np.divide(expected, x)


