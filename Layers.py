import numpy as np

class LinearLayer:
    def __init__(self, in_features, out_features):
        self.weights = np.random.randn(in_features, out_features).transpose()
        print(self.weights.shape)
        self.biases = np.random.randn(out_features)
        self.weight_grad = None
        self.bias_grad = None
        self.activations_gradient = None
        self.activations = None

    def forward(self, x):
        x = np.matmul(self.weights, x)
        x = x + self.biases
        self.activations = x
        return x

    def backward(self, grad):
        self.bias_grad = grad

        self.weight_grad = np.ones(self.weights.shape)
        self.weight_grad = self.weight_grad * grad[:, np.newaxis]
        self.weight_grad = self.weight_grad * self.activations[:, np.newaxis]

        self.activations_gradient = np.matmul(self.weights.transpose(), grad)

        return self.activations_gradient

class Softmax:
    def __init__(self):
        self.grad = None
        self.activations = None

    def forward(self, x):
        exp_matrix = np.exp(x)
        self.activations = exp_matrix / np.sum(exp_matrix)

        return self.activations

    def backward(self, grad):
        self.grad = np.matmul(self.__jacobian(self.activations), grad)

        return self.grad

    def __jacobian(self, x):
        return np.diag(x) - np.outer(x, x) 
    
    def jacobian(self, x):
        return self.__jacobian(x)

class MSELoss:
    def __init__(self):
        self.grad = None
        self.loss = 0

    def forward(self, x, expected):
        loss = x - expected
        loss = np.square(loss)
        self.loss = np.mean(loss)

        return self.loss
    
    def backward(self, x, expected):
        self.grad = (2 / len(expected)) * (x - expected)

        return self.grad

class CrossEntropyLoss:
    def __init__(self):
        self.grad = None
        self.loss = 0
        self.activations = None

    def forward(self, x, expected):
        self.activations = np.copy(x)

        log_x = np.log(x)
        self.loss = -np.sum(log_x * expected)

        return self.loss

    def backward(self, expected):
        self.grad = -np.divide(expected, self.activations)

        return self.grad

class Relu:
    def __init__(self):
        self.grad = None
        self.activations = None

    def forward(self, x):
        relu_func = np.vectorize(lambda x: 0 if x < 0 else x)

        self.activations = relu_func(x)

        return self.activations
    
    def backward(self, grad):
        relu_derivative_func = np.vectorize(lambda x: 0 if x < 0 else 1)

        self.grad = relu_derivative_func(self.activations) * grad

        return self.grad
