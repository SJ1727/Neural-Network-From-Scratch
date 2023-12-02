import numpy as np

# Base class for network layers
class Layer:
    LAYER_NAME = "Layer"

    def __init__(self):
        self.activations = None
        self.activations_gradient = None

    def activation_statistic(self):
        print(f"\n---{self.__class__.LAYER_NAME}---\n")
        print(f"Mean: {np.mean(self.activations)}")
        print(f"Standard deviation: {np.std(self.activations)}")

class LinearLayer(Layer):
    LAYER_NAME = "Linear Layer"

    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.random.uniform(-1, 1, (in_features, out_features)).transpose()
        self.biases = np.random.uniform(-1, 1, (out_features))
        self.weight_gradient = None
        self.bias_gradient = None

    def forward(self, x):
        x = np.matmul(self.weights, x)
        x = x + self.biases
        self.activations = x
        return x

    def backward(self, gradient):
        self.bias_gradient = gradient

        self.weight_gradient = np.ones(self.weights.shape)
        self.weight_gradient = self.weight_gradient * gradient[:, np.newaxis]
        self.weight_gradient = self.weight_gradient * self.activations[:, np.newaxis]

        self.activations_gradient = np.matmul(self.weights.transpose(), gradient)

        return self.activations_gradient
    
    # Using stohcastic gradient decent
    # TODO: Add option for other types of optimization (Adam, RSMprop, etc)
    def optimize(self, lr):
        self.biases = self.biases - lr * self.bias_gradient
        self.weights = self.weights - lr * self.weight_gradient
    
    # Initializes weights using kaiming algoritm
    # Only really effective when network consists of relu layers
    def kaiming_init(self):
        self.weights = np.random.uniform(-1, 1, (self.in_features, self.out_features)).transpose() * np.sqrt(2 / self.in_features)
        self.biases = np.random.uniform(-1, 1, (self.out_features)) * np.sqrt(2 / self.in_features)


class Softmax(Layer):
    LAYER_NAME = "Softmax"

    def __init__(self):
        super().__init__()

    def forward(self, x):
        exp_matrix = np.exp(x)
        self.activations = exp_matrix / np.sum(exp_matrix)

        return self.activations

    def backward(self, gradient):
        self.activations_gradient = np.matmul(self.__jacobian(self.activations), gradient)

        return self.activations_gradient

    def __jacobian(self, x):
        return np.diag(x) - np.outer(x, x) 

class MSELoss(Layer):
    LAYER_NAME = "MSE Loss"

    def __init__(self):
        super().__init__()
        self.loss = 0

    def forward(self, x, expected):
        loss = x - expected
        loss = np.square(loss)
        self.loss = np.mean(loss)

        return self.loss
    
    def backward(self, x, expected):
        self.activations_gradient = (2 / len(expected)) * (x - expected)

        return self.activations_gradient

class CrossEntropyLoss(Layer):
    LAYER_NAME = "Cross Entropy Loss"

    def __init__(self):
        super().__init__()
        self.loss = 0

    def forward(self, x, expected):
        self.activations = np.copy(x)

        log_x = np.log(x)
        self.loss = -np.sum(log_x * expected)

        return self.loss

    def backward(self, expected):
        self.activations_gradient = -np.divide(expected, self.activations)

        return self.activations_gradient

class Relu(Layer):
    LAYER_NAME = "Relu"

    def __init__(self):
        super().__init__()

    def forward(self, x):
        relu_func = np.vectorize(lambda x: 0 if x < 0 else x)

        self.activations = relu_func(x)

        return self.activations
    
    def backward(self, gradient):
        relu_derivative_func = np.vectorize(lambda x: 0 if x < 0 else 1)

        self.activations_gradient = relu_derivative_func(self.activations) * gradient

        return self.activations_gradient