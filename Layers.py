import numpy as np

class Layer:
    LAYER_NAME = "Layer"

    def __init__(self):
        self.activations = None
        self.activations_gradient = None

    def activation_statistic(self):
        """
        Displays the mean and standard deviation of the activations of the layer
        """
        print(f"\n---{self.__class__.LAYER_NAME}---\n")
        print(f"Mean: {np.mean(self.activations)}")
        print(f"Standard deviation: {np.std(self.activations)}")

class LinearLayer(Layer):
    LAYER_NAME = "Linear Layer"

    def __init__(self, in_features, out_features):
        """
        Initilise Linear layer

        Args:
            in_features (int): Number of in features
            out_features (int): Number of out features
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.random.uniform(-1, 1, (in_features, out_features)).transpose()
        self.biases = np.random.uniform(-1, 1, (out_features))
        self.weight_gradient = None
        self.bias_gradient = None

    def forward(self, x):
        """
        Forward propogation for linear layer

        Args:
            x (np.array): activations of previous layer or the input

        Returns:
            np.array: activations of this layer
        """
        self.activations = x
        self.activations = np.matmul(self.weights, self.activations)
        self.activations = self.activations + self.biases

        return self.activations

    def backward(self, gradient):
        """
        Backward propgation for linear layer

        Args:
            gradient (np.array): gradients of layer infront

        Returns:
            np.array: gradients of the activations for this layer
        """
        self.bias_gradient = gradient

        self.weight_gradient = np.ones(self.weights.shape)
        self.weight_gradient = self.weight_gradient * gradient[:, np.newaxis]
        self.weight_gradient = self.weight_gradient * self.activations[:, np.newaxis]

        self.activations_gradient = np.matmul(self.weights.transpose(), gradient)

        return self.activations_gradient
    
    # TODO: Add option for other types of optimization (Adam, RMSprop, etc)
    def optimize(self, lr):
        """
        Updates weights and biases using stohcastic gradient decent 

        Args:
            lr (float): Learning rate
        """
        self.biases = self.biases - lr * self.bias_gradient
        self.weights = self.weights - lr * self.weight_gradient

    def kaiming_init(self):
        """
        Uses kaiming initilisation to set the weights and biases
        """
        self.weights = np.random.uniform(-1, 1, (self.in_features, self.out_features)).transpose() * np.sqrt(2 / self.in_features)
        self.biases = np.random.uniform(-1, 1, (self.out_features)) * np.sqrt(2 / self.in_features)


class Softmax(Layer):
    LAYER_NAME = "Softmax"

    def __init__(self):
        """
        Initilise softmax layer
        """
        super().__init__()

    def forward(self, x):
        """
        Forward propogation for softmax layer

        Args:
            x (np.array): activations of previous layer

        Returns:
            np.array: activations of this layer
        """
        exp_matrix = np.exp(x)
        self.activations = exp_matrix / np.sum(exp_matrix)

        return self.activations

    def backward(self, gradient):
        """
        Backward propogation for softmax layer

        Args:
            gradient (np.array): gradients of layer infront

        Returns:
            np.array: gradients of activations for this layer
        """
        self.activations_gradient = np.matmul(self.__jacobian(self.activations), gradient)

        return self.activations_gradient

    def __jacobian(self, x):
        """
        Calculates jacobian for input array

        Args:
            x (np.array): array used to calculate jacobian

        Returns:
            np.array: jacobian of array
        """
        return np.diag(x) - np.outer(x, x) 

class Relu(Layer):
    LAYER_NAME = "Relu"

    def __init__(self):
        """
        Initilastion for relu layer
        """
        super().__init__()

    def forward(self, x):
        """
        Forward propogation for relu layer

        Args:
            x (np.array): activations of previous layer

        Returns:
            np.array: activations of this layer
        """
        relu_func = np.vectorize(lambda x: 0 if x < 0 else x)

        self.activations = relu_func(x)

        return self.activations
    
    def backward(self, gradient):
        """
        Backward propogation for relu layer

        Args:
            gradient (np.array): gradients of layer infront

        Returns:
            np.array: gradients of activations for this layer
        """
        relu_derivative_func = np.vectorize(lambda x: 0 if x < 0 else 1)

        self.activations_gradient = relu_derivative_func(self.activations) * gradient

        return self.activations_gradient

class MSELoss(Layer):
    LAYER_NAME = "MSE Loss"

    def __init__(self):
        """
        Initilisation for MSE loss
        """
        super().__init__()
        self.loss = 0

    def forward(self, x, expected):
        """
        Forward propogation for MSE loss

        Args:
            x (np.array): activations of previous layer
            expected (np.array): expected values for data

        Returns:
            float: calculated MSE loss 
        """
        loss = x - expected
        loss = np.square(loss)
        self.loss = np.mean(loss)

        return self.loss
    
    def backward(self, expected):
        """
        Backward propogation for MSE loss

        Args:
            expected (np.array): expected values for data

        Returns:
            np.array: gradients of activations for this layer 
        """
        self.activations_gradient = (2 / len(expected)) * (self.activations - expected)

        return self.activations_gradient

class CrossEntropyLoss(Layer):
    LAYER_NAME = "Cross Entropy Loss"

    def __init__(self):
        """
        Initilisation for Cross entropy loss
        """
        super().__init__()
        self.loss = 0

    def forward(self, x, expected):
        """
        Forward propogation for Cross entropy loss

        Args:
            x (np.array): activations of previous layer
            expected (np.array): expected values for data

        Returns:
            np.array: calculated Cross entropy loss
        """
        self.activations = np.copy(x)

        log_x = np.log(x)
        self.loss = -np.sum(log_x * expected)

        return self.loss

    def backward(self, expected):
        """
        Backward propogation for Cross entropy loss

        Args:
            expected (np.array): expected values for data

        Returns:
            np.array: gradients of activations for thus layer
        """
        self.activations_gradient = -np.divide(expected, self.activations)

        return self.activations_gradient
