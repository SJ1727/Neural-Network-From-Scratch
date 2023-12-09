import sys
sys.path.append(r"C:\Users\samue\projects\Neural-Network-From-Scratch")

from layers import *
import numpy as np
import math
from dataLoader import FashionDataLoader

def test_softmax_forward():
    print("\n---Softmax Forward---\n")

    softm = Softmax()
    
    test1 = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0]
    ])
    
    resum1 = math.exp(1) + math.exp(2) + math.exp(3)
    resum2 = math.exp(2) + math.exp(3) + math.exp(4)
    resum3 = math.exp(3) + math.exp(4) + math.exp(5)
    
    expected = np.array([
        [math.exp(1.0) / resum1, math.exp(2.0) / resum1, math.exp(3.0) / resum1],
        [math.exp(2.0) / resum2, math.exp(3.0) / resum2, math.exp(4.0) / resum2],
        [math.exp(3.0) / resum3, math.exp(4.0) / resum3, math.exp(5.0) / resum3],
    ])

    # test1 = np.array([1, 2, 3])
    # test2 = np.array([[1, 2, 3], [4, 5, 6]])

    # resum1 = math.exp(1) + math.exp(2) + math.exp(3)
    # result1 = np.array([math.exp(1) / resum1, math.exp(2) / resum1, math.exp(3) / resum1])

    # resum2 = math.exp(1) + math.exp(2) + math.exp(3) + math.exp(4) + math.exp(5) + math.exp(6)
    # result2 = np.array([[math.exp(1) / resum2, math.exp(2) / resum2, math.exp(3) / resum2], [math.exp(4) / resum2, math.exp(5) / resum2, math.exp(6) / resum2]])

    # assert np.array_equal(softm.forward(test1), result1)
    # assert np.array_equal(softm.forward(test2), result2)
    
    # print(softm.forward(test1))
    # print(expected)
    
    assert np.allclose(softm.forward(test1), expected)

def test_cross_entropy_forward():
    print("\n---Cross Entropy Forward---\n")

    crit = CrossEntropyLoss()

    expected = np.array([1.1394342831883648, 1.1394342831883648])

    y_true = np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]])
    y_pred = np.array([[0.8, 0.5, 0.6, 0.4], [0.8, 0.5, 0.6, 0.4]])

    assert np.allclose(crit.forward(y_pred, y_true), expected)

def test_softmax_backward():
    print("\n---Softmax Backward---\n")

    x = np.random.randn(2, 5)
    actual1 = np.array([[0, 0, 0, 0, 1], [0, 0, 1, 0, 0]])

    crit = CrossEntropyLoss()
    softm = Softmax()

    x = softm.forward(x)
    crit.forward(softm.activations, actual1)

    assert np.allclose(crit.activations, softm.activations)
    assert np.allclose(x, softm.activations)

    grad1 = crit.backward(actual1)
    grad2 = softm.backward(grad1)

    # When used with cross entropy loss the gradient of the input values of the softmax function
    # are equal to the predicted result (the inputs after softmax is applied) minus the actual
    # values
    assert np.allclose(grad2, x - actual1)

def test_linear_layer_activations():
    print("\n---Linear Layer---\n")
    l1 = LinearLayer(10, 3, 1)
    softm = Softmax()
    crit = CrossEntropyLoss()

    data = np.random.randn(1, 10)
    expected = np.array([0, 0, 1])
    x = data
    
    for _ in range(10):

        x = l1.forward(x)
        x = softm.forward(x)
        x = crit.forward(x, expected)

        g1 = crit.backward(expected)
        g2 = softm.backward(g1)
        g3 = l1.backward(g2)

        print(x)

def test_relu_forward():
    print("\n---Relu forward---\n")
    test1 = np.array([[1, -1, 2, 0, -1], [1, -1, 2, 0, -1]])
    expected1 = np.array([[1, 0, 2, 0, 0], [1, 0, 2, 0, 0]])

    test2 = np.array([[1, 2], [-1, 0]])
    expected2 = np.array([[1, 2], [0, 0]])

    relu = Relu()

    result1 = relu.forward(test1)
    result2 = relu.forward(test2)

    assert np.allclose(result1, expected1)
    assert np.allclose(result2, expected2)

def test_relu_backward():
    print("\n---Relu backward---\n")

    activations1 = np.array([[3, -7, -1, 3], [3, -7, -1, 3]]) 
    gradients1 = np.array([[1, 2, -3, 4], [1, 2, -3, 4]])
    expected_gradients1 = np.array([[1, 0, 0, 4], [1, 0, 0, 4]])

    relu = Relu()
    relu.activations = activations1
    new_gradients1 = relu.backward(gradients1)

    assert np.allclose(new_gradients1, expected_gradients1)

def test_example_network():
    lr = 0.05
    epochs = 20

    # Defining layers
    l1 = LinearLayer(784, 16, 5)
    r1 = Relu()
    l2 = LinearLayer(16, 16, 5)
    r2 = Relu()
    l3 = LinearLayer(16, 16, 5)
    sm = Softmax()
    r3 = Relu()
    l4 = LinearLayer(16, 10, 5)

    # Initilising linear layeer susnign kaiming initionlisation
    l1.kaiming_init()
    l2.kaiming_init()
    l3.kaiming_init()
    l4.kaiming_init()

    crit = CrossEntropyLoss()

    # Create random data
    data = np.random.uniform(-1, 1, (5, 784))
    expected = np.zeros((5, 10))
    expected[0,0] = 1
    expected[1,0] = 1
    expected[2,0] = 1
    expected[3,0] = 1
    expected[4,0] = 1
    
    dataloader = FashionDataLoader("data", (784,), 10, batch_size=16, norm=True)
    
    data = np.array([dataloader[i][0] for i in range(5)])
    expected = np.array([dataloader[i][1] for i in range(5)])

    for epoch in range(epochs):
        x = np.copy(data)

        # Forward pass
        x = l1.forward(x)
        # l1.print_weight_statistics()
        x = r1.forward(x)
        x = l2.forward(x)
        x = r2.forward(x)
        x = l3.forward(x)
        x = r3.forward(x)
        x = l4.forward(x)
        x = sm.forward(x)
        loss = crit.forward(x, expected)
        crit.print_activation_statistics()

        if epoch % 1 == 0:
            print(f"Epoch {epoch}:")
            print(f"Loss: {np.mean(loss)}")
            # print(f"Expected: {expected}")
            # print(f"Prediction: {x}")

        # Backward pass
        crit.backward(expected)
        sm.backward(crit.activations_gradient)
        l4.backward(sm.activations_gradient)
        r3.backward(l4.activations_gradient)
        l3.backward(r3.activations_gradient)
        r2.backward(l3.activations_gradient)
        l2.backward(r2.activations_gradient)
        r1.backward(l2.activations_gradient)
        l1.backward(r1.activations_gradient)

        # Update weights and biases
        l1.optimize(lr)
        l2.optimize(lr)
        l3.optimize(lr)
        l4.optimize(lr)

def test_linear_layer():
    l1 = LinearLayer(16, 16, 10)
    
    l1.kaiming_init()
    
    l1.print_weight_statistics()
    l1.print_bias_statistics()

def test_linear_optimize():
    l1 = LinearLayer(10, 10, 1)
    l1.weight_gradient = 0
    
if __name__ == "__main__":
    # test_cross_entropy_forward()
    # test_softmax_forward()
    # test_softmax_backward()
    # test_relu_forward()
    # test_relu_backward()
    # test_linear_layer_activations()
    test_example_network()
    
    pass