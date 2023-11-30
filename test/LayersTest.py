import sys
sys.path.append(r"C:\Users\samue\projects\Neural-Network-From-Scratch")

from Layers import *
import numpy as np
import math

def test_softmax_forward():
    print("\n---Softmax Forward---\n")

    softm = Softmax()

    test1 = np.array([1, 2, 3])
    test2 = np.array([[1, 2, 3], [4, 5, 6]])

    resum1 = math.exp(1) + math.exp(2) + math.exp(3)
    result1 = np.array([math.exp(1) / resum1, math.exp(2) / resum1, math.exp(3) / resum1])

    resum2 = math.exp(1) + math.exp(2) + math.exp(3) + math.exp(4) + math.exp(5) + math.exp(6)
    result2 = np.array([[math.exp(1) / resum2, math.exp(2) / resum2, math.exp(3) / resum2], [math.exp(4) / resum2, math.exp(5) / resum2, math.exp(6) / resum2]])

    assert np.array_equal(softm.forward(test1), result1)
    assert np.array_equal(softm.forward(test2), result2)

def test_cross_entropy_forward():
    print("\n---Cross Entropy Forward---\n")

    crit = CrossEntropyLoss()

    expected = 1.1394342831883648

    y_true = np.array([1.0, 0.0, 0.0, 1.0])
    y_pred = np.array([0.8, 0.5, 0.6, 0.4])

    assert math.isclose(crit.forward(y_pred, y_true), expected)

def test_softmax_backward():
    print("\n---Softmax Backward---\n")

    x = np.random.randn(5)
    actual1 = np.array([0, 0, 0, 0, 1])

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

def test_linear_layer():
    print("\n---Linear Layer---\n")
    l1 = LinearLayer(10, 3)
    softm = Softmax()
    crit = CrossEntropyLoss()

    data = np.random.randn(10)
    expected = np.array([0, 0, 1])
    x = data

    x = l1.forward(x)
    x = softm.forward(x)
    x = crit.forward(x, expected)

    g1 = crit.backward(expected)
    g2 = softm.backward(g1)
    g3 = l1.backward(g2)

    # print(l1.weight_grad)

def test_relu_forward():
    print("\n---Relu forward---\n")
    test1 = np.array([1, -1, 2, 0, -1])
    expected1 = np.array([1, 0, 2, 0, 0])

    test2 = np.array([[1, 2], [-1, 0]])
    expected2 = np.array([[1, 2], [0, 0]])

    relu = Relu()

    result1 = relu.forward(test1)
    result2 = relu.forward(test2)

    assert np.allclose(result1, expected1)
    assert np.allclose(result2, expected2)

def test_relu_backward():
    print("\n---Relu backward---\n")

    activations1 = np.array([3, -7, -1, 3]) 
    gradients1 = np.array([1, 2, -3, 4])
    expected_gradients1 = np.array([1, 0, 0, 4])

    relu = Relu()
    relu.activations = activations1
    new_gradients1 = relu.backward(gradients1)

    assert np.allclose(new_gradients1, expected_gradients1)

if __name__ == "__main__":
    test_cross_entropy_forward()
    test_softmax_forward()
    test_softmax_backward()
    test_relu_forward()
    test_relu_backward()
    test_linear_layer()

    # s = Softmax()
    # t = np.array([1, 2, 3])
    # g = np.array([1, 3, 4])

    # j = s.jacobian(t)
    # print(j)
    # print(np.matmul(j, g))