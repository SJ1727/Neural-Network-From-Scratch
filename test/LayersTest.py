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

    y_true = np.array([1.0, 0.0, 0.0, 1.0])
    y_pred = np.array([0.8, 0.5, 0.6, 0.4])

    print(crit.forward(y_pred, y_true))

def test_softmax_backward():
    print("\nSoftmax Backward\n")

    predicted1 = np.random.randn(5)
    actual1 = np.array([0, 0, 0, 0, 1])

    crit = CrossEntropyLoss()
    softm = Softmax()

    predicted1 = softm.forward(predicted1)

    crit.backward(predicted1, actual1)
    softm.backward(predicted1, crit.grad)

    # When used with cross entropy loss the gradient of the input values of the softmax function
    # are equal to the predicted result (the inputs after softmax is applied) minus the actual
    # values
    assert np.allclose(softm.grad, predicted1 - actual1)

if __name__ == "__main__":
    #test_softmax_forward()
    test_softmax_backward()
    #test_cross_entropy_forward()
