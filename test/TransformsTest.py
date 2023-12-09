import sys
sys.path.append(r"C:\Users\samue\projects\Neural-Network-From-Scratch")

from transforms import normilize
import numpy as np

def test_normilization():
    arr = np.random.randn(10)    

    assert np.isclose(np.mean(normilize(arr)), 0)
    assert np.isclose(np.std(normilize(arr)), 1)


if __name__ == "__main__":
    test_normilization()

