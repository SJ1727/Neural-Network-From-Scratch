import numpy as np
import os
import gzip

class FashionDataLoader:
    def __init__(self, dir_path):

        labels_path = os.path.join(dir_path, "train-labels-idx1-ubyte.gz")

        images_path = os.path.join(dir_path, "train-images-idx3-ubyte.gz")

        with gzip.open(labels_path, 'rb') as lbpath:
            self.labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            self.images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(self.labels), 784)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.labels)