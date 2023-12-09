import numpy as np
import os
import gzip
from transforms import normilize

class FashionDataLoader:
    def __init__(self, dir_path, shape, expected_size, norm=False, batch_size=1):
        
        self.batch_size = batch_size

        labels_path = os.path.join(dir_path, "train-labels-idx1-ubyte.gz")

        images_path = os.path.join(dir_path, "train-images-idx3-ubyte.gz")

        with gzip.open(labels_path, 'rb') as lbpath:
            self.labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
            self.one_hot_labels = np.zeros((*self.labels.shape, expected_size))
            
            for i, label in enumerate(self.labels):
                self.one_hot_labels[i, label] = 1

        with gzip.open(images_path, 'rb') as imgpath:
            self.images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(self.labels), *shape)

        if norm:
            self.images = [normilize(image) for image in self.images]

    def __getitem__(self, index):
        return self.images[index], self.one_hot_labels[index]

    def __len__(self):
        return len(self.labels)