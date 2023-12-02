import sys
sys.path.append(r"C:\Users\samue\projects\Neural-Network-From-Scratch")

from DataLoader import FashionDataLoader
import matplotlib.pyplot as plt

human_labels = {
    0 : "Tshirt/top",
    1 : "Trouser",
    2 : "Pullover",
    3 : "Dress",
    4 : "Coat",
    5 : "Sandal",
    6 : "Shirt",
    7 : "Sneaker",
    8 : "Bag",
    9 : "Ankle boot",
}

# Displays images using matplotlib and prits label in console
def test_image_label_pairing():
    dataloader = FashionDataLoader("data", (28, 28))

    for i in range(len(dataloader)):
        image, label = dataloader[i]

        print(human_labels[label])

        plt.imshow(image)
        plt.show()

if __name__ == "__main__":
    test_image_label_pairing()