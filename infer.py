import torch
import torchvision.models as models
import classNN
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


model = torch.load('model.pth')

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

infer_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
random_sample = next(iter(infer_dataloader))
#print (random_sample)

with torch.no_grad():
    X, y = next(iter(infer_dataloader))
    X = X[0]
    y = y[0]
    output = model(X)
    pred = torch.argmax(output, dim=1).item()
    print(f'pred {pred}')

    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }

    print(f'prediction is {labels_map[pred]}')

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 1, 1
    for i in range(1, cols * rows + 1):
        img, label = X, y
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[pred])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()
