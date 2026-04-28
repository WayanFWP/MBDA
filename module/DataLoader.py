import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torchvision import datasets, transforms

class DataLoader:
    def __init__(self, path, type="MNIST", batch=32):
        self.path = path
        self.batch_size = batch
        self.train_dat = None
        self.test_dat = None
        
        transform = transforms.ToTensor()

        if type == "EMNIST":
            self.train_dat = datasets.EMNIST(root=path, split="balanced", train=True,  download=True, transform=transform)
            self.test_dat  = datasets.EMNIST(root=path, split="balanced", train=False, download=True, transform=transform)
        elif type == "MNIST":
            self.train_dat = datasets.MNIST(root=path, train=True,  download=True, transform=transform)
            self.test_dat  = datasets.MNIST(root=path, train=False, download=True, transform=transform)
        else:
            raise ValueError(f"Unknown dataset type: {type}")

    def Loader(self):
        train_loader = TorchDataLoader(self.train_dat, batch_size=self.batch_size, shuffle=True)
        test_loader = TorchDataLoader(self.test_dat, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader
    
    """
    Helper function to extract images and its labels from Loader function and flatten into 1D array.
    """
    def extract(self, data):
        images, labels = [], []
        for img, label in data:
            for i in range(img.size(0)):  # Iterate through batch
                images.append(img[i].view(-1))  # Flatten individual image
                labels.append(label[i])
        return torch.stack(images), torch.tensor(labels)