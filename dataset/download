from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def get_data_loader() -> DataLoader:
    # Download training data from open datasets.
    dataset = MNIST(root="data/", download=True, transform=ToTensor())

    # Create a data loader from the dataset
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    return data_loader
