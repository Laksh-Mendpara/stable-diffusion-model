from torchvision.datasets import MNIST
from torchvision.transforms import transforms, ToTensor
from torch.utils.data import DataLoader
import numpy as np


def get_data_loader(batch_size: int) -> DataLoader:

    transformation = transforms.Compose([
        transforms.Resize([28, 28]),
        ToTensor()
    ])

    dataset = MNIST(root="data/", download=True, train=True, transform=transformation)
    
    data_loader = DataLoader(
        dataset=dataset,
        drop_last=True,
        batch_size=batch_size, 
        shuffle=True)
    
    return data_loader


if __name__=="__main__":
    dataloader = get_data_loader(64)
    for idx, (img, label) in enumerate(dataloader):
        print(img.shape)
