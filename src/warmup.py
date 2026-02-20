import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Preparing the data for training or in other words warming up ;)
def warmup() -> tuple[DataLoader, DataLoader]:
  # transform data from different datasets
  emnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x)  # Invert colors
  ])

  svhn_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
  ])

  emnist_train = datasets.EMNIST(root="./datasets", split="digits", train=True, download=True, transform=emnist_transform)
  svhn_train = datasets.SVHN(root="./datasets", split="train", download=True, transform=svhn_transform)

  emnist_DataLoader = DataLoader(emnist_train, batch_size=64, shuffle=True)
  svhn_DataLoader = DataLoader(svhn_train, batch_size=64, shuffle=True)
  
  return emnist_DataLoader, svhn_DataLoader