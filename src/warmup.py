import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Preparing the data for training or in other words warming up ;)
def warmup() -> tuple[DataLoader, DataLoader]:
  # transform data from different datasets
  emnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x),
    transforms.Normalize((0.5,), (0.5,))
  ])

  svhn_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
  ])

  # Load the datasets
  emnist_train = datasets.EMNIST(root="./datasets", split="digits", train=True, download=True, transform=emnist_transform)
  svhn_train = datasets.SVHN(root="./datasets", split="train", download=True, transform=svhn_transform)

  emnist_dataLoader = DataLoader(emnist_train, batch_size=64, shuffle=True)
  svhn_dataLoader = DataLoader(svhn_train, batch_size=64, shuffle=True)

  return emnist_dataLoader, svhn_dataLoader

if __name__ == "__main__":
  emnist_loader, svhn_loader = warmup()
  print(f"EMNIST DataLoader: {len(emnist_loader)} batches")
  print(f"SVHN DataLoader: {len(svhn_loader)} batches")