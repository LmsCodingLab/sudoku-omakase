import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Preparing the data for training or in other words warming up ;)
def warmup(batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
  """
  Prepare the SVHN datasets for training by applying necessary transformations and creating DataLoader.

  Parameters:
  - batch_size: int, the number of samples per batch to load.

  Returns:
  - tuple[DataLoader, DataLoader]: A tuple containing the training and test DataLoaders for the SVHN dataset.
  """


  svhn_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
  ])

  svhn_train = datasets.SVHN(root="./model_training/datasets", split="train", download=True, transform=svhn_transform)
  svhn_test = datasets.SVHN(root="./model_training/datasets", split="test", download=True, transform=svhn_transform)


  use_cuda = torch.cuda.is_available()
  data_loader = DataLoader(
    dataset=svhn_train,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=use_cuda
  )
  test_loader = DataLoader(
    dataset=svhn_test,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=use_cuda
  )

  return data_loader, test_loader

