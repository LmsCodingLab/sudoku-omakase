import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.sudoku_omakase.helpers.dev_info import dev_show_message

# Preparing the data for training or in other words warming up ;)
def warmup(batch_size: int = 64, dev_mode: bool = False) -> tuple[DataLoader, DataLoader]:
  """
  Prepare the SVHN datasets for training by applying necessary transformations and creating DataLoader.

  Parameters:
  - batch_size: int, the number of samples per batch to load.
  - dev_mode: bool, whether to show development messages

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

  dev_show_message(dev_mode, f"Data loaders created with {len(svhn_train)} training samples and {len(svhn_test)} test samples.")

  return data_loader, test_loader

