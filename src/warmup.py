from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from src.helpers.dev_info import dev_show_message

# Preparing the data for training or in other words warming up ;)
def warmup(dev_mode: bool = False) -> tuple[DataLoader, DataLoader]:
  """
  Prepare the SVHN datasets for training by applying necessary transformations and creating DataLoader.

  Parameters:
  - dev_mode: bool, whether to show development messages

  Returns:
  - tuple[DataLoader, DataLoader]: A tuple containing the training and test DataLoaders for the SVHN dataset.
  """

  svhn_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
  ])

  svhn_train = datasets.SVHN(root="./datasets", split="train", download=True, transform=svhn_transform)
  svhn_test = datasets.SVHN(root="./datasets", split="test", download=True, transform=svhn_transform)


  data_loader = DataLoader(dataset=svhn_train, batch_size=64, shuffle=True)
  test_loader = DataLoader(dataset=svhn_test, batch_size=64, shuffle=False)

  dev_show_message(dev_mode, f"Data loaders created with {len(svhn_train)} training samples and {len(svhn_test)} test samples.")

  return data_loader, test_loader

