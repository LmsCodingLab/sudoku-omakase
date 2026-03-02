from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset


# Preparing the data for training or in other words warming up ;)
def warmup() -> tuple[DataLoader, DataLoader]:
  """
  Prepare the EMNIST and SVHN datasets for training by applying necessary transformations and creating DataLoader.

  Parameters:
  - None

  Returns:
  - tuple[DataLoader, DataLoader]: A tuple containing the training and test DataLoaders for the combined EMNIST and SVHN datasets.
  """

  # Define transformation of datasets to be compatible with the model and each other (grayscale, 32x32, normalized)
  emnist_transform = transforms.Compose([
    transforms.Resize(size=(32, 32)),
    transforms.RandomInvert(p=1),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.transpose(1, 2)),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
  ])

  svhn_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
  ])

  # Load the datasets
  emnist_train = datasets.EMNIST(root="./datasets", split="digits", train=True, download=True, transform=emnist_transform)
  svhn_train = datasets.SVHN(root="./datasets", split="train", download=True, transform=svhn_transform)

  emnist_test = datasets.EMNIST(root="./datasets", split="digits", train=False, download=True, transform=emnist_transform)
  svhn_test = datasets.SVHN(root="./datasets", split="test", download=True, transform=svhn_transform)

  # Combine the datasets and create a DataLoader
  combined_dataset = ConcatDataset(datasets=[emnist_train, svhn_train])
  combined_test_dataset = ConcatDataset(datasets=[emnist_test, svhn_test])
  data_loader = DataLoader(dataset=combined_dataset, batch_size=64, shuffle=True)
  test_loader = DataLoader(dataset=combined_test_dataset, batch_size=64, shuffle=False)

  return data_loader, test_loader

