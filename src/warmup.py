from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset


# Preparing the data for training or in other words warming up ;)
def warmup() -> DataLoader:
  """
  Prepare the EMNIST and SVHN datasets for training by applying necessary transformations and creating DataLoader.

  Parameters:
  - None

  Returns:
  - DataLoader: A DataLoader for the combined EMNIST and SVHN datasets.
  """

  # Define transofrmation of datasets to be compatible with the model and each other (grayscale, 32x32, normalized)
  emnist_transform = transforms.Compose([
    transforms.Resize(size=(32, 32)),
    transforms.RandomInvert(p=1),
    transforms.Lambda(lambda x: x.transpose(1, 2)),
    transforms.ToTensor(),
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

  # Combine the datasets and create a DataLoader
  combined_dataset = ConcatDataset(datasets=[emnist_train, svhn_train])
  data_loader = DataLoader(dataset=combined_dataset, batch_size=64, shuffle=True)

  return data_loader

if __name__ == "__main__":
  data_loader = warmup()
  print(f"Number of batches in combined dataset: {len(data_loader)}")

