import torch
from torch import nn
from torchvision import models
from enum import Enum

class ModelType(Enum):
  BAD = "basic"
  NORMAL = "resnet"
  BIG = "resnext"

class BasicCNNModel(nn.Module):
  """
  A basic Convolutional Neural Network (CNN) model for image classification tasks.
  This model consists of convolutional layers followed by fully connected layers for classification.

  Replicating this architecture (with small tweaks): https://poloclub.github.io/cnn-explainer/

  Parameters:
  - input_shape: int, the number of channels in the input images (e.g., 1 for grayscale, 3 for RGB).
  - output_shape: int, the number of classes for the output layer.
  - image_size: int, the height and width of the input images (default is 32, assuming square images).
  """
  def __init__(self, input_shape: int, output_shape: int, image_size: int = 32):
    super().__init__()
    if image_size % 4 != 0:
      raise ValueError("Image size must be divisible by 4 to ensure proper downsampling through the convolutional layers.")
    hidden_units = 10
    self.conv_block1 = nn.Sequential(
      nn.Conv2d(in_channels=input_shape, 
                out_channels=hidden_units, 
                kernel_size=3, 
                stride=1, 
                padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=hidden_units, 
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),
    )
    self.conv_block2 = nn.Sequential(
      nn.Conv2d(in_channels=hidden_units,
                out_channels=hidden_units*4,
                kernel_size=3,
                stride=1,
                padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=hidden_units*4,
                out_channels=hidden_units*4,
                kernel_size=3,
                stride=1,
                padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2)
    )
    final_size = image_size // 4 # after two max pooling layers with kernel size 2, the image size is reduced by a factor of 4
    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Dropout(p=0.33),
      nn.Linear(in_features=hidden_units*4*final_size*final_size, out_features=output_shape)
    )
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Passes the input through the CNN layers and then through the classifier to produce the output.

    Parameters:
    - x: torch.Tensor, the input tensor representing a batch of images.

    Returns:
    - torch.Tensor, the output tensor representing the class scores for each input image.
    """
    x = self.conv_block1(x)
    x = self.conv_block2(x)
    x = self.classifier(x)
    return x

class ResNet18_32(nn.Module):
  """
  A ResNet-18 architecture adapted for 32x32 input images.
  This model consists of residual blocks that allow for deeper networks without the vanishing gradient problem.

  Parameters:
  - input_shape: int, the number of channels in the input images (e.g., 1 for grayscale, 3 for RGB).
  - output_shape: int, the number of classes for the output layer.
  """
  def __init__(self, input_shape: int, output_shape: int):
    super(ResNet18_32, self).__init__()
    self.model = models.resnet18(weights=None)
    self.model.conv1 = nn.Conv2d(in_channels=input_shape, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    self.model.maxpool = nn.Identity()  # type: ignore[assignment] # Remove the max pooling layer to preserve spatial dimensions for 32x32 input
    self.model.fc = nn.Sequential(
      nn.Dropout(p=0.2),
      nn.Linear(in_features=512, out_features=output_shape)
    )
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.model(x)
  
class ResNeXt_101(nn.Module):
  """
  A ResNeXt-101 architecture adapted for 32x32 input images.
  This model consists of grouped convolutions that allow for more efficient learning of features.

  Parameters:
  - input_shape: int, the number of channels in the input images (e.g., 1 for grayscale, 3 for RGB).
  - output_shape: int, the number of classes for the output layer.
  """
  def __init__(self, input_shape: int, output_shape: int):
    super(ResNeXt_101, self).__init__()
    self.model = models.resnext101_32x8d(weights=None)
    self.model.conv1 = nn.Conv2d(in_channels=input_shape, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    self.model.maxpool = nn.Identity()  # type: ignore[assignment] # Remove the max pooling layer to preserve spatial dimensions for 32x32 input
    self.model.fc = nn.Sequential(
      nn.Dropout(p=0.3),
      nn.Linear(in_features=2048, out_features=output_shape)
    )
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.model(x)