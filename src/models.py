import torch
from torch import nn

# TODO
class BasicCNNModel(nn.Module):
  """
  A basic Convolutional Neural Network (CNN) model for image classification tasks.
  This model consists of convolutional layers followed by fully connected layers for classification.

  Replecating this architecture: https://poloclub.github.io/cnn-explainer/
  """
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
    super(BasicCNNModel, self).__init__()
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
      nn.MaxPool2d(kernel_size=2)
    )
    self.classifer = nn.Sequential(
      nn.Flatten(),
      nn.Linear(in_features=hidden_units*0, out_features=output_shape) # replace 0 with actual value
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
    print(x.shape)
    x = self.conv_block2(x)
    print(x.shape)
    x = self.classifer(x)
    return x
  

if __name__ == "__main__":
  torch.manual_seed(42)
  model = BasicCNNModel(input_shape=1, hidden_units=10, output_shape=10)
  print(model)
        