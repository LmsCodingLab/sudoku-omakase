import torch
from torch import nn
from timeit import default_timer as timer
from pathlib import Path
from src.training.models import BasicCNNModel, ResNeXt_101, ResNet18_32
from src.training.warmup import warmup
from src.helpers.train_help import accuracy_fn, training_step, testing_step
from src.helpers.dev_info import dev_show_message

def basic_training_loop(model_type: str, dev_mode: bool = False) -> nn.Module:
  """
  A basic training loop for training a CNN model on a dataset. This function initializes the model, sets up the loss function and optimizer, and runs the training and testing steps for a specified number of epochs.

  Parameters:
  - model_type: str, the type of model to train ("basic", "resnet", or "resnext").
  - dev_mode: bool, whether to print development messages and information during training.

  Returns:
  - nn.Module: The trained model after the training loop is completed.
  """
  torch.cuda.empty_cache() # Clear GPU memory before starting training
  if model_type == "basic":
    dev_show_message(dev_mode, "Training BasicCNNModel...")
    model = BasicCNNModel(input_shape=1, output_shape=10)
  elif model_type == "resnet":
    dev_show_message(dev_mode, "Training ResNet18_32...")
    model = ResNet18_32(input_shape=1, output_shape=10)
  elif model_type == "resnext":
    dev_show_message(dev_mode, "Training ResNeXt101...")
    model = ResNeXt_101(input_shape=1, output_shape=10)
  else:    
    raise ValueError(f"Unknown model type: {model_type}. Expected 'basic', 'resnet', or 'resnext'.")

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  dev_show_message(dev_mode, f"Using device: {device}")
  
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
  batch_size = 16 if model_type == "resnext" else 64
  data_loader, test_loader = warmup(batch_size=batch_size, dev_mode=dev_mode)

  train_time_start = timer()
  epochs = 20
  Path("weights").mkdir(parents=True, exist_ok=True)
  for epoch in range(epochs):
    dev_show_message(dev_mode, f"Epoch: {epoch}")
    training_step(model=model,
                  data_loader=data_loader,
                  loss_fn=loss_fn,
                  optimizer=optimizer,
                  accuracy_fn=accuracy_fn,
                  device=device, 
                  dev_mode=dev_mode) 
    
    testing_step(model=model,
              data_loader=test_loader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device,
              dev_mode=dev_mode) 
    torch.save(model.state_dict(), f"weights/{model_type}_model{epoch}.pth")
    
  train_time_end = timer()
  dev_show_message(dev_mode, f"Total training time: {train_time_end - train_time_start:.3f} seconds")

  return model

if __name__ == "__main__":
  model = basic_training_loop(model_type="resnet", dev_mode=True)
  
