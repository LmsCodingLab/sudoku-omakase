import torch
from torch import nn
from timeit import default_timer as timer
from src.models import BasicCNNModel, ResNet18_32
from src.warmup import warmup
from src.helpers.train_help import accuracy_fn, training_step, testing_step
from src.helpers.dev_info import dev_show_message

def basic_training_loop(model: nn.Module, dev_mode: bool = False):
  device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # type: ignore[assignment]
  model.to(device)
  dev_show_message(dev_mode, f"Using device: {device}")
  
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
  data_loader, test_loader = warmup(dev_mode=dev_mode)

  train_time_start = timer()
  epochs = 7
  for epoch in range(epochs):
    dev_show_message(dev_mode, f"Epoch: {epoch}")
    training_step(model=model,
                  data_loader=data_loader,
                  loss_fn=loss_fn,
                  optimizer=optimizer,
                  accuracy_fn=accuracy_fn,
                  device=device) # type: ignore[call-arg]
    
    testing_step(model=model,
              data_loader=test_loader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device) # type: ignore[call-arg]
    
  train_time_end = timer()
  dev_show_message(dev_mode, f"Total training time: {train_time_end - train_time_start:.3f} seconds")

if __name__ == "__main__":
  torch.manual_seed(42)
  model = BasicCNNModel(input_shape=1, output_shape=10)
  basic_training_loop(model=model, dev_mode=True)
  model = ResNet18_32(input_shape=1, output_shape=10)
  basic_training_loop(model=model, dev_mode=True)
