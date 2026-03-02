import torch
from torch import nn
from timeit import default_timer as timer
from tqdm import tqdm
from src.models import BasicCNNModel
from src.warmup import warmup
from src.helpers.train_help import accuracy_fn, training_step, testing_step

def basic_training_loop(model: nn.Module):
  device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
  print(f"Using device: {device}")
  
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
  data_loader = warmup()

  train_time_start = timer()
  epochs = 5
  for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}")
    training_step(model=model,
                  data_loader=data_loader,
                  loss_fn=loss_fn,
                  optimizer=optimizer,
                  accuracy_fn=accuracy_fn,
                  device=device)
    
    testing_step(model=model,
              data_loader=data_loader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)
    
  train_time_end = timer()
  print(f"Total training time: {train_time_end - train_time_start:.3f} seconds")

if __name__ == "__main__":
  torch.manual_seed(42)
  model = BasicCNNModel(input_shape=1, hidden_units=10, output_shape=10)
  basic_training_loop(model=model)
