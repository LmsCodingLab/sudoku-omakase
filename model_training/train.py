import torch
from torch import nn
from pathlib import Path

from sudoku_omakase.model.models import BasicCNNModel, ResNeXt_101, ResNet18_32, ModelType
from model_training.warmup import warmup
from model_training.steps import accuracy_fn, training_step, testing_step

def basic_training_loop(model_type: ModelType) -> nn.Module:
  """
  A basic training loop for training a CNN model on a dataset. This function initializes the model, sets up the loss function and optimizer, and runs the training and testing steps for a specified number of epochs.

  Parameters:
  - model_type: str, the type of model to train ("basic", "resnet", or "resnext").

  Returns:
  - nn.Module: The trained model after the training loop is completed.
  """
  torch.cuda.empty_cache() # Clear GPU memory before starting training
  if model_type == ModelType.BAD:
    model = BasicCNNModel(input_shape=1, output_shape=10)
  elif model_type == ModelType.NORMAL:
    model = ResNet18_32(input_shape=1, output_shape=10)
  elif model_type == ModelType.BIG:
    model = ResNeXt_101(input_shape=1, output_shape=10)
  else:    
    raise ValueError(f"Unknown model type: {model_type}. Expected 'basic', 'resnet', or 'resnext'.")

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  
  loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
  optimizer = torch.optim.AdamW(params=model.parameters(), lr=3e-4, weight_decay=1e-4)
  batch_size = 16 if model_type == ModelType.BIG else 64
  data_loader, test_loader = warmup(batch_size=batch_size)

  epochs = 20
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
  print(f"Starting training for {epochs} epochs with model type: {model_type}. Using device: {device}.")
  Path("weights").mkdir(parents=True, exist_ok=True)
  for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train_loss, train_acc = training_step(model=model,
                  data_loader=data_loader,
                  loss_fn=loss_fn,
                  optimizer=optimizer,
                  accuracy_fn=accuracy_fn,
                  device=device)
    
    test_loss, test_acc = testing_step(model=model,
              data_loader=test_loader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device) 
    scheduler.step()
    torch.save(model.state_dict(), f"weights/{model_type}_model{epoch}.pth")
    print(f"Saved model weights to weights/{model_type}_model{epoch}.pth")
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, LR: {current_lr:.6f}")

  return model

if __name__ == "__main__":
  model = basic_training_loop(model_type=ModelType.BAD)
  print(f"Training completed for model type: {ModelType.BAD}")
  # model = basic_training_loop(model_type=ModelType.NORMAL)
  # print(f"Training completed for model type: {ModelType.NORMAL}")
  model = basic_training_loop(model_type=ModelType.BIG)
  print(f"Training completed for model type: {ModelType.BIG}")
