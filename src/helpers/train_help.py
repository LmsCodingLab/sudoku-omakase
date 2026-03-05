import torch
from src.helpers.dev_info import dev_show_message

def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Calculates accuracy between truth labels and predictions.

    Parameters:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to truth labels.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def training_step(
               model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = torch.device("cpu"), 
               dev_mode: bool = False) -> None:
    """
    Performs a training step for a given model, data loader, loss function, optimizer, and accuracy function.
    Parameters:
        model (torch.nn.Module): The model to be trained.
        data_loader (torch.utils.data.DataLoader): The data loader providing training data.
        loss_fn (torch.nn.Module): The loss function to calculate the loss.
        optimizer (torch.optim.Optimizer): The optimizer to update the model parameters.
        accuracy_fn: A function to calculate the accuracy of predictions.
        device (torch.device, optional): The device to perform training on. Defaults to torch.device("cpu").
    
    Returns:
        None
    """
    train_loss, train_acc = 0, 0
    model.train() # put model in train mode
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    dev_show_message(dev_mode, f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def testing_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = torch.device("cpu"),
              dev_mode: bool = False) -> None:
    """
    Performs a testing step for a given model, data loader, loss function, and accuracy function.

    Parameters:
        data_loader (torch.utils.data.DataLoader): The data loader providing testing data.
        model (torch.nn.Module): The model to be evaluated.
        loss_fn (torch.nn.Module): The loss function to calculate the loss.
        accuracy_fn: A function to calculate the accuracy of predictions.
        device (torch.device, optional): The device to perform testing on. Defaults to torch.device("cpu").
    
    Returns:
        None
    """
    test_loss, test_acc = 0, 0
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y).item()
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
        
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        dev_show_message(dev_mode, f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")