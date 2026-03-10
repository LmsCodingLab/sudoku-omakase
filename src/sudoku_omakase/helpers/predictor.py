from functools import cache
import numpy
import torch
from src.sudoku_omakase.models import ResNet18_32, BasicCNNModel, ResNeXt_101
from tests.test_scripts.test_model import test_model


def guess_num(data: numpy.ndarray, model_type: str, dev_mode: bool = False) -> int:
  if dev_mode:
    return test_model(data=data, model_type=model_type)

  model = load_model(model_type)

  input_data = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
  if input_data.max().item() > 1.0:
    input_data = input_data / 255.0

  device = next(model.parameters()).device
  input_data = input_data.to(device)
  
  with torch.no_grad():
    THRESHOLD = 1.1
    logits = model(input_data)
    probabilities = torch.softmax(logits,dim=1)
    dist = torch.distributions.Categorical(probs=probabilities)
    ent = dist.entropy().item()
    _, predicted_class = torch.max(probabilities, dim=1)
    if ent > THRESHOLD:
      prediction = 0 # If the model is not confident enough, return 0 (which could represent "unknown" or "uncertain")
    else:
      prediction = int(predicted_class.item())

  return prediction

@cache
def load_model(model_type: str) -> torch.nn.Module:
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if model_type == "basic":
    model = BasicCNNModel(1, 10)
    state_dict = torch.load("weights/basic_cnn_model.pth", weights_only=True, map_location=device)
  elif model_type == "resnet":
    model = ResNet18_32(1, 10)
    state_dict = torch.load("weights/resnet_model.pth", weights_only=True, map_location=device)
  elif model_type == "resnext":
    model = ResNeXt_101(1, 10)
    state_dict = torch.load("weights/resnext_model.pth", weights_only=True, map_location=device)
  else:
    raise ValueError(f"Unknown model type: {model_type}. Expected 'basic', 'resnet', or 'resnext'.")
  
  model.load_state_dict(state_dict)
  model.to(device)
  model.eval()

  return model 
