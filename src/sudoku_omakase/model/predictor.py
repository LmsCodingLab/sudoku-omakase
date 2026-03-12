from functools import cache
import numpy
import torch
from sudoku_omakase.model.models import ResNet18_32, BasicCNNModel, ResNeXt_101, ModelType
from sudoku_omakase.model.weights import load_model_from_origin

def guess_num(data: numpy.ndarray, model_type: ModelType) -> int:
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
def load_model(model_type: ModelType) -> torch.nn.Module:
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if model_type == ModelType.BAD:
    model = BasicCNNModel(1, 10)
  elif model_type == ModelType.NORMAL:
    model = ResNet18_32(1, 10)
  elif model_type == ModelType.BIG:
    model = ResNeXt_101(1, 10)
  else:
    raise ValueError(f"Unknown model type: {model_type}. Expected ModelType.BAD, ModelType.NORMAL, or ModelType.BIG.")
  
  path_to_weights = load_model_from_origin(model_type)
  state_dict = torch.load(path_to_weights, map_location=device)
  model.load_state_dict(state_dict)
  model.to(device)
  model.eval()

  return model 
