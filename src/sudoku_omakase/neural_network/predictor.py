from functools import cache

import torch

from sudoku_omakase.neural_network.models import BasicCNNModel, ModelType, ResNet18_32, ResNeXt_101
from sudoku_omakase.neural_network.weights import ensure_weights
from sudoku_omakase.vision.sudoku_image import Image


MODEL_FACTORIES = {
  ModelType.BAD: lambda: BasicCNNModel(1, 10),
  ModelType.NORMAL: lambda: ResNet18_32(1, 10),
  ModelType.BIG: lambda: ResNeXt_101(1, 10),
}

def guess_num(data: Image, model_type: ModelType) -> int:
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
  model = MODEL_FACTORIES[model_type]()
  weights_path = ensure_weights(model_type)
  state_dict = torch.load(weights_path, weights_only=True, map_location=device)
  
  model.load_state_dict(state_dict)
  model.to(device)
  model.eval()

  return model 
