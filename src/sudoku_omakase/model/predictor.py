from functools import cache
import numpy
import torch
from sudoku_omakase.model.models import ResNet18_32, BasicCNNModel, ResNeXt_101, ModelType
from sudoku_omakase.model.weights import load_model_from_origin

def guess_num(data: numpy.ndarray, model_type: ModelType) -> int:
  """
  Uses the specified model to predict the digit represented by the input data.

  Parameters:
  - data: numpy.ndarray, the input data representing the image of a single cell (should be a 2D array of pixel values).
  - model_type: ModelType, the type of model to use for prediction (e.g., ModelType.SMALL, ModelType.NORMAL, ModelType.BIG).  

  Returns:
  - int, the predicted digit (0-9) for the input cell. If the model is not confident enough, it may return 0 to indicate "uncertain".
  """
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


def _load_compatible_state_dict(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
  """Load checkpoints while handling legacy fc naming differences.

  Some released checkpoints store the classifier weights as
  `model.fc.1.{weight,bias}` (Sequential with Dropout + Linear), while the
  current model definition may expose them as `model.fc.{weight,bias}`
  (direct Linear). This keeps loading backward/forward compatible.
  """
  try:
    model.load_state_dict(state_dict)
    return
  except RuntimeError:
    pass

  if "model.fc.1.weight" in state_dict and "model.fc.weight" not in state_dict:
    state_dict = dict(state_dict)
    state_dict["model.fc.weight"] = state_dict.pop("model.fc.1.weight")
    state_dict["model.fc.bias"] = state_dict.pop("model.fc.1.bias")
    model.load_state_dict(state_dict)
    return

  if "model.fc.weight" in state_dict and "model.fc.1.weight" not in state_dict:
    state_dict = dict(state_dict)
    state_dict["model.fc.1.weight"] = state_dict.pop("model.fc.weight")
    state_dict["model.fc.1.bias"] = state_dict.pop("model.fc.bias")
    model.load_state_dict(state_dict)
    return

  model.load_state_dict(state_dict)

@cache
def load_model(model_type: ModelType) -> torch.nn.Module:
  """
  Loads the specified model type with pre-trained weights.

  Parameters:
  - model_type: ModelType, the type of model to load (e.g., ModelType.SMALL, ModelType.NORMAL, ModelType.BIG).

  Returns:
  - torch.nn.Module, the loaded model ready for inference.
  """
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if model_type == ModelType.SMALL:
    model = BasicCNNModel(1, 10)
  elif model_type == ModelType.NORMAL:
    model = ResNet18_32(1, 10)
  elif model_type == ModelType.BIG:
    model = ResNeXt_101(1, 10)
  else:
    raise ValueError(f"Unknown model type: {model_type}. Expected ModelType.SMALL, ModelType.NORMAL, or ModelType.BIG.")
  
  path_to_weights = load_model_from_origin(model_type)
  state_dict = torch.load(path_to_weights, map_location=device)
  _load_compatible_state_dict(model, state_dict)
  model.to(device)
  model.eval()

  return model 
