import numpy
import torch
from src.models import ResNet18_32, BasicCNNModel

def test_model(data: numpy.ndarray, model_type: str) -> int:
  if model_type == "basic":
    model = BasicCNNModel(1, 10)
    state_dict = torch.load("weights/basic_cnn_model.pth", weights_only=True)
  elif model_type == "resnet":
    model = ResNet18_32(1, 10)
    state_dict = torch.load("weights/resnet18_32_model.pth", weights_only=True)
  else:
    raise ValueError(f"Unknown model type: {model_type}. Expected 'basic' or 'resnet'.")
  model.load_state_dict(state_dict)

  model.eval()

  input_data = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
  if input_data.max().item() > 1.0:
    input_data = input_data / 255.0
  
  with torch.no_grad():
    THRESHOLD = 0.75
    logits = model(input_data)
    probabilities = torch.softmax(logits,dim=1)
    max_prob, confedence = torch.max(probabilities, dim=1)
    if max_prob.item() < THRESHOLD:
      prediction = 0 # If the model is not confident enough, return 0 (which could represent "unknown" or "uncertain")
      print(f"Model confidence {max_prob.item():.2f} is below the threshold of {THRESHOLD}. Returning prediction: {prediction}")
    else:
      prediction = int(confedence.item())
  
  print(prediction)
  return prediction




