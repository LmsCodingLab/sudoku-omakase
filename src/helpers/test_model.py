import numpy
import torch
from src.training.models import ResNeXt_101, ResNet18_32, BasicCNNModel

def test_model(data: numpy.ndarray, model_type: str) -> int:
  if model_type == "basic":
    model = BasicCNNModel(1, 10)
    state_dict = torch.load("weights/basic_cnn_model.pth", weights_only=True, map_location=torch.device('cpu'))
  elif model_type == "resnet":
    model = ResNet18_32(1, 10)
    state_dict = torch.load("weights/resnet_model.pth", weights_only=True, map_location=torch.device('cpu'))
  elif model_type == "resnext":
    model = ResNeXt_101(1, 10)
    state_dict = torch.load("weights/resnext_model.pth", weights_only=True, map_location=torch.device('cpu'))
  else:
    raise ValueError(f"Unknown model type: {model_type}. Expected 'basic', 'resnet', or 'resnext'.")
  model.load_state_dict(state_dict)

  model.eval()

  input_data = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
  if input_data.max().item() > 1.0:
    input_data = input_data / 255.0
  
  with torch.no_grad():
    THRESHOLD = 0.75
    logits = model(input_data)
    probabilities = torch.softmax(logits,dim=1)
    max_prob, predicted_class = torch.max(probabilities, dim=1)
    if max_prob.item() < THRESHOLD:
      prediction = 0 # If the model is not confident enough, return 0 (which could represent "unknown" or "uncertain")
    else:
      prediction = int(predicted_class.item())
    print(f"Model confidence {max_prob.item():.2f}. The threshold of {THRESHOLD}. Returning prediction: {prediction}")
  
  return prediction




