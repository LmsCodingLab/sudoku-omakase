import numpy
import torch
from src.training.models import ResNet18_32, BasicCNNModel

def manually_test_basic_model(data: numpy.ndarray) -> int:
  model = BasicCNNModel(1, 10)

  state_dict = torch.load("weights/basic_cnn_model.pth", weights_only=True)
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

def manually_test_resnet_model(data: numpy.ndarray) -> int:
  model = ResNet18_32(1, 10)

  state_dict = torch.load("weights/resnet18_32_model.pth", weights_only=True)
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




