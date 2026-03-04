import numpy
import torch
from src.models import BasicCNNModel, ResNet18_32

def manually_test_model(data: numpy.ndarray) -> int:
  model = ResNet18_32(1, 10)

  state_dict = torch.load("weights/resnet18_32_model.pth", weights_only=True)
  model.load_state_dict(state_dict)

  model.eval()

  input_data = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
  if input_data.max().item() > 1.0:
    input_data = input_data / 255.0
  
  with torch.no_grad():
    logits = model(input_data)
    prediction = int(torch.argmax(logits, dim=1).item())
  
  print(prediction)
  return prediction




