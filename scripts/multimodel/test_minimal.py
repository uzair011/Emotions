# test_minimal.py
import torch
from torchvision import models

# Simplified model
class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet18(weights=None)
        self.base_model.fc = torch.nn.Linear(512, 7)
        
model = TestModel()
torch.save(model.state_dict(), "test.pth")

# Test loading
loaded_model = TestModel()
loaded_model.load_state_dict(torch.load("test.pth", weights_only=True))
print("Success!")