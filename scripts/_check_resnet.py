import torch
import torchvision.models as models
m = models.resnet50()
m.fc = torch.nn.Linear(m.fc.in_features, 1)
print('Params:', sum(p.numel() for p in m.parameters()))
