import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.models as models

TEST_FILES = [
    "poses/DIO_1.png",
    "poses/Caesar_Zeppeli_1.png",
    "poses/bucciarati_3.jpg",
]

device = torch.device("cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Identity()
model.eval().to(device)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def emb(path):
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(x).squeeze().cpu().numpy()

for p in TEST_FILES:
    if not os.path.exists(p):
        print("Missing:", p)
        continue
    e = emb(p)
    print(p)
    print("  shape:", e.shape)
    print("  first5:", np.round(e[:5], 4))