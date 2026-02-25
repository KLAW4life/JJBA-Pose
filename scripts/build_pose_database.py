import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as T
import torchvision.models as models


# -----------------------
# Settings
# -----------------------
POSES_DIR = "poses"
OUT_PATH = "data/pose_database.json"
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")


# -----------------------
# Load pretrained model (ResNet18 as embedding extractor)
# -----------------------
device = torch.device("cpu")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Identity()  # remove final classification layer
model.eval()
model.to(device)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


def image_to_embedding(img_path: str) -> np.ndarray:
    """Load image -> preprocess -> resnet embedding (512,)"""
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(x).squeeze().cpu().numpy()

    return emb  # shape (512,)


def main():
    os.makedirs("data", exist_ok=True)

    database = []

    files = [f for f in os.listdir(POSES_DIR) if f.lower().endswith(IMG_EXTS)]
    if not files:
        print(f"No images found in {POSES_DIR}")
        return

    for fname in tqdm(sorted(files), desc="Building DB"):
        path = os.path.join(POSES_DIR, fname)

        # KEEP full filename (without extension) as the pose name
        pose_name = os.path.splitext(fname)[0]

        try:
            emb = image_to_embedding(path)
            database.append({
                "pose_name": pose_name,
                "file": path.replace("\\", "/"),
                "embedding": emb.tolist()
            })
            print(f"[OK] {pose_name}")

        except Exception as e:
            print(f"[ERR] {fname}: {e}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(database, f)

    print(f"\nSaved {len(database)} poses to {OUT_PATH}")


if __name__ == "__main__":
    main()