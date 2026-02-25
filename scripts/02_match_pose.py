import sys
import json
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
import torchvision.models as models


DB_PATH = "data/pose_database.json"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# Load database
with open(DB_PATH, "r", encoding="utf-8") as f:
    database = json.load(f)

if len(database) == 0:
    raise RuntimeError("Database is empty. Run scripts/01_build_pose_database.py first.")


# Load embedding model
device = torch.device("cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Identity()
model.eval()
model.to(device)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


def image_to_embedding(img_path: str) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(x).squeeze().cpu().numpy()
    return emb


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/02_match_pose.py path/to/query_image.png")
        sys.exit(1)

    query_path = sys.argv[1]
    q = image_to_embedding(query_path)

    results = []
    for item in database:
        db_emb = np.array(item["embedding"], dtype=np.float32)
        sim = cosine_similarity(q, db_emb)
        results.append((item["pose_name"], sim, item["file"]))

    results.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 5 matches:")
    for name, score, file in results[:5]:
        print(f"{name:25s}  sim={score:.4f}   ({file})")

    best_name, best_score, _ = results[0]
    print(f"\nBEST MATCH: {best_name} (sim={best_score:.4f})")

    # Optional threshold suggestion:
    # - If best_score < 0.75, probably "no confident match"
    if best_score < 0.75:
        print("⚠️ Low confidence: consider this 'no match'.")


if __name__ == "__main__":
    main()