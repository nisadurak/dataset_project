import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# =====================
# SABİTLER
# =====================

CLASS_NAMES = [
    "First-Person",
    "Isometric",
    "Side-Scroller",
    "Third-Person",
    "Top-Down",
]

IMAGE_SIZE = 224

RESNET_PATH = "models/resnet50_best.pth"
CNN_PATH = "models/cnn_best.pth"

# =====================
# TRANSFORM
# =====================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# =====================
# MODELLER
# =====================

class GameCamNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


_resnet_model = None
_cnn_model = None


def load_resnet():
    global _resnet_model
    if _resnet_model is not None:
        return _resnet_model

    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(CLASS_NAMES))

    checkpoint = torch.load(RESNET_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    _resnet_model = model
    return model


def load_cnn():
    global _cnn_model
    if _cnn_model is not None:
        return _cnn_model

    model = GameCamNet(num_classes=len(CLASS_NAMES))
    checkpoint = torch.load(CNN_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    _cnn_model = model
    return model


# =====================
# TAHMİN YARDIMCILARI
# =====================

def _predict_with_model(model, image_path: str):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    top_idx = probs.argmax().item()
    top_label = CLASS_NAMES[top_idx]
    top_conf = float(probs[top_idx])

    top3_vals, top3_idxs = torch.topk(probs, 3)
    top3 = [
        (CLASS_NAMES[i], float(v))
        for v, i in zip(top3_vals, top3_idxs)
    ]

    return top_label, top_conf, top3


def predict_with_model(model_name: str, image_path: str):
    """
    model_name: 'resnet' veya 'cnn'
    """
    if model_name == "cnn":
        model = load_cnn()
    else:
        model = load_resnet()

    return _predict_with_model(model, image_path)
