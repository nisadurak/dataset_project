import os
from pathlib import Path
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm


# ================== AYARLAR ==================

DATA_DIR = Path("dataset") 
BATCH_SIZE = 32
NUM_EPOCHS_CNN = 15
NUM_EPOCHS_RESNET = 10
LR_CNN = 1e-3
LR_RESNET = 1e-4
IMAGE_SIZE = 224

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ================== MODEL 1: SIFIRDAN KÜÇÜK CNN ==================

class GameCamNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            # 3 x 224 x 224
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 32 x 112 x 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 64 x 56 x 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 128 x 28 x 28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 256 x 14 x 14
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),   # 256 x 1 x 1
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


# ================== YARDIMCI FONKSİYONLAR ==================

def get_dataloaders():
    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"

    # augmentation + normalizasyon
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, train_dataset.classes


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val  ", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, num_epochs, lr, tag, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_path = MODELS_DIR / f"{tag}_best.pth"

    for epoch in range(1, num_epochs + 1):
        print(f"\n[{tag}] Epoch {epoch}/{num_epochs}")
        t0 = time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        dt = time() - t0
        print(f"[{tag}] "
              f"Train loss: {train_loss:.4f}  acc: {train_acc*100:.2f}% | "
              f"Val loss: {val_loss:.4f}  acc: {val_acc*100:.2f}%  "
              f"({dt:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "epoch": epoch,
            }, best_path)
            print(f"[{tag}] -> Yeni en iyi model kaydedildi: {best_path}")

    print(f"\n[{tag}] Eğitim bitti. En iyi val acc: {best_val_acc*100:.2f}%\n")


# ================== MAIN ==================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device: {device}")

    train_loader, val_loader, classes = get_dataloaders()
    num_classes = len(classes)
    print(f"[+] Sınıflar ({num_classes}): {classes}")

    # ------- Model 1: Sıfırdan CNN -------
    print("\n==============================")
    print("== 1) GameCamNet (scratch) ==")
    print("==============================")

    cnn_model = GameCamNet(num_classes=num_classes).to(device)
    train_model(
        model=cnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS_CNN,
        lr=LR_CNN,
        tag="cnn",
        device=device,
    )

    # ------- Model 2: ResNet50 (pretrained) -------
    print("\n==============================")
    print("== 2) ResNet50 (pretrained) ==")
    print("==============================")

    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Son katmanı değiştirdim
    in_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features, num_classes)
    resnet = resnet.to(device)

    train_model(
        model=resnet,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS_RESNET,
        lr=LR_RESNET,
        tag="resnet50",
        device=device,
    )


if __name__ == "__main__":
    main()
