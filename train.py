"""
VigilDrive — Training Script v7 (Windows fix)
Fixes:
  - Весь код обёрнут в if __name__ == '__main__' (обязательно на Windows)
  - num_workers=0 на CPU (Windows не поддерживает fork)
  - pin_memory=False на CPU
  - GradScaler обновлён до нового API
"""

import time
import copy
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = "final_dataset"
EPOCHS      = 20
BATCH_SIZE  = 32
LR          = 3e-4
DROPOUT     = 0.4
VAL_SPLIT   = 0.15
PATIENCE    = 5
SAVE_PATH   = "vigil_model_v7_best.pth"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP     = DEVICE.type == "cuda"
NUM_WORKERS = 4 if DEVICE.type == "cuda" else 0
PIN_MEMORY  = DEVICE.type == "cuda"
# ─────────────────────────────────────────────────────────────────────────────

# ── Transforms ────────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(12),
    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.25, hue=0.05),
    transforms.RandomGrayscale(p=0.10),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.10)),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


if __name__ == '__main__':

    print(f"Device      : {DEVICE}")
    print(f"Data        : {DATA_DIR}")
    print(f"AMP         : {USE_AMP}")
    print(f"Num workers : {NUM_WORKERS}")

    # ── Dataset split ─────────────────────────────────────────────────────────
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_tf)
    n_val   = int(len(full_dataset) * VAL_SPLIT)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    val_ds.dataset = copy.deepcopy(full_dataset)
    val_ds.dataset.transform = val_tf

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    print(f"Train       : {n_train} samples")
    print(f"Val         : {n_val} samples")
    print(f"Classes     : {full_dataset.classes}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(in_features, 2),
    )
    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params      : {total_params:,}")

    # ── Loss, optimiser, scheduler ────────────────────────────────────────────
    class_weights = torch.tensor([1.5, 1.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler    = GradScaler(device=DEVICE.type, enabled=USE_AMP)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc     = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):

        # Train
        model.train()
        t0 = time.time()
        running_loss, correct, total = 0.0, 0, 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            with autocast(device_type=DEVICE.type, enabled=USE_AMP):
                outputs = model(inputs)
                loss    = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 20 == 0:
                print(f"  [{epoch+1}/{EPOCHS}] batch {batch_idx}/{len(train_loader)}"
                      f"  loss={loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        train_acc  = 100. * correct / total

        # Val
        model.eval()
        val_correct, val_total, val_loss_sum = 0, 0, 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                with autocast(device_type=DEVICE.type, enabled=USE_AMP):
                    outputs = model(inputs)
                val_loss_sum += criterion(outputs.float(), labels).item()
                _, predicted  = outputs.max(1)
                val_total    += labels.size(0)
                val_correct  += predicted.eq(labels).sum().item()

        val_acc  = 100. * val_correct / val_total
        val_loss = val_loss_sum / len(val_loader)
        scheduler.step()

        duration = time.time() - t0
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch+1:>3}/{EPOCHS}  ({duration:.1f}s)")
        print(f"  Train  loss={train_loss:.4f}  acc={train_acc:.2f}%")
        print(f"  Val    loss={val_loss:.4f}  acc={val_acc:.2f}%")
        print(f"  LR     {scheduler.get_last_lr()[0]:.2e}")

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✅ Новый лучший результат! Сохранено → {SAVE_PATH}  "
                  f"(val_acc={best_val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  Нет улучшения ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"\n⏹  Early stopping на эпохе {epoch+1}.")
                break

        print(f"{'─'*60}\n")

    print(f"\n🏁 Обучение завершено. Лучшая val accuracy: {best_val_acc:.2f}%")
    print(f"   Модель сохранена: {SAVE_PATH}")