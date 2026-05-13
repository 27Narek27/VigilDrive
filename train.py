import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# 1. Настройки
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "C:/VigilDrive/VigilDrive_Data/DDD_Dataset"
MODEL_PATH_V4 = "C:/VigilDrive/vigil_model_v4_combined.pth" # Путь к старой модели
BATCH_SIZE = 32
EPOCHS = 3 # Для дообучения много эпох не нужно

# 2. Данные (такие же, как были раньше для совместимости)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 3. Загружаем модель v4
model = models.mobilenet_v2(weights=None) # Не берем стандартные, берем ТВОИ
model.classifier[1] = nn.Linear(model.last_channel, 2)

# ВАЖНО: Загружаем твой прогресс
print(f"Загружаю прогресс из {MODEL_PATH_V4}...")
model.load_state_dict(torch.load(MODEL_PATH_V4, map_location=device))
model = model.to(device)

# 4. Настройка оптимизатора (МАЛЕНЬКИЙ lr, чтобы не сломать старые знания)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001) # В 10 раз меньше обычного

# 5. Дообучение
print("Начинаю дообучение v5...")
model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Эпоха {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# 6. Сохраняем как v5
torch.save(model.state_dict(), 'vigil_model_v5_final.pth')
print("✅ Прогресс сохранен в vigil_model_v5_final.pth!")