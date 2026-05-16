import time

print(f"Начинаю дообучение v5 на устройстве: {device}")
model.train()

for epoch in range(EPOCHS):
    start_time = time.time()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Считаем точность (accuracy) на ходу
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Вывод прогресса каждые 10 батчей
        if batch_idx % 10 == 0:
            print(f'Эпоха [{epoch+1}/{EPOCHS}] | Батч [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}')

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    duration = time.time() - start_time
    
    print(f"--- Эпоха {epoch+1} завершена за {duration:.1f}с ---")
    print(f"Средний Loss: {epoch_loss:.4f} | Точность: {epoch_acc:.2f}%")
    print("-" * 40)

# Сохранение с учетом версии
FINAL_NAME = 'vigil_model_v5_final.pth'
torch.save(model.state_dict(), FINAL_NAME)
print(f"✅ Модель успешно сохранена как {FINAL_NAME}!")