import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import soundfile as sf
import os

# Import our custom modules
from dataset import AudioDataset
from model import CRNN

# configurations
DATA_PATH = "./data"
MODEL_SAVE_PATH = "best_crnn_model.pth"
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_SAVED_MODEL = True
LAST_CHECKPOINT_NAME = "best_crnn_model.pth"

def train():
    # Setup Dataloaders
    
    full_dataset = AudioDataset(DATA_PATH)
    
    # split into training and validation sets with 80% Train, 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training on {DEVICE} with {len(train_data)} samples.")
    
    # Initialize Model
    model = CRNN(num_classes=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # load saved model if available
    start_epoch = 0
    best_val_acc = 0.0
    if USE_SAVED_MODEL and os.path.exists(LAST_CHECKPOINT_NAME):
        checkpoint = torch.load(LAST_CHECKPOINT_NAME)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        print(f"Loaded checkpoint from '{LAST_CHECKPOINT_NAME}'. Resuming from epoch {start_epoch}.")

    # Training Loop
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }
            torch.save(checkpoint, MODEL_SAVE_PATH)
            print(f"  -> Model Saved! (New Best: {best_val_acc:.2f}%)")

if __name__ == "__main__":
    train()