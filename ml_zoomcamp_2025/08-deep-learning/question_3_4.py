"""
ML Zoomcamp Homework - Hair Type Classification
Questions 3 & 4: Training without augmentation

Question 3: What is the median of training accuracy for all the epochs?
Question 4: What is the standard deviation of training loss for all the epochs?

Steps:
1. Download dataset using Python (works on Windows/Linux/Mac)
2. Set up data loaders with batch_size=20, shuffle=True for train
3. Build the model
4. Train for 10 epochs
5. Calculate median accuracy and std dev of loss
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import urllib.request
import zipfile
import os

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# Download and prepare dataset (Python version - works on Windows!)
# ============================================================================
print("\nDownloading dataset...")

url = "https://github.com/SVizor42/ML_Zoomcamp/releases/download/straight-curly-data/data.zip"
filename = "data.zip"

# Download the file
try:
    print(f"Downloading from: {url}")
    urllib.request.urlretrieve(url, filename)
    print(f"Downloaded to {filename}")
    
    # Extract the zip file
    print("Extracting files...")
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall()
    print("Dataset extracted successfully")
    
    # Clean up zip file
    os.remove(filename)
    print("Cleanup complete")
except Exception as e:
    print(f"Error downloading dataset: {e}")
    print("Please download manually from:")
    print("https://github.com/SVizor42/ML_Zoomcamp/releases/download/straight-curly-data/data.zip")

# ============================================================================
# Define model architecture
# ============================================================================
class HairTypeCNN(nn.Module):
    def __init__(self):
        super(HairTypeCNN, self).__init__()
        
        # Convolutional layer: 3 input channels -> 32 filters, kernel 3x3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1)
        self.relu1 = nn.ReLU()
        
        # Max pooling: 2x2
        self.pool = nn.MaxPool2d((2, 2))
        
        # Fully connected layers
        # After conv+pool: 32 * 100 * 100 = 320,000
        self.fc1 = nn.Linear(32 * 100 * 100, 64)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        # Conv + ReLU + Pool
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        
        return x

# ============================================================================
# Prepare data loaders
# ============================================================================
train_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("\nLoading datasets...")
train_dataset = datasets.ImageFolder("data/train", transform=train_transforms)
test_dataset = datasets.ImageFolder("data/test", transform=test_transforms)

# Split into train and validation (80-20 split)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, validation_dataset = random_split(
    train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

batch_size = 20
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

print(f"Train set size: {len(train_dataset)}")
print(f"Validation set size: {len(validation_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# ============================================================================
# Setup model, criterion, and optimizer
# ============================================================================
model = HairTypeCNN()
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.8)

# ============================================================================
# Training loop (10 epochs without augmentation)
# ============================================================================
print("\n" + "="*70)
print("TRAINING FOR 10 EPOCHS (without augmentation)")
print("="*70)

num_epochs = 10
history = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        labels = labels.float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct_train / total_train
    history['loss'].append(epoch_loss)
    history['acc'].append(epoch_acc)
    
    # Validation
    model.eval()
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_running_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    
    val_epoch_loss = val_running_loss / len(validation_dataset)
    val_epoch_acc = correct_val / total_val
    history['val_loss'].append(val_epoch_loss)
    history['val_acc'].append(val_epoch_acc)
    
    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
          f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")

# ============================================================================
# QUESTION 3: Median of training accuracy
# ============================================================================
print("\n" + "="*70)
print("QUESTION 3: Median of training accuracy for all epochs")
print("="*70)

print("\nTraining accuracies for all epochs:")
for i, acc in enumerate(history['acc'], 1):
    print(f"  Epoch {i}: {acc:.4f}")

median_acc = np.median(history['acc'])
print(f"\nMedian: {median_acc:.4f}")

# Compare with options
q3_options = [0.05, 0.12, 0.40, 0.84]
print(f"\nComparing with options:")
for opt in q3_options:
    diff = abs(median_acc - opt)
    marker = " ✓ CLOSEST" if diff == min(abs(median_acc - o) for o in q3_options) else ""
    print(f"  {opt:.2f}: difference = {diff:.4f}{marker}")

closest_q3 = min(q3_options, key=lambda x: abs(x - median_acc))

print("\n" + "="*70)
print(f"ANSWER (Q3): {closest_q3}")
print("="*70)

# ============================================================================
# QUESTION 4: Standard deviation of training loss
# ============================================================================
print("\n" + "="*70)
print("QUESTION 4: Standard deviation of training loss for all epochs")
print("="*70)

print("\nTraining losses for all epochs:")
for i, loss in enumerate(history['loss'], 1):
    print(f"  Epoch {i}: {loss:.4f}")

std_loss = np.std(history['loss'])
print(f"\nStandard deviation: {std_loss:.4f}")

# Compare with options
q4_options = [0.007, 0.078, 0.171, 1.710]
print(f"\nComparing with options:")
for opt in q4_options:
    diff = abs(std_loss - opt)
    marker = " ✓ CLOSEST" if diff == min(abs(std_loss - o) for o in q4_options) else ""
    print(f"  {opt:.3f}: difference = {diff:.4f}{marker}")

closest_q4 = min(q4_options, key=lambda x: abs(x - std_loss))

print("\n" + "="*70)
print(f"ANSWER (Q4): {closest_q4}")
print("="*70)

# ============================================================================
# Additional statistics
# ============================================================================
print("\n" + "="*70)
print("TRAINING STATISTICS")
print("="*70)

print("\nAccuracy statistics:")
print(f"  Min: {min(history['acc']):.4f}")
print(f"  Max: {max(history['acc']):.4f}")
print(f"  Mean: {np.mean(history['acc']):.4f}")
print(f"  Median: {median_acc:.4f}")
print(f"  Std Dev: {np.std(history['acc']):.4f}")

print("\nLoss statistics:")
print(f"  Min: {min(history['loss']):.4f}")
print(f"  Max: {max(history['loss']):.4f}")
print(f"  Mean: {np.mean(history['loss']):.4f}")
print(f"  Std Dev: {std_loss:.4f}")

print("\n" + "="*70)
print("FINAL ANSWERS")
print("="*70)
print(f"Question 3 - Median accuracy: {closest_q3}")
print(f"Question 4 - Std dev loss: {closest_q4}")
print("="*70)
