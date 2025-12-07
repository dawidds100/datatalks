"""
Questions 5 & 6: Training with data augmentation (10 more epochs)

Question 5: What is the mean of test loss for all epochs with augmentations?
Answer options: 0.008, 0.08, 0.88, 8.88

Question 6: What's the average of test accuracy for the last 5 epochs (6-10)?
Answer options: 0.08, 0.28, 0.68, 0.98

This script:
1. Loads the pre-trained model from questions 3 & 4
2. Adds data augmentations to the training data
3. Continues training for 10 more epochs
4. Evaluates on the test set
5. Calculates answers for questions 5 and 6

NOTE: Run question_3_4.py first to generate the pre-trained model!
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os

# ============================================================================
# REPRODUCIBILITY SETUP
# ============================================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ============================================================================
# STEP 1: Setup directories
# ============================================================================
train_dir = "data/train"
test_dir = "data/test"

# ============================================================================
# STEP 2: Define transformations WITH augmentation
# ============================================================================
train_transforms_aug = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.RandomRotation(50),
    transforms.RandomResizedCrop(200, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
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

# ============================================================================
# STEP 3: Load datasets
# ============================================================================
# Training set WITH augmentation
train_dataset_full = datasets.ImageFolder(train_dir, transform=train_transforms_aug)
train_size = int(0.8 * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size
train_dataset, _ = random_split(
    train_dataset_full,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

# Test set (no augmentation)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# Create data loaders
batch_size = 20
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}\n")

# ============================================================================
# STEP 4: Define and load model
# ============================================================================
class HairTypeCNN(nn.Module):
    def __init__(self):
        super(HairTypeCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d((2, 2))
        
        self.fc1 = nn.Linear(32 * 100 * 100, 64)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x

model = HairTypeCNN()
model = model.to(device)

# Load pre-trained weights from questions 3 & 4
if os.path.exists('model_after_10_epochs.pth'):
    model.load_state_dict(torch.load('model_after_10_epochs.pth', map_location=device))
    print("Loaded pre-trained model from 'model_after_10_epochs.pth'\n")
else:
    print("WARNING: Pre-trained model not found!")
    print("Make sure to run question_3_4.py first to generate it.\n")

# ============================================================================
# STEP 5: Setup loss function and optimizer
# ============================================================================
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.8)

# ============================================================================
# STEP 6: Continue training with augmentation (10 more epochs)
# ============================================================================
print("=" * 80)
print("TRAINING WITH AUGMENTATION (10 MORE EPOCHS)")
print("=" * 80)

num_epochs = 10
test_history = {'loss': [], 'acc': []}

for epoch in range(num_epochs):
    # Training phase
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
    
    # Test phase (evaluation on test set)
    model.eval()
    test_running_loss = 0.0
    correct_test = 0
    total_test = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_running_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    
    test_epoch_loss = test_running_loss / len(test_dataset)
    test_epoch_acc = correct_test / total_test
    test_history['loss'].append(test_epoch_loss)
    test_history['acc'].append(test_epoch_acc)
    
    print(f"Epoch {epoch+11:2d}/{num_epochs+10} | "
          f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
          f"Test Loss: {test_epoch_loss:.4f} | Test Acc: {test_epoch_acc:.4f}")

# ============================================================================
# QUESTION 5: Mean of test loss
# ============================================================================
print("\n" + "=" * 80)
print("QUESTION 5: Mean of test loss for augmented model")
print("=" * 80)

test_losses = test_history['loss']
mean_test_loss = np.mean(test_losses)

print(f"\nTest losses per epoch (epochs 11-20):")
for i, loss in enumerate(test_losses, 11):
    print(f"  Epoch {i:2d}: {loss:.4f}")

print(f"\nMean test loss: {mean_test_loss:.4f}")

# Find closest option
options_q5 = [0.008, 0.08, 0.88, 8.88]
closest_q5 = min(options_q5, key=lambda x: abs(x - mean_test_loss))
print(f"Closest option: {closest_q5}")

# ============================================================================
# QUESTION 6: Average of test accuracy for last 5 epochs (6-10 of augmented training)
# ============================================================================
print("\n" + "=" * 80)
print("QUESTION 6: Average test accuracy for last 5 epochs (epochs 16-20)")
print("=" * 80)

test_accs = test_history['acc']
last_5_accs = test_accs[-5:]
avg_last_5_acc = np.mean(last_5_accs)

print(f"\nTest accuracies for last 5 epochs:")
for i, acc in enumerate(last_5_accs, 16):
    print(f"  Epoch {i:2d}: {acc:.4f}")

print(f"\nAverage of last 5 epochs: {avg_last_5_acc:.4f}")

# Find closest option
options_q6 = [0.08, 0.28, 0.68, 0.98]
closest_q6 = min(options_q6, key=lambda x: abs(x - avg_last_5_acc))
print(f"Closest option: {closest_q6}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY - QUESTIONS 5 & 6")
print("=" * 80)
print(f"Question 5 - Mean test loss:                    {mean_test_loss:.4f} (closest: {closest_q5})")
print(f"Question 6 - Avg test accuracy (last 5 epochs): {avg_last_5_acc:.4f} (closest: {closest_q6})")
print("=" * 80)

# Save final model
torch.save(model.state_dict(), 'model_final_after_augmentation.pth')
print("\nFinal model saved as 'model_final_after_augmentation.pth'")
