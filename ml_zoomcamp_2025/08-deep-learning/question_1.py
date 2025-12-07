"""
Question 1: Which loss function to use for binary hair classification?

Answer: nn.BCEWithLogitsLoss()

Explanation:
- We have a binary classification problem (straight vs curly hair)
- The output layer has 1 neuron (single output)
- BCEWithLogitsLoss combines sigmoid activation with binary cross-entropy
- This is the most appropriate choice for binary classification with single output
"""

import torch
import torch.nn as nn

# Define the loss function
criterion = nn.BCEWithLogitsLoss()

print("Question 1: Which loss function to use?")
print("=" * 50)
print(f"Answer: nn.BCEWithLogitsLoss()")
print("\nReason:")
print("- Binary classification task (straight vs curly hair)")
print("- Single output neuron in the final layer")
print("- BCEWithLogitsLoss combines sigmoid + BCE")
print("- Numerically stable and appropriate for this problem")
print("=" * 50)

# Alternative approaches that would NOT work as well:
print("\nWhy not the other options:")
print("- MSELoss(): Can be used but not optimal for classification")
print("- CrossEntropyLoss(): Designed for multi-class problems (requires multiple outputs)")
print("- CosineEmbeddingLoss(): Used for similarity/embedding tasks, not classification")
