# Computer Vision Project: CIFAR-4 Classification

## Overview
This project implements a custom CNN with ResNet-style residual blocks to classify a 4-class subset of the CIFAR-10 dataset. The subset includes cats, frogs, airplanes, and cars.

## Project Setup

### Dependencies
```python
# Standard libraries
import os
import math
import numpy as np
import time
import random
import hashlib

# PyTorch and vision
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.utils.data
import torchvision

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
```

### Dataset Configuration
- Classes: cat, frog, airplane, car
- Train split: 4000 images (1000 per class)
- Validation split: 1200 images (300 per class)
- Test split: 1200 images (300 per class)

## Model Architecture

### ResidualBlock
The custom CNN includes ResNet-style residual blocks with:
- Two convolutional layers with batch normalization
- Skip connections
- ReLU activation functions

### SimpleCustomCNN
```python
class SimpleCustomCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, 1, stride=1)
        self.layer2 = self._make_layer(32, 1, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)
```

## Training Configuration

### Hyperparameters
- Loss Function: CrossEntropyLoss
- Optimizer: Stochastic Gradient Descent
- Initial Learning Rate: 0.3
- Learning Rate Scheduler: Cosine Annealing
- Training Epochs: 55
- Batch Size: 100

### Data Augmentation
```python
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

## Results

### Training Progress
- Best Validation Accuracy: 84.58% (Epoch 44)
- Final Test Accuracy: 82.83%

### Confusion Matrix Results
```
[[233  35  26   6]
 [ 43 251   3   3]
 [ 18   6 263  13]
 [ 10   2  41 247]]
```

Class-wise Performance:
- Cat: 233/300 correct predictions
- Frog: 251/300 correct predictions
- Airplane: 263/300 correct predictions
- Car: 247/300 correct predictions

### Model Checkpointing
- Best model saved as 'best_val_model.pt'
- Checkpoint includes:
  - Model state dict
  - Optimizer state
  - Epoch number

## Usage

### Training
```python
# Prepare data
train_dataset, train_data_loader, val_dataset, val_data_loader, test_dataset, test_data_loader = prepare_data()

# Create and train model
CNN_model = SimpleCustomCNN(num_classes=4)
CNN_loss_module = nn.CrossEntropyLoss()
CNN_optimizer = torch.optim.SGD(CNN_model.parameters(), lr=0.3)
CNN_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(CNN_optimizer, T_max=50)

# Train
train_val_ckp_model(CNN_model, CNN_optimizer, train_data_loader, val_data_loader, 
                    CNN_loss_module, scheduler=CNN_scheduler, num_epochs=55)
```

### Evaluation
```python
# Load best model
CNN_model, CNN_optimizer, CNN_best_epoch = load_ckp('best_val_model.pt', CNN_model, CNN_optimizer)

# Evaluate
eval_model(CNN_model, test_data_loader)
```

## Future Improvements
1. Experiment with deeper architectures
2. Implement additional data augmentation techniques
3. Try different optimizers (Adam, AdamW)
4. Add more residual blocks
5. Implement learning rate warmup
