#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CIFAR-10 classification training script using three loss functions:

    1. CELoss: Standard CrossEntropy loss.
    2. TopKLoss: Uses the top k% of images in a batch (default k=0.5) for loss computation.
    3. AdaptiveKLoss: Selects the smallest set of top-loss samples whose cumulative loss 
       is at least 2/3 of the total batch loss.

For each batch, the loss function returns:
    - The mean loss.
    - The list of per-sample loss values.
    - The effective number of examples used in the loss computation.

During training, at the end of each epoch the following metrics are logged:
    - examples_used: Total training examples processed.
    - effective_examples_used: Total effective examples used in loss computation.
    - effective_ratio: Ratio of effective examples to total examples.
    - train_accuracy: Average training accuracy.
    - test_accuracy: Evaluation accuracy.
    - train_loss: Average training loss.

These metrics are printed at the end of every epoch and immediately appended to a CSV file
("training_log.csv"). This ensures that if training stops after x epochs, you have the data for the first x epochs.
By default, the script uses ResNet18 (with a modified final layer) as the classifier.
"""

import os
import csv
import cv2
import shutil
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

##########################################
# Loss Functions for Classification
##########################################

def CELoss(outputs, targets):
    """
    Standard CrossEntropy Loss computed over the entire batch.
    
    Returns:
        - Mean loss over the batch.
        - List of per-sample loss values.
        - Effective examples used (all examples in the batch).
    """
    criterion = nn.CrossEntropyLoss(reduction='none')
    losses = criterion(outputs, targets)
    effective_count = losses.size(0)  # All examples are used.
    return losses.mean(), losses.tolist(), effective_count

def TopKLoss(outputs, targets, k=0.5):
    """
    Computes CrossEntropy loss per sample and returns the mean loss computed only over 
    the top k% (default 50%) highest-loss samples.
    
    Args:
        outputs: Model outputs.
        targets: Ground-truth labels.
        k (float): Fraction of images to keep (default 0.5).
    
    Returns:
        - Mean loss over the top k% samples.
        - List of per-sample loss values (for the whole batch).
        - Effective examples used (number of samples selected).
    """
    criterion = nn.CrossEntropyLoss(reduction='none')
    losses = criterion(outputs, targets)
    batch_size = losses.size(0)
    top_k = max(1, int(k * batch_size))
    sorted_losses, _ = torch.sort(losses, descending=True)
    top_k_losses = sorted_losses[:top_k]
    return top_k_losses.mean(), losses.tolist(), top_k

def AdaptiveKLoss(outputs, targets):
    """
    Computes per-sample CrossEntropy losses and then selects the smallest number of top-loss 
    samples whose cumulative loss is at least 2/3 of the total batch loss.
    
    Returns:
        - Mean loss over the selected samples.
        - List of per-sample loss values (for the whole batch).
        - Effective examples used (number of samples selected).
    """
    criterion = nn.CrossEntropyLoss(reduction='none')
    losses = criterion(outputs, targets)
    total_loss = losses.sum()
    threshold = (2/3) * total_loss
    sorted_losses, _ = torch.sort(losses, descending=True)
    cum_sum = torch.cumsum(sorted_losses, dim=0)
    m_indices = (cum_sum >= threshold).nonzero(as_tuple=False)
    if m_indices.numel() == 0:
        m = losses.size(0)
    else:
        m = m_indices[0].item() + 1  # +1 because indices start at 0
    selected_losses = sorted_losses[:m]
    return selected_losses.mean(), losses.tolist(), m

##########################################
# Classifier Models for CIFAR-10
##########################################

class CNN_large(nn.Module):
    """
    A simple CNN for CIFAR-10.
    """
    def __init__(self):
        super(CNN_large, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * 4 * 4, 10)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 32x32 -> 16x16
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # 16x16 -> 8x8
        x = F.relu(F.max_pool2d(self.conv3(x), 2))  # 8x8 -> 4x4
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.softmax(x, dim=1)

def get_resnet18_cifar10():
    """
    Returns a ResNet18 model with its final layer adjusted for 10 CIFAR-10 classes.
    """
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

##########################################
# Training and Evaluation Functions
##########################################

def train(model, train_loader, optimizer, loss_function, epochs=20, test_loader=None, csv_fp=None):
    """
    Train the classifier on CIFAR-10.
    
    Args:
        model: The classifier model.
        train_loader: DataLoader for CIFAR-10 training set.
        optimizer: Optimizer.
        loss_function: Loss function to use (CELoss, TopKLoss, or AdaptiveKLoss).
        epochs: Number of epochs.
        test_loader: DataLoader for evaluation (optional).
        csv_fp: Open file pointer for CSV logging (if provided).
    
    Returns:
        A list of dictionaries (one per epoch) with the following keys:
            - 'epoch': Epoch number.
            - 'examples_used': Total training examples processed.
            - 'effective_examples_used': Total effective examples used in loss computation.
            - 'effective_ratio': Ratio of effective examples to total examples.
            - 'train_accuracy': Average training accuracy.
            - 'test_accuracy': Evaluation accuracy (if test_loader provided).
            - 'train_loss': Average training loss.
    """
    epoch_metrics = []
    fieldnames = ["epoch", "examples_used", "effective_examples_used", "effective_ratio",
                  "train_accuracy", "test_accuracy", "train_loss"]
    csv_writer = None
    if csv_fp is not None:
        csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_fp.flush()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        effective_total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            
            # Each loss function returns (loss, losses_list, effective_count)
            loss, batch_losses, effective_count = loss_function(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_correct += (output.argmax(dim=1) == target).sum().item()
            batch_samples = data.size(0)
            total_samples += batch_samples
            effective_total += effective_count
        
        epoch_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / len(train_loader.dataset)
        effective_ratio = effective_total / total_samples
        
        # Evaluate on test set if provided.
        if test_loader is not None:
            test_accuracy = evaluate(model, test_loader, noise_level=0.0)
        else:
            test_accuracy = None
        
        # Prepare the epoch summary.
        epoch_summary = {
            "epoch": epoch,
            "examples_used": total_samples,
            "effective_examples_used": effective_total,
            "effective_ratio": effective_ratio,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_loss": epoch_loss
        }
        
        # Print epoch summary.
        print(f"Epoch {epoch} Summary:")
        print(f"  Total examples processed: {total_samples}")
        print(f"  Total effective examples used: {effective_total} (Ratio = {effective_ratio:.4f})")
        print(f"  Train Loss: {epoch_loss:.6f}")
        print(f"  Train Acc: {train_accuracy:.4f}")
        if test_accuracy is not None:
            print(f"  Test Acc: {test_accuracy:.4f}")
        
        # Log metrics for this epoch.
        epoch_metrics.append(epoch_summary)
        if csv_writer is not None:
            csv_writer.writerow(epoch_summary)
            csv_fp.flush()
    
    return epoch_metrics

def evaluate(model, test_loader, noise_level=0.0):
    """
    Evaluate the classifier on the CIFAR-10 test set.
    
    Args:
        model: The trained classifier.
        test_loader: DataLoader for CIFAR-10 test set.
        noise_level (float): Standard deviation of Gaussian noise to add (default 0.0).
    
    Returns:
        Accuracy on the test set.
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if noise_level > 0:
                noise = torch.randn_like(data) * noise_level
                data = data + noise
            output = model(data)
            predicted = output.argmax(dim=1)
            total_correct += (predicted == target).sum().item()
            total_samples += data.size(0)
    
    accuracy = total_correct / total_samples
    return accuracy

##########################################
# Data Loading for CIFAR-10
##########################################

def get_cifar10_loaders(batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_loader = DataLoader(
        datasets.CIFAR10(root='../data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )
    
    test_loader = DataLoader(
        datasets.CIFAR10(root='../data', train=False, download=True, transform=transform),
        batch_size=1024, shuffle=False
    )
    
    return train_loader, test_loader

##########################################
# Main Training Script with CSV Logging
##########################################

def main():
    # Load CIFAR-10 data.
    train_loader, test_loader = get_cifar10_loaders(batch_size=256)
    
    # Choose the classifier model.
    # Option 1: Use the simple CNN:
    # model = CNN_large().to(device)
    #
    # Option 2: Use ResNet18 (default best model):
    model = get_resnet18_cifar10().to(device)
    
    # Define optimizer.
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Choose a loss function.
    # Options: CELoss, TopKLoss, AdaptiveKLoss.
    # For example, to use TopKLoss with the default 50%:
    # loss_fn = lambda outputs, targets: TopKLoss(outputs, targets, k=0.5)
    #
    # To use AdaptiveKLoss:
    # loss_fn = AdaptiveKLoss
    #
    # Or simply use CELoss:
    loss_fn = AdaptiveKLoss  # Change as desired.
    
    epochs = 100
    # Open the CSV file for logging; data will be written at each epoch.
    csv_filename = "adaptive_log.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        epoch_metrics = train(model, train_loader, optimizer, loss_fn,
                              epochs=epochs, test_loader=test_loader, csv_fp=csvfile)
    
    print(f"Training log saved to {csv_filename}")
    
    # Plot training accuracy and loss over epochs.
    epochs_range = range(epochs)
    train_acc = [entry["train_accuracy"] for entry in epoch_metrics]
    test_acc = [entry["test_accuracy"] for entry in epoch_metrics]
    train_loss = [entry["train_loss"] for entry in epoch_metrics]
    
    plt.figure()
    plt.plot(epochs_range, train_acc, label="Train Accuracy")
    plt.plot(epochs_range, test_acc, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Test Accuracy")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(epochs_range, train_loss, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
