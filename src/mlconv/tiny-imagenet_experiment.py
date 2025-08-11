import os
import torch
import torch.nn as nn
from torchvision import models
from src.mlconv.cnn2mlconv import cnn2mlconv
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import argparse
import wandb  # Import wandb


def train_and_evaluate(model, train_loader, val_loader, device, epochs, lr_schedule, weight_decay, max_norm):
    """
    Train and evaluate the model on the given data loaders.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr_schedule[0], momentum=0.9, weight_decay=weight_decay)

    for epoch in range(epochs):
        # Adjust learning rate based on schedule
        if epoch in lr_schedule:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[epoch]

        # Training
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Apply max-norm regularization
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.data.clamp(-max_norm, max_norm)

            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        # Validation
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = running_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val

        # Log metrics to wandb
        wandb.log({
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_accuracy
        })

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return train_loss, val_loss, train_accuracy, val_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models on Tiny-ImageNet")
    parser.add_argument("--model", type=str, choices=["original_resnet18", "mlconv_resnet18"], required=True,
                        help="Choose which model to train: original_resnet18 or mlconv_resnet18")
    parser.add_argument("--rank", type=int, default=1,
                        help="Rank for MLConv layers (only applicable for mlconv_resnet18)")
    parser.add_argument("--wandb_project", type=str, default="DynamicRankMTL",
                        help="Name of the wandb project")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Tiny-ImageNet dataset
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.875, 1.0)),  # Random crops between 56 and 64 pixels
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = ImageFolder(root="./data/tiny-imagenet-200/train", transform=transform_train)
    val_dataset = ImageFolder(root="./data/tiny-imagenet-200/val", transform=transform_test)
    print("The number of training data: {}".format(len(train_dataset)))
    print("The number of validation data: {}".format(len(val_dataset)))

    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=200, shuffle=False)

    # Learning rate schedules
    if args.model == "original_resnet18":
        lr_schedule = {0: 0.01, 60: 0.001, 120: 0.0001}
    elif args.model == "mlconv_resnet18":
        lr_schedule = {0: 0.001, 80: 0.0001, 110: 0.00001}

    # Regularization parameters
    weight_decay = 0.0005
    max_norm = 2.0

    if args.model == "original_resnet18":
        # Original ResNet18
        model = models.resnet18(num_classes=200).to(device)
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Original ResNet18 parameters: {model_params}")

        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            name=args.exp_name,
            config={
                "model": args.model,
                "rank": None,
                "dataset": "Tiny-ImageNet",
                "epochs": 140,
                "batch_size": 200,
                "optimizer": "SGD",
                "learning_rate_schedule": lr_schedule,
                "weight_decay": weight_decay,
                "max_norm": max_norm,
                "trainable_parameters": model_params,
                "architecture": str(model)
            }
        )

        print("Training Original ResNet18...")
        train_and_evaluate(
            model, train_loader, val_loader, device, epochs=140, lr_schedule=lr_schedule,
            weight_decay=weight_decay, max_norm=max_norm
        )

    elif args.model == "mlconv_resnet18":
        # MLConv ResNet18
        model = cnn2mlconv("resnet18", num_classes=200, rank=args.rank).to(device)
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"MLConv ResNet18 parameters: {model_params}")

        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            name=args.exp_name,
            config={
                "model": args.model,
                "rank": args.rank,
                "dataset": "Tiny-ImageNet",
                "epochs": 140,
                "batch_size": 200,
                "optimizer": "SGD",
                "learning_rate_schedule": lr_schedule,
                "weight_decay": weight_decay,
                "max_norm": max_norm,
                "trainable_parameters": model_params,
                "architecture": str(model)
            }
        )

        print(f"Training MLConv ResNet18 with rank {args.rank}...")
        train_and_evaluate(
            model, train_loader, val_loader, device, epochs=140, lr_schedule=lr_schedule,
            weight_decay=weight_decay, max_norm=max_norm
        )

    # Finish wandb run
    wandb.finish()