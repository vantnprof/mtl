# main_convnext.py
# A complete script for distributed training of a ConvNeXt model on ImageNet-1K.
#
# To run this script, use the `torchrun` command-line utility.
# Example:
# torchrun --nproc_per_node=<NUM_GPUS> main_convnext.py --data_dir <PATH_TO_IMAGENET>
#
# Replace <NUM_GPUS> with the number of GPUs to use (e.g., 4 or 8) and
# <PATH_TO_IMAGENET> with the root directory of your ImageNet dataset.

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

def setup_distributed(rank, world_size):
    """
    Initializes the distributed training environment.
    `torchrun` automatically sets the MASTER_ADDR, MASTER_PORT, RANK, and WORLD_SIZE
    environment variables.
    """
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"Initialized process {rank}/{world_size} on GPU {torch.cuda.current_device()}.")

def cleanup_distributed():
    """Cleans up the distributed environment."""
    dist.destroy_process_group()

def get_dataloaders(data_dir, batch_size, num_workers, world_size, rank):
    """
    Creates the training and validation DataLoaders for ImageNet-1K.
    Includes standard data augmentations for training.
    """
    # Standard ImageNet normalization values
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Data augmentation pipeline for the training set
    # TrivialAugmentWide is a strong, standard augmentation policy.
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        normalize,
    ])

    # Simple resize and center crop for the validation set
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Create datasets. Assumes ImageNet is in the standard folder structure.
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)

    # DistributedSampler ensures each GPU gets a unique shard of the data.
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, train_sampler

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, rank, sampler):
    """Runs one full epoch of training."""
    model.train()
    # Set the epoch for the sampler to ensure proper shuffling in distributed training.
    sampler.set_epoch(epoch)
    
    # Progress bar is only shown on the main process (rank 0) to avoid clutter.
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1} Training", disable=(rank != 0))
    
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Update progress bar on the main process
        if rank == 0:
            progress_bar.set_postfix(loss=loss.item())
            
@torch.no_grad()
def validate(model, loader, criterion, device, world_size):
    """Validates the model on the validation set."""
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    total_samples = 0

    # Progress bar for validation on the main process
    progress_bar = tqdm(loader, desc="Validating", disable=(dist.get_rank() != 0))

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Accumulate loss and accuracy metrics for this process's shard
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total_samples += labels.size(0)
        correct_top1 += predicted.eq(labels).sum().item()

    # Create tensors to hold metrics from this process
    metrics_tensor = torch.tensor([total_loss, correct_top1, total_samples]).to(device)

    # Use all_reduce to sum the metrics across all GPUs.
    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

    # Extract the summed metrics
    total_loss_global = metrics_tensor[0].item()
    correct_top1_global = metrics_tensor[1].item()
    total_samples_global = metrics_tensor[2].item()
    
    # Calculate final average loss and accuracy
    avg_loss = total_loss_global / total_samples_global
    accuracy = 100. * correct_top1_global / total_samples_global
    
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description='PyTorch ConvNeXt Training on ImageNet-1K')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the root of the ImageNet dataset')
    parser.add_argument('--model_name', type=str, default='convnext_tiny.in1k', help='Model name from timm library (e.g., convnext_tiny, convnext_small)')
    parser.add_argument('--epochs', type=int, default=100, help='Total number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=4e-3, help='Base learning rate')
    parser.add_argument('--wd', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers per GPU')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save model checkpoints')
    args = parser.parse_args()

    # Get rank and world size from environment variables set by torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    setup_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    # Create DataLoaders
    train_loader, val_loader, train_sampler = get_dataloaders(
        args.data_dir, args.batch_size, args.num_workers, world_size, rank
    )

    # Create Model using timm
    # Using pretrained=True provides a good starting point (transfer learning).
    model = timm.create_model(args.model_name, pretrained=True, num_classes=1000).to(device)
    # Wrap the model with DDP for distributed training
    model = DDP(model, device_ids=[rank])

    # Loss, Optimizer, and Learning Rate Scheduler
    criterion = nn.CrossEntropyLoss()
    # AdamW is the standard optimizer for modern architectures like ConvNeXt.
    # The learning rate is scaled by the world size, a common practice in large-batch distributed training.
    optimizer = optim.AdamW(model.parameters(), lr=args.lr * world_size, weight_decay=args.wd)
    # CosineAnnealingLR is a common and effective learning rate schedule.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, rank, train_sampler)
        
        val_loss, val_acc = validate(model, val_loader, criterion, device, world_size)
        scheduler.step()

        # All logging and model saving is done only on the main process (rank 0)
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs} -> "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
            
            # Save the best performing model based on validation accuracy
            if val_acc > best_acc:
                best_acc = val_acc
                # Save the model's state_dict. It's important to access the underlying
                # model via `model.module` to save the weights without the DDP wrapper.
                save_path = os.path.join(args.output_dir, f"{args.model_name}_best.pth")
                torch.save(model.module.state_dict(), save_path)
                print(f"New best model saved to {save_path} with accuracy: {best_acc:.2f}%")

    cleanup_distributed()


if __name__ == '__main__':
    main()
