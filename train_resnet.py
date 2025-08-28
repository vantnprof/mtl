import argparse
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import os
import time
import wandb
import random
from src.mtl import do_cnn2mtl
from src.mlconv import do_cnn2mlconv
from src.lowrank import do_cnn2lowrank
from src.lowrank_mtl import do_cnn2lowrank_mtl
from src.lowrank_channel_mtl import do_cnn2lowrankchannel_mtl
from tqdm import tqdm
from torchvision import models
import numpy as np
import json
from src.regularization import group_lasso_regularization, do_clipping, compute_excess_parameters
from src.lowrank_channel_mtl.lowrank_channel_mtl import LowRankChannelMTL
from utils.computation import calculate_model_multiplications, measure_cpu_inference_time
            

def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    args.device_configuration = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else "CPU"
    torch.set_float32_matmul_precision('high')

    os.makedirs(os.path.join(args.output_dir, args.exp_name), exist_ok=True)
    args.saving_path = os.path.join(args.output_dir, args.exp_name + "_model.pth")
    args.best_path = os.path.join(args.output_dir, 'best_' + args.exp_name+"_model.pth")
    run = wandb.init(project=args.wandb_project, name=args.exp_name, config=args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Data transforms
    if 'cifar' in args.data:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif 'imagenet' in args.data:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    if args.data == "cifar10":
        train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    elif args.data == "imagenet":
        train_dataset = datasets.ImageNet(root="./data", split='train', transform=transform)
        val_dataset = datasets.ImageNet(root="./data", split="val", transform=transform)

    indices = np.random.choice(len(train_dataset), int(args.fraction * len(train_dataset)), replace=False)
    train_dataset = torch.utils.data.Subset(train_dataset, indices)
    
    # Data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, 
                                               shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, 
                                             shuffle=False, num_workers=args.num_workers)
    print("Modules will be replaced: ", args.replaces)

    original_model_name = args.model.split("_")[-1]
    model = getattr(models, original_model_name)(num_classes=args.num_classes)

    if 'mtl' in args.model and 'mlconv' in args.model:
        model = do_cnn2mlconv(model=model,
                            rank=args.rank
        )
        model = do_cnn2mtl(model=model,
                            replaces=args.replaces, 
        )
    elif 'mtl' in args.model and 'lowrank' in args.model and 'channel' in args.model:
        with open(args.rank_budget, 'r') as f:
            rank_budget = json.load(f)
        model = do_cnn2lowrankchannel_mtl(model=model,
                            replaces=args.replaces, 
                            rank_budget=rank_budget
        )
    elif "mtl" in args.model and "lowrank" in args.model:
        model = do_cnn2lowrank_mtl(model=model,
                            replaces=args.replaces, 
                            rank=args.rank
        )
    elif "mtl" in args.model:
        model = do_cnn2mtl(model=model,
                            replaces=args.replaces, 
        )
    elif 'mlconv' in args.model:
        model = do_cnn2mlconv(model=model,
                            rank=args.rank
        )
    elif 'lowrank' in args.model:
        model = do_cnn2lowrank(model=model,
                            rank=args.rank
        )

    total_computation = calculate_model_multiplications(model, input_shape=(1, 3, 32, 32))
    print("Total computation in theory: ", total_computation)

    # Measure CPU inference time
    timing_results = measure_cpu_inference_time(
        model=model,
        input_shape=(1, 3, 32, 32),
        num_runs=100,
        warmup_runs=20
    )
    print(f"Average: {timing_results['mean_ms']:.3f} ms")
    # exit(0)
    
    num_cnn_layers = sum(1 for name, _ in model.named_modules() if isinstance(_, nn.Conv2d))
    print(f"Number of CNN layers in the model: {num_cnn_layers}")
    
    model.to(args.device)

    wandb.watch(model, log="all", log_freq=1, log_graph=True)

    # compute number of trainable parameters
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {model_params:,}")

    # wandb.log({"num_params": model_params})

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Training loop
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    patience = int(1e2)  # You can make this an argument if you want
    epochs_no_improve = 0
    # E_dict = {}
    # J_dict = {}

    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        train_loss_accum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            outputs = model(images)
            # loss = criterion(outputs, labels)
            ce_loss = criterion(outputs, labels)

            if args.lambda_reg < 1:
                reg_loss = group_lasso_regularization(model=model)
            else:
                reg_loss = 0
            loss = (1 - args.lambda_reg) * ce_loss + args.lambda_reg * reg_loss

            loss.backward()
            optimizer.step()
            train_loss_accum += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        epoch_duration = time.time() - start_time
        avg_train_loss = train_loss_accum / len(train_loader.dataset)
        train_accuracy = 100 * train_correct / train_total

        model.eval()
        val_loss_accum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                images, labels = images.to(args.device), labels.to(args.device)
                outputs = model(images)
                loss = criterion(outputs, labels) 
                val_loss_accum += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        val_loss = val_loss_accum / len(val_loader.dataset)        

        if args.clip_every and (epoch + 1) % args.clip_every == 0:
            # threshold = (args.clipping_threshold/args.epochs)*(epoch+1)
            # threshold = (args.clipping_threshold/args.epochs)*(args.epochs-epoch)
            threshold = args.clipping_threshold
            h_dict = do_clipping(model, threshold=threshold, use_absolute=args.use_absolute_clipping)
            wandb.log({
                "clipping_threshold": threshold,
                "epoch": epoch + 1,
                "h_dict": h_dict,
            })
        
        # Count number of parameters after clipping
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model's parameters count: {model_params}")

        # Early stopping
        if val_accuracy > best_val_accuracy:
            epochs_no_improve = 0
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), args.best_path)
            print(f"New best model saved to {args.best_path} - val_loss: {best_val_loss} - val_acc: {best_val_accuracy}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement in {patience} epochs)")
                break
        rank_dict = {}
        for name, module in model.named_modules():
            if isinstance(module, LowRankChannelMTL) and module.A3 is not None:
                rank_dict[f"rank/{name}"] = module.A3.shape[1]
        wandb.log(rank_dict)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "training_time": epoch_duration,
            "best_val_loss": best_val_loss,
            "best_val_accuracy": best_val_accuracy,
            "model_params": model_params,
        })

        print(f"Epoch {epoch + 1}/{args.epochs} - "
              f"Train Loss: {avg_train_loss:.4f} - "
              f"Train Acc: {train_accuracy:.2f}% - "
              f"Val Loss: {val_loss:.4f} - "
              f"Val Acc: {val_accuracy:.2f}% - "
              f"Time: {epoch_duration:.2f}s"
              f"Best Val Loss: {best_val_loss:.4f} - "
              f"Best Val Acc: {best_val_accuracy:.2f}%")
    
    final_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Final model parameters: {final_model_params}")

    wandb.log({
        "final_model_params": final_model_params,
    })
    
    print("Training complete. \nBest validation loss:", best_val_loss)
    print("Best validation accuracy:", best_val_accuracy)    
    wandb.save(args.saving_path)
    torch.save(model.state_dict(), args.saving_path)
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DNN model on Dataset")
    parser.add_argument("--model", type=str, default="resnet18", help="Model architecture")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes for classification")
    parser.add_argument("--batch", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--lambda_reg", type=float, default=0.0, help="Weight for Reg loss vs. regularization")
    parser.add_argument("--clip_every", type=int, default=None, help="How often to perform rank clipping for LR-MTL")
    parser.add_argument("--clipping_threshold", type=float, default=0, help="Threshold for clipping the columns of LR-MTL")
    parser.add_argument("--use_absolute_clipping", action="store_true", help="Use absolute threshold or scaling")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--data", type=str, default="./data", help="Directory for dataset")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset to use for training")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of workers for data loading")
    parser.add_argument("--device", type=str, default=torch.device("cuda") if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--seed", type=int, default=2505, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save the model")
    parser.add_argument("--wandb_project", type=str, default="MultilinearTransformationLayer", help="WandB project name")
    parser.add_argument("--exp_name", type=str, default="Resnet152-CIFAR10", help="Name of the experiment")
    parser.add_argument('--replaces', nargs='+', default=[], help='List of layer names will be replaced.', required=False)   
    parser.add_argument("--rank", type=int, default=1, help="Rank for MLConv layers (only applicable for MLConv)")
    parser.add_argument("--rank_budget", type=str, default=None, help="File to rank budget for low-rank MTL")
    # parser.add_argument("--lr_warmup_epochs", type=int, default=0, help="Epochs to warm up learning rate")
    args = parser.parse_args()

    train(args)
