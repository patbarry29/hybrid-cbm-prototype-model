# scripts/train_x_to_c.py
import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Adjust sys.path to find project modules
project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root_path)

from config import CUB_CONFIG
from src.models import ModelXtoC
from src.preprocessing.CUB import preprocessing_main
from src.utils import find_class_imbalance
from src.training import run_epoch_x_to_c

N_CLASSES, N_TRIMMED_CONCEPTS = CUB_CONFIG['N_CLASSES'], CUB_CONFIG['N_TRIMMED_CONCEPTS']

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train X -> C Concept Model')
    # Data/Paths
    parser.add_argument('--log_dir', default='models', help='Directory to save logs and best model')
    # Model Hyperparameters
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze convolutional layers of InceptionV3')
    parser.add_argument('--use_aux', action='store_true', default=True, help='Use auxiliary logits') # Defaulting to True as in notebook
    # Training Hyperparameters
    parser.add_argument('--epochs', '-e', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00004, help='Weight decay for SGD optimizer')
    parser.add_argument('--scheduler_step', type=int, default=1000, help='Step size for StepLR scheduler')
    parser.add_argument('--use_weighted_loss', action='store_true', default=True, help='Use weighted BCE loss for concepts') # Defaulting to True
    # Logging/Saving
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true', help='Print progress')
    # Device
    parser.add_argument('--device', default=None, help='Device override (cuda, mps, cpu). Auto-detects if None.')

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    # --- Setup ---
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available()
                        else "cpu")
    print(f"Using device: {device}")
    # Log directory
    os.makedirs(args.log_dir, exist_ok=True)
    # Logger can be added here if desired: logger = Logger(os.path.join(args.log_dir, 'train.log'))

    # --- Data ---
    print("Loading and preprocessing data...")
    concept_labels, train_loader, test_loader = preprocessing_main(class_concepts=False, verbose=args.verbose)
    print("Data loaded.")

    # --- Model ---
    print("Initializing model...")
    model = ModelXtoC(pretrained=True,
                    freeze=args.freeze_backbone,
                    n_classes=N_CLASSES,
                    use_aux=args.use_aux,
                    n_concepts=N_TRIMMED_CONCEPTS)
    model = model.to(device)
    print("Model initialized (X -> C).")

    # --- Loss ---
    print("Setting up loss...")
    if args.use_weighted_loss:
        concept_weights = find_class_imbalance(concept_labels)
        print(f"Using weighted loss. Example weights (first 5): {[f'{w:.2f}' for w in concept_weights[:5]]}")
        attr_criterion = [nn.BCEWithLogitsLoss(weight=torch.tensor([ratio], device=device, dtype=torch.float))
                    for ratio in concept_weights]
    else:
        print("Using unweighted loss.")
        attr_criterion = [nn.BCEWithLogitsLoss() for _ in range(N_TRIMMED_CONCEPTS)]
    print("Loss setup complete.")

    # --- Optimizer and Scheduler ---
    print("Setting up optimizer and scheduler...")
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr,
                            momentum=0.9,
                            weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.1)
    print("Optimizer and scheduler ready.")

    # --- Training Loop ---
    print("\nStarting Training Loop...")
    best_test_acc = 0.0
    best_epoch = -1

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        epoch_start_time = time.time()

        # Train
        train_loss, train_acc = run_epoch_x_to_c(
            model, train_loader, attr_criterion, optimizer, is_training=True, use_aux=True,
            n_concepts=N_TRIMMED_CONCEPTS, device=device, verbose=args.verbose
        )

        print(f"Epoch {epoch+1} Train Summary | Loss: {train_loss:.4f} | Acc: {train_acc:.3f}")

        test_loss, test_acc = 0.0, 0.0
        if test_loader:
            test_loss, test_acc = run_epoch_x_to_c(
                model, test_loader, attr_criterion, optimizer, n_concepts=N_TRIMMED_CONCEPTS,
                device=device, verbose=args.verbose
            )

            print(f"Epoch {epoch+1} Test Summary   | Loss: {test_loss:.4f} | Acc: {test_acc:.3f}")

            # Save best model based on test accuracy
            if test_acc > best_test_acc:
                print(f"  Test accuracy improved ({best_test_acc:.3f} -> {test_acc:.3f}). Saving model...")
                best_test_acc = test_acc
                best_epoch = epoch + 1
                torch.save(model, os.path.join(args.log_dir, 'instance_level_model.pth'))
                print(f"  Model saved to {os.path.join(args.log_dir, 'instance_level_model.pth')}")

        # Scheduler step
        scheduler.step()
        epoch_end_time = time.time()
        print(f"Epoch {epoch+1} Time: {epoch_end_time - epoch_start_time:.2f}s | Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        # logger.write(...) # Add logging to file here if using Logger

    print(f"\nTraining Finished. Best test accuracy: {best_test_acc:.3f} at epoch {best_epoch}")

if __name__ == '__main__':
    main()
