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

from config import N_CONCEPTS, N_CLASSES
from src import ModelXtoC
from scripts.run_preprocessing import preprocessing_main
from src.utils import find_class_imbalance
from src.training import run_epoch_x_to_c

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train X -> C Concept Model')
    # Data/Paths
    parser.add_argument('--log_dir', default='outputs/x_to_c_model', help='Directory to save logs and best model')
    # Model Hyperparameters
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze convolutional layers of InceptionV3')
    parser.add_argument('--use_aux', action='store_true', default=True, help='Use auxiliary logits') # Defaulting to True as in notebook
    # Training Hyperparameters
    parser.add_argument('--epochs', '-e', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00004, help='Weight decay for SGD optimizer')
    parser.add_argument('--scheduler_step', type=int, default=1000, help='Step size for StepLR scheduler')
    parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam'], help='Optimizer type')
    parser.add_argument('--use_weighted_loss', action='store_true', default=True, help='Use weighted BCE loss for concepts') # Defaulting to True
    # Logging/Saving
    parser.add_argument('--log_interval', type=int, default=50, help='How often to print training progress (batches)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
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
    # Assuming preprocessing_main returns: full_concept_labels_matrix, train_loader, val_loader, test_loader
    concept_labels_all, train_loader, val_loader, test_loader = preprocessing_main(verbose=False)
    print("Data loaded.")

    # --- Model ---
    print("Initializing model...")
    model = ModelXtoC(pretrained=True,
                    freeze=args.freeze_backbone,
                    n_classes=N_CLASSES, # Needed for InceptionV3 structure, even if not used for loss
                    use_aux=args.use_aux,
                    n_concepts=N_CONCEPTS)
    model = model.to(device)
    print("Model initialized (X -> C).")

    # --- Loss ---
    print("Setting up loss...")
    if args.use_weighted_loss:
        # Pass the full concept label matrix (before splitting) if possible,
        # otherwise calculate from train_loader (less accurate)
        # Here, using the matrix returned by preprocessing_main
        concept_weights = find_class_imbalance(concept_labels_all)
        print(f"Using weighted loss. Example weights (first 5): {[f'{w:.2f}' for w in concept_weights[:5]]}")
        attr_criterion = [nn.BCEWithLogitsLoss(weight=torch.tensor(ratio, device=device, dtype=torch.float))
                        for ratio in concept_weights]
    else:
        print("Using unweighted loss.")
        attr_criterion = [nn.BCEWithLogitsLoss() for _ in range(N_CONCEPTS)]
    print("Loss setup complete.")

    # --- Optimizer and Scheduler ---
    print("Setting up optimizer and scheduler...")
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
         optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.lr,
                                weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.1)
    print("Optimizer and scheduler ready.")

    # --- Training Loop ---
    print("\nStarting Training Loop...")
    best_val_acc = 0.0
    best_epoch = -1

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        epoch_start_time = time.time()

        # Train
        train_loss, train_acc = run_epoch_x_to_c(
            model, train_loader, attr_criterion, optimizer, device=device,
            is_training=True, use_aux=args.use_aux, n_concepts=N_CONCEPTS,
            log_interval=args.log_interval
        )
        print(f"Epoch {epoch+1} Train Summary | Loss: {train_loss:.4f} | Acc: {train_acc:.3f}")

        # Validate
        val_loss, val_acc = 0.0, 0.0
        if val_loader:
            val_loss, val_acc = run_epoch_x_to_c(
                model, val_loader, attr_criterion, optimizer, device=device, # Optimizer passed but not used when is_training=False
                is_training=False, use_aux=False, n_concepts=N_CONCEPTS, # No aux loss in validation
                log_interval=args.log_interval
            )
            print(f"Epoch {epoch+1} Val Summary   | Loss: {val_loss:.4f} | Acc: {val_acc:.3f}")

            # Save best model based on validation accuracy
            if val_acc > best_val_acc:
                print(f"  Validation accuracy improved ({best_val_acc:.3f} -> {val_acc:.3f}). Saving model...")
                best_val_acc = val_acc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(args.log_dir, 'x_to_c_best_model.pth'))
                print(f"  Model saved to {os.path.join(args.log_dir, 'x_to_c_best_model.pth')}")

        # Scheduler step
        scheduler.step()
        epoch_end_time = time.time()
        print(f"Epoch {epoch+1} Time: {epoch_end_time - epoch_start_time:.2f}s | Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        # logger.write(...) # Add logging to file here if using Logger

    print(f"\nTraining Finished. Best validation accuracy: {best_val_acc:.3f} at epoch {best_epoch}")

    # --- Optional: Test on Test Set ---
    if test_loader and os.path.exists(os.path.join(args.log_dir, 'x_to_c_best_model.pth')):
        print("\nLoading best model and evaluating on test set...")
        model.load_state_dict(torch.load(os.path.join(args.log_dir, 'x_to_c_best_model.pth'), map_location=device))
        test_loss, test_acc = run_epoch_x_to_c(
            model, test_loader, attr_criterion, None, device=device, # No optimizer needed
            is_training=False, use_aux=False, n_concepts=N_CONCEPTS,
            log_interval=args.log_interval
        )
        print(f"Test Set Results | Loss: {test_loss:.4f} | Acc: {test_acc:.3f}")
        # logger.write(...) # Add test results to log


if __name__ == '__main__':
    main()