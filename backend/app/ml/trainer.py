"""
Training pipeline for the Hierarchical Transformer model.

This module provides training functionality for the smart contract
vulnerability detection model using the newALLBUGS dataset.
"""

import os
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, classification_report
)

from .model import HierarchicalTransformer
from .dataset import (
    VulnerabilityDataset, WindowDataset,
    create_dataloaders, CLASS_NAMES, NUM_CLASSES
)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Data
    data_dir: str = "newALLBUGS"
    use_windows: bool = False  # Use windowed dataset for hierarchical model

    # Model
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 3
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.1

    # Training
    batch_size: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    warmup_epochs: int = 5
    patience: int = 10  # Early stopping patience
    gradient_clip: float = 1.0

    # Paths
    save_dir: str = "models"
    log_dir: str = "logs"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingMetrics:
    """Metrics for a single epoch."""
    loss: float
    accuracy: float
    f1_macro: float
    f1_per_class: List[float]
    precision_macro: float
    recall_macro: float
    roc_auc: Optional[float] = None


class Trainer:
    """
    Trainer for the Hierarchical Transformer model.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Create directories
        os.makedirs(config.save_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.training_history = []

    def setup(self):
        """Set up data loaders, model, optimizer, etc."""
        print("Setting up training...")

        # Create data loaders
        print("Loading data...")
        self.train_loader, self.val_loader, self.test_loader, vocab_size = create_dataloaders(
            data_dir=self.config.data_dir,
            batch_size=self.config.batch_size,
            max_seq_len=self.config.max_seq_len,
            use_windows=self.config.use_windows
        )

        # Get class weights for imbalanced data
        class_weights = self.train_loader.dataset.get_class_weights()
        class_weights = class_weights.to(self.device)
        print(f"Class weights: {class_weights.tolist()}")

        # Create model
        print("Creating model...")
        self.model = HierarchicalTransformer(
            vocab_size=vocab_size + 10,  # Extra buffer for special tokens
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            d_ff=self.config.d_ff,
            max_seq_len=self.config.max_seq_len,
            num_classes=NUM_CLASSES,
            dropout=self.config.dropout
        ).to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999)
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs - self.config.warmup_epochs
        )

        # Loss function with class weights
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

        print("Setup complete!")

    def train_epoch(self) -> TrainingMetrics:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            if self.config.use_windows:
                input_ids = batch["input_ids"].to(self.device)
                window_mask = batch["window_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Create dummy static features (batch, num_windows, 12)
                batch_size, num_windows, seq_len = input_ids.shape
                static_features = torch.zeros(batch_size, num_windows, 12).to(self.device)

                # Forward pass
                logits, _, _ = self.model(input_ids, static_features, window_mask)
            else:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Reshape for non-windowed input: (batch, seq_len) -> (batch, 1, seq_len)
                input_ids = input_ids.unsqueeze(1)
                static_features = torch.zeros(input_ids.size(0), 1, 12).to(self.device)

                # Forward pass
                logits, _, _ = self.model(input_ids, static_features)

            # Compute loss
            loss = self.criterion(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )

            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()

            preds = torch.sigmoid(logits) > 0.5
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        # Calculate metrics
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        metrics = self._calculate_metrics(all_preds, all_labels, total_loss / len(self.train_loader))

        return metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> TrainingMetrics:
        """Evaluate on a data loader."""
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []

        for batch in loader:
            if self.config.use_windows:
                input_ids = batch["input_ids"].to(self.device)
                window_mask = batch["window_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                batch_size, num_windows, seq_len = input_ids.shape
                static_features = torch.zeros(batch_size, num_windows, 12).to(self.device)

                logits, _, _ = self.model(input_ids, static_features, window_mask)
            else:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                input_ids = input_ids.unsqueeze(1)
                static_features = torch.zeros(input_ids.size(0), 1, 12).to(self.device)

                logits, _, _ = self.model(input_ids, static_features)

            loss = self.criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = probs > 0.5

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_probs = np.vstack(all_probs)
        all_labels = np.vstack(all_labels)

        metrics = self._calculate_metrics(
            all_preds, all_labels,
            total_loss / len(loader),
            all_probs
        )

        return metrics

    def _calculate_metrics(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        loss: float,
        probs: Optional[np.ndarray] = None
    ) -> TrainingMetrics:
        """Calculate evaluation metrics."""
        # Accuracy
        accuracy = (preds == labels).mean()

        # F1 scores
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
        f1_per_class = f1_score(labels, preds, average=None, zero_division=0).tolist()

        # Precision and Recall
        precision_macro = precision_score(labels, preds, average="macro", zero_division=0)
        recall_macro = recall_score(labels, preds, average="macro", zero_division=0)

        # ROC-AUC (if probabilities available)
        roc_auc = None
        if probs is not None:
            try:
                roc_auc = roc_auc_score(labels, probs, average="macro")
            except ValueError:
                pass  # May fail if a class is not present

        return TrainingMetrics(
            loss=loss,
            accuracy=accuracy,
            f1_macro=f1_macro,
            f1_per_class=f1_per_class,
            precision_macro=precision_macro,
            recall_macro=recall_macro,
            roc_auc=roc_auc
        )

    def train(self):
        """Run the full training loop."""
        print(f"\nStarting training on {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        print("-" * 60)

        for epoch in range(1, self.config.epochs + 1):
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.evaluate(self.val_loader)

            # Update scheduler (after warmup)
            if epoch > self.config.warmup_epochs:
                self.scheduler.step()

            # Log
            elapsed = time.time() - start_time
            lr = self.optimizer.param_groups[0]["lr"]

            print(f"Epoch {epoch:3d}/{self.config.epochs} | "
                  f"Train Loss: {train_metrics.loss:.4f} | "
                  f"Val Loss: {val_metrics.loss:.4f} | "
                  f"Val F1: {val_metrics.f1_macro:.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Time: {elapsed:.1f}s")

            # Print per-class F1
            f1_str = " | ".join(f"{CLASS_NAMES[i]}: {f1:.3f}"
                               for i, f1 in enumerate(val_metrics.f1_per_class))
            print(f"         Per-class F1: {f1_str}")

            # Save history
            self.training_history.append({
                "epoch": epoch,
                "train": asdict(train_metrics),
                "val": asdict(val_metrics),
                "lr": lr
            })

            # Early stopping check
            if val_metrics.f1_macro > self.best_val_f1:
                self.best_val_f1 = val_metrics.f1_macro
                self.patience_counter = 0
                self.save_checkpoint("best_model.pt")
                print(f"         New best model! F1: {self.best_val_f1:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        # Final evaluation on test set
        print("\n" + "=" * 60)
        print("Final evaluation on test set:")
        self.load_checkpoint("best_model.pt")
        test_metrics = self.evaluate(self.test_loader)

        print(f"Test Loss: {test_metrics.loss:.4f}")
        print(f"Test Accuracy: {test_metrics.accuracy:.4f}")
        print(f"Test F1 (Macro): {test_metrics.f1_macro:.4f}")
        print(f"Test Precision: {test_metrics.precision_macro:.4f}")
        print(f"Test Recall: {test_metrics.recall_macro:.4f}")
        if test_metrics.roc_auc:
            print(f"Test ROC-AUC: {test_metrics.roc_auc:.4f}")

        print("\nPer-class Test F1:")
        for i, f1 in enumerate(test_metrics.f1_per_class):
            print(f"  {CLASS_NAMES[i]}: {f1:.4f}")

        # Save final results
        self.save_results(test_metrics)

        return test_metrics

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.config.save_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": asdict(self.config),
            "best_val_f1": self.best_val_f1
        }, path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.config.save_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def save_results(self, test_metrics: TrainingMetrics):
        """Save training results and history."""
        results = {
            "config": asdict(self.config),
            "test_metrics": asdict(test_metrics),
            "training_history": self.training_history,
            "best_val_f1": self.best_val_f1
        }

        path = os.path.join(self.config.log_dir, "training_results.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {path}")


def main():
    """Main training entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train vulnerability detection model")
    parser.add_argument("--data_dir", type=str, default="newALLBUGS",
                       help="Path to newALLBUGS dataset")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--use_windows", action="store_true",
                       help="Use windowed dataset for hierarchical model")

    args = parser.parse_args()

    config = TrainingConfig(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_windows=args.use_windows
    )

    trainer = Trainer(config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
