"""
Training Script for Hybrid GNN + GraphCodeBERT Vulnerability Detector.

Follows the PDF spec:
  - 5 classes: reentrancy, arithmetic, access_control, unchecked_calls, timestamp
  - GraphCodeBERT used as frozen feature extractor (no_grad during embed)
  - Early stopping with patience=3
  - 20 epochs, batch_size=8
  - BCEWithLogitsLoss with class weights for imbalance

Usage:
    cd backend
    python training/train.py [--data-dir ./data/splits] [--epochs 20] [--batch-size 8]

Requirements:
    pip install torch torch-geometric transformers scikit-learn tqdm pandas
"""
import os
import sys
import csv
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.graph_builder.cfg_builder import CFGBuilder
from app.ml.embedding import CodeEmbedder
from app.ml.gnn_model import HybridGNN, SimpleMLP, TORCH_GEOMETRIC_AVAILABLE

# 5 vulnerability classes per PDF spec
CLASS_NAMES = ["reentrancy", "arithmetic", "access_control", "unchecked_calls", "timestamp"]
NUM_CLASSES = len(CLASS_NAMES)

# Class weights: higher weight = rarer/harder class
CLASS_WEIGHTS = torch.tensor([2.0, 2.5, 2.5, 1.5, 3.0])


def load_csv_dataset(csv_path: Path) -> List[Dict]:
    """Load contracts from CSV (has 'code' column with actual source)."""
    contracts = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                code = row.get('code', '').strip()
                if not code:
                    # Fall back to reading from path
                    path = row.get('path', '')
                    if path and Path(path).exists():
                        code = Path(path).read_text(encoding='utf-8', errors='replace').strip()
                    if not code:
                        continue

                labels = [
                    int(row.get('reentrancy', 0)),
                    int(row.get('arithmetic', 0)),
                    int(row.get('access_control', 0)),
                    int(row.get('unchecked_calls', 0)),
                    int(row.get('timestamp', 0)),
                ]
                contracts.append({
                    'code': code,
                    'labels': labels,
                })
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping row: {e}")
    return contracts


def prepare_batch_gnn(
    contracts: List[Dict],
    cfg_builder: CFGBuilder,
    embedder: CodeEmbedder,
    device: str,
) -> Optional[Tuple]:
    """Prepare a batch as graph data for GNN training."""
    from torch_geometric.data import Data, Batch

    data_list = []
    all_static = []
    all_labels = []

    for item in contracts:
        code = item['code']
        try:
            graph = cfg_builder.build(code, labels=item['labels'])
        except Exception as e:
            logger.debug(f"CFG build failed: {e}")
            continue

        # Get GraphCodeBERT embeddings for each function node
        # GraphCodeBERT is frozen (no_grad inside embedder.embed)
        node_embeddings = []
        for fn_name in graph.nodes:
            fn_body = _extract_fn_body(code, fn_name)
            emb = embedder.embed(fn_body[:1000])
            node_embeddings.append(emb)

        if not node_embeddings:
            continue

        x = torch.tensor(node_embeddings, dtype=torch.float)  # [N, 768]

        if graph.edges:
            edge_index = torch.tensor(graph.edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        y = torch.tensor(item['labels'], dtype=torch.float)  # [5]

        # Static features: mean of node feature vectors
        n = len(graph.node_features)
        if n == 0:
            continue
        static = torch.tensor(
            [sum(f[i] for f in graph.node_features) / n for i in range(12)],
            dtype=torch.float
        )

        data_list.append(Data(x=x, edge_index=edge_index))
        all_static.append(static)
        all_labels.append(y)

    if not data_list:
        return None

    batch = Batch.from_data_list(data_list)
    static_features = torch.stack(all_static).to(device)
    labels = torch.stack(all_labels).to(device)

    return (
        batch.x.to(device),
        batch.edge_index.to(device),
        batch.batch.to(device),
        static_features,
        labels,
    )


def prepare_batch_mlp(
    contracts: List[Dict],
    cfg_builder: CFGBuilder,
    device: str,
) -> Optional[Tuple]:
    """Fallback batch using only static features."""
    all_static = []
    all_labels = []

    for item in contracts:
        code = item['code']
        try:
            graph = cfg_builder.build(code)
        except Exception:
            continue
        if not graph.node_features:
            continue

        n = len(graph.node_features)
        static = [sum(f[i] for f in graph.node_features) / n for i in range(12)]
        all_static.append(static)
        all_labels.append(item['labels'])

    if not all_static:
        return None

    static_features = torch.tensor(all_static, dtype=torch.float).to(device)
    labels = torch.tensor(all_labels, dtype=torch.float).to(device)
    return static_features, labels


def _extract_fn_body(code: str, fn_name: str) -> str:
    """Extract function body for embedding."""
    idx = code.find(fn_name + '(')
    if idx == -1:
        return code[:500]
    brace_idx = code.find('{', idx)
    if brace_idx == -1:
        return code[idx:idx + 500]
    return code[idx:brace_idx + 500]


def compute_metrics(all_preds: List, all_labels: List) -> Dict:
    """Compute precision, recall, F1 from accumulated predictions."""
    if not all_preds:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    preds_cat = torch.cat(all_preds)
    labels_cat = torch.cat(all_labels)

    tp = (preds_cat * labels_cat).sum(0)
    fp = (preds_cat * (1 - labels_cat)).sum(0)
    fn = ((1 - preds_cat) * labels_cat).sum(0)

    precision = (tp / (tp + fp + 1e-8)).mean().item()
    recall = (tp / (tp + fn + 1e-8)).mean().item()
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {"f1": f1, "precision": precision, "recall": recall}


def evaluate_on_data(model, val_data, cfg_builder, embedder, device, use_gnn):
    """Evaluate model on validation/test data."""
    model.eval()
    val_preds, val_labels_all = [], []

    with torch.no_grad():
        for contract in val_data:
            if use_gnn and TORCH_GEOMETRIC_AVAILABLE:
                batch_data = prepare_batch_gnn([contract], cfg_builder, embedder, device)
            else:
                batch_data = prepare_batch_mlp([contract], cfg_builder, device)

            if batch_data is None:
                continue

            if use_gnn and TORCH_GEOMETRIC_AVAILABLE:
                x, edge_index, batch_idx, static, labels = batch_data
                logits = model(x, edge_index, batch_idx, static)
            else:
                static, labels = batch_data
                logits = model(static)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            val_preds.append(preds.cpu())
            val_labels_all.append(labels.cpu())

    return compute_metrics(val_preds, val_labels_all)


def train(args):
    """Main training loop with early stopping."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Training on device: {device}")
    logger.info(f"PyTorch Geometric available: {TORCH_GEOMETRIC_AVAILABLE}")
    logger.info(f"Classes ({NUM_CLASSES}): {CLASS_NAMES}")

    # Load datasets
    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"

    if not train_path.exists():
        logger.error(f"Train CSV not found at {train_path}")
        logger.error("Run: python data/build_dataset.py first")
        return

    logger.info("Loading datasets...")
    train_data = load_csv_dataset(train_path)
    val_data = load_csv_dataset(val_path) if val_path.exists() else []
    logger.info(f"Train: {len(train_data)} | Val: {len(val_data)} contracts")

    if len(train_data) < 50:
        logger.error("Too few training samples. Run build_dataset.py first.")
        return

    # Initialize components
    logger.info("Initializing CFG builder...")
    cfg_builder = CFGBuilder()
    force_no_gnn = getattr(args, 'no_gnn', False)

    # GraphCodeBERT is used as frozen feature extractor
    # (embed() always uses torch.no_grad() internally)
    embedder = None
    if TORCH_GEOMETRIC_AVAILABLE and not force_no_gnn:
        logger.info("Loading GraphCodeBERT (frozen feature extractor)...")
        embedder = CodeEmbedder(device=device)
        if not embedder.is_loaded():
            logger.warning("GraphCodeBERT failed to load, using fallback embeddings")

    use_gnn = TORCH_GEOMETRIC_AVAILABLE and embedder is not None and not force_no_gnn

    # Build model
    if use_gnn:
        logger.info("Building HybridGNN (GraphCodeBERT + GAT)...")
        model = HybridGNN(
            node_in_dim=768,
            static_feature_dim=12,
            hidden_dim=256,
            gat_heads=4,
            num_classes=NUM_CLASSES,
            dropout=0.3,
        )
    else:
        logger.info("Building SimpleMLP (static features only)...")
        model = SimpleMLP(input_dim=12, num_classes=NUM_CLASSES)

    model = model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Note: GraphCodeBERT parameters are NOT included in optimizer
    # (embedder is separate and used with no_grad - effectively frozen)
    n_train = len(train_data)
    if use_gnn and n_train < 500:
        logger.info(f"< 500 samples ({n_train}): GraphCodeBERT used as frozen extractor (no_grad)")

    # Loss with class weights for imbalanced dataset
    weights = CLASS_WEIGHTS.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training setup
    best_f1 = 0.0
    best_epoch = 0
    patience = args.patience
    no_improve = 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "hybrid_gnn.pt"
    history = []

    logger.info(f"Starting training: {args.epochs} epochs, patience={patience}, batch_size={args.batch_size}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()

        import random
        indices = list(range(len(train_data)))
        random.shuffle(indices)

        for batch_start in range(0, len(indices), args.batch_size):
            batch_indices = indices[batch_start:batch_start + args.batch_size]
            batch_contracts = [train_data[i] for i in batch_indices]

            if use_gnn:
                batch_data = prepare_batch_gnn(batch_contracts, cfg_builder, embedder, device)
            else:
                batch_data = prepare_batch_mlp(batch_contracts, cfg_builder, device)

            if batch_data is None:
                continue

            optimizer.zero_grad()

            if use_gnn and TORCH_GEOMETRIC_AVAILABLE:
                x, edge_index, batch_idx, static, labels = batch_data
                logits = model(x, edge_index, batch_idx, static)
            else:
                static, labels = batch_data
                logits = model(static)

            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if n_batches % 10 == 0:
                logger.info(f"  Epoch {epoch} | Batch {n_batches} | Loss: {loss.item():.4f}")

        scheduler.step()
        elapsed = time.time() - start_time
        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        val_metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        if val_data:
            val_metrics = evaluate_on_data(model, val_data, cfg_builder, embedder, device, use_gnn)

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val P: {val_metrics['precision']:.4f} | "
            f"Val R: {val_metrics['recall']:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        history.append({"epoch": epoch, "train_loss": avg_loss, **val_metrics})

        # Save best model
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1,
                "config": {
                    "use_gnn": use_gnn,
                    "num_classes": NUM_CLASSES,
                    "class_names": CLASS_NAMES,
                },
            }, model_path)
            logger.info(f"  [SAVED] Best model (Epoch {epoch}, F1={best_f1:.4f})")
        else:
            no_improve += 1
            logger.info(f"  No improvement ({no_improve}/{patience}). Best F1={best_f1:.4f} at epoch {best_epoch}")

        # Early stopping
        if no_improve >= patience:
            logger.info(f"\nEarly stopping triggered after {epoch} epochs (patience={patience})")
            break

    # If no val improvement at all, save final model
    if best_f1 == 0.0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_f1": 0.0,
            "config": {
                "use_gnn": use_gnn,
                "num_classes": NUM_CLASSES,
                "class_names": CLASS_NAMES,
            },
        }, model_path)
        logger.info(f"  [SAVED] Final model (no val data available)")

    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"\nTraining complete!")
    logger.info(f"Best Val F1: {best_f1:.4f} at epoch {best_epoch}")
    logger.info(f"Model saved: {model_path}")
    logger.info(f"History saved: {history_path}")

    # Evaluate on test set
    test_path = Path(args.data_dir) / "test.csv"
    if test_path.exists():
        logger.info("\nEvaluating on test set...")
        test_data = load_csv_dataset(test_path)
        # Load best model
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate_on_data(model, test_data, cfg_builder, embedder, device, use_gnn)
        logger.info(
            f"Test: F1={test_metrics['f1']:.4f} | "
            f"P={test_metrics['precision']:.4f} | "
            f"R={test_metrics['recall']:.4f}"
        )
        # Append test metrics to history
        history.append({"epoch": "test", **test_metrics})
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid GNN vulnerability detector (5 classes)")
    parser.add_argument("--data-dir", default="./data/splits",
                        help="Directory with train.csv, val.csv, test.csv")
    parser.add_argument("--output-dir", default="./models",
                        help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=20, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--no-gnn", action="store_true", help="Use MLP only (no GNN, no GraphCodeBERT)")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
