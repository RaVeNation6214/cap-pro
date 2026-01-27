"""
Dataset loader for the newALLBUGS smart contract vulnerability dataset.

Dataset Structure:
- contract/: Full Solidity source files
- threelines-tokenseq/: 3-line windows centered on fault lines (tokenized)
- ast/: AST token sequences
- pretrain_label/label190.pkl: Contract address -> [vulnerability IDs]
- code_w2i.pkl, code_i2w.pkl: Pre-built vocabulary

Vulnerability ID Mapping:
- Arithmetic: 31, 32, 36, 89 (integer overflow/underflow)
- Access Control: 39 (tx.origin authentication)
- Unchecked Calls: 40, 42, 43, 82 (unchecked return values)
- Reentrancy: 41 (external call before state update)
"""

import os
import pickle
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


# Vulnerability ID to class mapping
VULN_ID_TO_CLASS = {
    31: 0,  # Arithmetic
    32: 0,  # Arithmetic
    36: 0,  # Arithmetic
    89: 0,  # Arithmetic
    39: 1,  # Access Control
    40: 2,  # Unchecked Calls
    42: 2,  # Unchecked Calls
    43: 2,  # Unchecked Calls
    82: 2,  # Unchecked Calls
    41: 3,  # Reentrancy
}

CLASS_NAMES = ["Arithmetic", "Access Control", "Unchecked Calls", "Reentrancy"]
NUM_CLASSES = 4


@dataclass
class ContractSample:
    """Single contract sample with its data."""
    address: str
    full_code: str
    fault_window: str  # 3-line tokenized window
    ast_tokens: str
    labels: List[int]  # Original vulnerability IDs
    class_labels: List[int]  # Mapped to 4 classes (multi-hot)


class VulnerabilityDataset(Dataset):
    """
    PyTorch Dataset for smart contract vulnerability detection.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_seq_len: int = 512,
        use_fault_windows: bool = True,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Path to newALLBUGS directory
            split: One of "train", "val", "test"
            max_seq_len: Maximum sequence length
            use_fault_windows: Whether to use 3-line fault windows (True) or full contracts (False)
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.split = split
        self.max_seq_len = max_seq_len
        self.use_fault_windows = use_fault_windows

        # Load vocabulary
        self.word2idx, self.idx2word = self._load_vocabulary()
        self.vocab_size = len(self.word2idx)

        # Load labels
        self.labels = self._load_labels()

        # Load samples
        self.samples = self._load_samples()

        # Split data
        random.seed(seed)
        all_addresses = list(self.samples.keys())
        random.shuffle(all_addresses)

        n_train = int(len(all_addresses) * train_ratio)
        n_val = int(len(all_addresses) * val_ratio)

        if split == "train":
            self.addresses = all_addresses[:n_train]
        elif split == "val":
            self.addresses = all_addresses[n_train:n_train + n_val]
        else:  # test
            self.addresses = all_addresses[n_train + n_val:]

        print(f"Loaded {len(self.addresses)} samples for {split} split")

    def _load_vocabulary(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Load pre-built vocabulary."""
        w2i_path = os.path.join(self.data_dir, "code_w2i.pkl")
        i2w_path = os.path.join(self.data_dir, "code_i2w.pkl")

        with open(w2i_path, "rb") as f:
            word2idx = pickle.load(f)
        with open(i2w_path, "rb") as f:
            idx2word = pickle.load(f)

        # Ensure special tokens exist
        if "<PAD>" not in word2idx:
            word2idx["<PAD>"] = len(word2idx)
            idx2word[word2idx["<PAD>"]] = "<PAD>"
        if "<CLS>" not in word2idx:
            word2idx["<CLS>"] = len(word2idx)
            idx2word[word2idx["<CLS>"]] = "<CLS>"

        return word2idx, idx2word

    def _load_labels(self) -> Dict[str, List[int]]:
        """Load vulnerability labels."""
        label_path = os.path.join(self.data_dir, "pretrain_label", "label190.pkl")
        with open(label_path, "rb") as f:
            labels = pickle.load(f)
        return labels

    def _load_samples(self) -> Dict[str, ContractSample]:
        """Load all contract samples."""
        samples = {}

        contract_dir = os.path.join(self.data_dir, "contract")
        window_dir = os.path.join(self.data_dir, "threelines-tokenseq")
        ast_dir = os.path.join(self.data_dir, "ast")

        # Only load contracts that have labels
        for address, vuln_ids in self.labels.items():
            # Try to find the contract file (may have suffix like _1, _2)
            contract_file = None
            window_file = None
            ast_file = None

            # Check for exact match first
            exact_path = os.path.join(contract_dir, f"{address}.sol")
            if os.path.exists(exact_path):
                contract_file = exact_path
                window_file = os.path.join(window_dir, f"{address}.sol")
                ast_file = os.path.join(ast_dir, f"{address}.sol")
            else:
                # Check for suffixed versions
                for suffix in ["_1", "_2", "_3", "_4", "_5"]:
                    suffixed_path = os.path.join(contract_dir, f"{address}{suffix}.sol")
                    if os.path.exists(suffixed_path):
                        contract_file = suffixed_path
                        window_file = os.path.join(window_dir, f"{address}{suffix}.sol")
                        ast_file = os.path.join(ast_dir, f"{address}{suffix}.sol")
                        break

            if contract_file is None:
                continue

            # Read files
            try:
                with open(contract_file, "r", encoding="utf-8", errors="ignore") as f:
                    full_code = f.read()

                fault_window = ""
                if os.path.exists(window_file):
                    with open(window_file, "r", encoding="utf-8", errors="ignore") as f:
                        fault_window = f.read().strip()

                ast_tokens = ""
                if os.path.exists(ast_file):
                    with open(ast_file, "r", encoding="utf-8", errors="ignore") as f:
                        ast_tokens = f.read().strip()

                # Convert vulnerability IDs to multi-hot class labels
                class_labels = [0] * NUM_CLASSES
                for vid in vuln_ids:
                    if vid in VULN_ID_TO_CLASS:
                        class_labels[VULN_ID_TO_CLASS[vid]] = 1

                samples[address] = ContractSample(
                    address=address,
                    full_code=full_code,
                    fault_window=fault_window,
                    ast_tokens=ast_tokens,
                    labels=vuln_ids,
                    class_labels=class_labels
                )
            except Exception as e:
                print(f"Error loading {address}: {e}")
                continue

        return samples

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text using the pre-built vocabulary."""
        # Split on whitespace and special characters
        tokens = text.replace("<sep>", " <sep> ").split()

        # Convert to indices
        indices = []
        for token in tokens:
            if token in self.word2idx:
                indices.append(self.word2idx[token])
            else:
                indices.append(self.word2idx.get("<UNK>", 0))

        return indices

    def __len__(self) -> int:
        return len(self.addresses)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        address = self.addresses[idx]
        sample = self.samples[address]

        # Choose which text to use
        if self.use_fault_windows and sample.fault_window:
            text = sample.fault_window
        else:
            text = sample.full_code

        # Tokenize
        token_ids = self.tokenize(text)

        # Truncate or pad
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
        else:
            token_ids = token_ids + [0] * (self.max_seq_len - len(token_ids))

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "labels": torch.tensor(sample.class_labels, dtype=torch.float),
            "address": address
        }

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced data."""
        class_counts = [0] * NUM_CLASSES

        for address in self.addresses:
            sample = self.samples[address]
            for i, label in enumerate(sample.class_labels):
                class_counts[i] += label

        total = sum(class_counts)
        weights = [total / (NUM_CLASSES * count) if count > 0 else 1.0 for count in class_counts]

        return torch.tensor(weights, dtype=torch.float)


class WindowDataset(Dataset):
    """
    Dataset that creates multiple windows per contract for hierarchical model.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_seq_len: int = 256,
        window_size: int = 3,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42
    ):
        """
        Initialize the windowed dataset.

        Args:
            data_dir: Path to newALLBUGS directory
            split: One of "train", "val", "test"
            max_seq_len: Maximum sequence length per window
            window_size: Number of lines per window
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            seed: Random seed
        """
        self.data_dir = data_dir
        self.split = split
        self.max_seq_len = max_seq_len
        self.window_size = window_size

        # Load vocabulary
        self.word2idx, self.idx2word = self._load_vocabulary()
        self.vocab_size = len(self.word2idx)

        # Load labels
        self.labels = self._load_labels()

        # Load and process samples
        self.samples = self._load_and_window_samples()

        # Split data
        random.seed(seed)
        all_addresses = list(self.samples.keys())
        random.shuffle(all_addresses)

        n_train = int(len(all_addresses) * train_ratio)
        n_val = int(len(all_addresses) * val_ratio)

        if split == "train":
            self.addresses = all_addresses[:n_train]
        elif split == "val":
            self.addresses = all_addresses[n_train:n_train + n_val]
        else:
            self.addresses = all_addresses[n_train + n_val:]

        print(f"Loaded {len(self.addresses)} contracts for {split} split")

    def _load_vocabulary(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Load pre-built vocabulary."""
        w2i_path = os.path.join(self.data_dir, "code_w2i.pkl")
        i2w_path = os.path.join(self.data_dir, "code_i2w.pkl")

        with open(w2i_path, "rb") as f:
            word2idx = pickle.load(f)
        with open(i2w_path, "rb") as f:
            idx2word = pickle.load(f)

        # Add special tokens if missing
        special_tokens = ["<PAD>", "<CLS>", "<SEP>", "<MASK>"]
        for token in special_tokens:
            if token not in word2idx:
                word2idx[token] = len(word2idx)
                idx2word[word2idx[token]] = token

        return word2idx, idx2word

    def _load_labels(self) -> Dict[str, List[int]]:
        """Load vulnerability labels."""
        label_path = os.path.join(self.data_dir, "pretrain_label", "label190.pkl")
        with open(label_path, "rb") as f:
            return pickle.load(f)

    def _load_and_window_samples(self) -> Dict[str, Dict]:
        """Load contracts and create windows."""
        samples = {}
        contract_dir = os.path.join(self.data_dir, "contract")

        for address, vuln_ids in self.labels.items():
            # Find contract file
            contract_file = None
            for suffix in ["", "_1", "_2", "_3", "_4", "_5"]:
                path = os.path.join(contract_dir, f"{address}{suffix}.sol")
                if os.path.exists(path):
                    contract_file = path
                    break

            if contract_file is None:
                continue

            try:
                with open(contract_file, "r", encoding="utf-8", errors="ignore") as f:
                    code = f.read()

                # Create windows from lines
                lines = code.split("\n")
                windows = []
                for i in range(0, len(lines), self.window_size):
                    window = "\n".join(lines[i:i + self.window_size])
                    windows.append(window)

                if not windows:
                    windows = [code]

                # Convert vulnerability IDs to class labels
                class_labels = [0] * NUM_CLASSES
                for vid in vuln_ids:
                    if vid in VULN_ID_TO_CLASS:
                        class_labels[VULN_ID_TO_CLASS[vid]] = 1

                samples[address] = {
                    "code": code,
                    "windows": windows,
                    "labels": vuln_ids,
                    "class_labels": class_labels
                }
            except Exception as e:
                print(f"Error loading {address}: {e}")

        return samples

    def tokenize_window(self, window: str) -> List[int]:
        """Tokenize a single window."""
        tokens = window.split()
        indices = []
        for token in tokens:
            if token in self.word2idx:
                indices.append(self.word2idx[token])
            else:
                indices.append(self.word2idx.get("<UNK>", 0))

        # Pad or truncate
        if len(indices) > self.max_seq_len:
            indices = indices[:self.max_seq_len]
        else:
            pad_idx = self.word2idx.get("<PAD>", 0)
            indices = indices + [pad_idx] * (self.max_seq_len - len(indices))

        return indices

    def __len__(self) -> int:
        return len(self.addresses)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a contract with all its windows."""
        address = self.addresses[idx]
        sample = self.samples[address]

        # Tokenize all windows
        max_windows = 50  # Limit number of windows per contract
        windows = sample["windows"][:max_windows]

        window_ids = []
        for window in windows:
            ids = self.tokenize_window(window)
            window_ids.append(ids)

        # Pad to max_windows
        pad_window = [0] * self.max_seq_len
        while len(window_ids) < max_windows:
            window_ids.append(pad_window)

        # Create mask for valid windows
        window_mask = [1] * len(sample["windows"][:max_windows])
        window_mask += [0] * (max_windows - len(window_mask))

        return {
            "input_ids": torch.tensor(window_ids, dtype=torch.long),
            "window_mask": torch.tensor(window_mask, dtype=torch.float),
            "labels": torch.tensor(sample["class_labels"], dtype=torch.float),
            "num_windows": len(sample["windows"][:max_windows]),
            "address": address
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    max_seq_len: int = 512,
    use_windows: bool = False,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Create train, validation, and test dataloaders.

    Returns:
        train_loader, val_loader, test_loader, vocab_size
    """
    if use_windows:
        DatasetClass = WindowDataset
        kwargs = {"max_seq_len": max_seq_len // 2, "window_size": 3}
    else:
        DatasetClass = VulnerabilityDataset
        kwargs = {"max_seq_len": max_seq_len, "use_fault_windows": True}

    train_dataset = DatasetClass(data_dir, split="train", **kwargs)
    val_dataset = DatasetClass(data_dir, split="val", **kwargs)
    test_dataset = DatasetClass(data_dir, split="test", **kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn if use_windows else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn if use_windows else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn if use_windows else None
    )

    return train_loader, val_loader, test_loader, train_dataset.vocab_size


def collate_fn(batch):
    """Custom collate function for variable-length windows."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    window_mask = torch.stack([item["window_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    return {
        "input_ids": input_ids,
        "window_mask": window_mask,
        "labels": labels
    }


# For testing
if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "newALLBUGS"

    print("Testing VulnerabilityDataset...")
    dataset = VulnerabilityDataset(data_dir, split="train")
    print(f"Vocab size: {dataset.vocab_size}")
    print(f"Train samples: {len(dataset)}")

    sample = dataset[0]
    print(f"Sample shape: {sample['input_ids'].shape}")
    print(f"Labels: {sample['labels']}")

    print("\nClass weights:", dataset.get_class_weights())
