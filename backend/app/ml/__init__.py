"""Lightweight ML package exports.

Avoid importing heavy ML dependencies (like `torch`) at package import
time so demo-mode services can run without installing PyTorch.
"""

from .tokenizer import SolidityTokenizer
from .features import FeatureExtractor
from .dataset import (
    VulnerabilityDataset, WindowDataset, create_dataloaders,
    VULN_ID_TO_CLASS, CLASS_NAMES, NUM_CLASSES
)

# Model and trainer are optional imports — they may require heavy deps
# (torch, torch-geometric). Import them lazily and fall back to `None`
# so demo-mode (which only needs `FeatureExtractor`) works in a minimal
# environment.
try:
    from .model import HierarchicalTransformer
except Exception:
    HierarchicalTransformer = None

try:
    from .trainer import Trainer, TrainingConfig, TrainingMetrics
except Exception:
    Trainer = None
    TrainingConfig = None
    TrainingMetrics = None

__all__ = [
    "HierarchicalTransformer",
    "SolidityTokenizer",
    "FeatureExtractor",
    "VulnerabilityDataset",
    "WindowDataset",
    "create_dataloaders",
    "VULN_ID_TO_CLASS",
    "CLASS_NAMES",
    "NUM_CLASSES",
    "Trainer",
    "TrainingConfig",
    "TrainingMetrics",
]
