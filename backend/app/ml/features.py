import re
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class StaticFeatures:
    """Container for static features extracted from code."""
    has_call_value: bool = False
    has_send: bool = False
    has_transfer: bool = False
    has_delegatecall: bool = False
    has_tx_origin: bool = False
    has_selfdestruct: bool = False
    num_arithmetic_ops: int = 0
    num_external_calls: int = 0
    num_state_writes: int = 0
    num_require_assert: int = 0
    has_reentrancy_pattern: bool = False
    has_unchecked_return: bool = False

    def to_vector(self) -> List[float]:
        """Convert features to a numerical vector."""
        return [
            float(self.has_call_value),
            float(self.has_send),
            float(self.has_transfer),
            float(self.has_delegatecall),
            float(self.has_tx_origin),
            float(self.has_selfdestruct),
            min(self.num_arithmetic_ops / 10.0, 1.0),  # Normalize
            min(self.num_external_calls / 5.0, 1.0),
            min(self.num_state_writes / 10.0, 1.0),
            min(self.num_require_assert / 5.0, 1.0),
            float(self.has_reentrancy_pattern),
            float(self.has_unchecked_return),
        ]


class FeatureExtractor:
    """
    Extracts static features from Solidity code for vulnerability detection.
    """

    # Patterns for vulnerability indicators
    PATTERNS = {
        'call_value': r'\.call\s*\{?\s*value\s*:',
        'call_generic': r'\.call\s*[\({]',
        'send': r'\.send\s*\(',
        'transfer': r'\.transfer\s*\(',
        'delegatecall': r'\.delegatecall\s*\(',
        'tx_origin': r'tx\.origin',
        'selfdestruct': r'selfdestruct\s*\(',
        'arithmetic_ops': r'[\+\-\*\/\%](?!=)',
        'state_write': r'=(?!=)',
        'require': r'require\s*\(',
        'assert': r'assert\s*\(',
        'external_call': r'\.(call|send|transfer|delegatecall)\s*[\(\{]',
    }

    # Reentrancy pattern: external call followed by state change
    REENTRANCY_PATTERN = re.compile(
        r'\.call[\s\S]*?=\s*[^=]',  # call followed by assignment
        re.MULTILINE
    )

    def __init__(self):
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.PATTERNS.items()
        }

    def extract(self, code: str) -> StaticFeatures:
        """Extract static features from Solidity code."""
        features = StaticFeatures()

        # Boolean features
        features.has_call_value = bool(self.compiled_patterns['call_value'].search(code))
        features.has_send = bool(self.compiled_patterns['send'].search(code))
        features.has_transfer = bool(self.compiled_patterns['transfer'].search(code))
        features.has_delegatecall = bool(self.compiled_patterns['delegatecall'].search(code))
        features.has_tx_origin = bool(self.compiled_patterns['tx_origin'].search(code))
        features.has_selfdestruct = bool(self.compiled_patterns['selfdestruct'].search(code))

        # Count features
        features.num_arithmetic_ops = len(self.compiled_patterns['arithmetic_ops'].findall(code))
        features.num_external_calls = len(self.compiled_patterns['external_call'].findall(code))
        features.num_state_writes = len(self.compiled_patterns['state_write'].findall(code))
        features.num_require_assert = (
            len(self.compiled_patterns['require'].findall(code)) +
            len(self.compiled_patterns['assert'].findall(code))
        )

        # Complex patterns
        features.has_reentrancy_pattern = bool(self.REENTRANCY_PATTERN.search(code))
        features.has_unchecked_return = self._check_unchecked_return(code)

        return features

    def _check_unchecked_return(self, code: str) -> bool:
        """Check for unchecked return values from external calls."""
        # Look for .call() without checking the return value
        call_pattern = re.compile(r'\.call\s*[\(\{][^;]*;', re.MULTILINE)
        matches = call_pattern.findall(code)

        for match in matches:
            # Check if return value is captured
            if not re.search(r'\([^)]*,?\s*\)\s*=', match) and 'bool' not in match:
                return True

        return False

    def extract_line_features(self, code: str) -> List[Dict[str, float]]:
        """Extract features for each line of code."""
        lines = code.split('\n')
        line_features = []

        for line in lines:
            features = {
                'has_call': 1.0 if re.search(r'\.call', line) else 0.0,
                'has_send': 1.0 if re.search(r'\.send', line) else 0.0,
                'has_transfer': 1.0 if re.search(r'\.transfer', line) else 0.0,
                'has_delegatecall': 1.0 if re.search(r'\.delegatecall', line) else 0.0,
                'has_tx_origin': 1.0 if re.search(r'tx\.origin', line) else 0.0,
                'has_arithmetic': 1.0 if re.search(r'[\+\-\*\/\%](?!=)', line) else 0.0,
                'has_assignment': 1.0 if re.search(r'=(?!=)', line) else 0.0,
                'has_require': 1.0 if re.search(r'require\s*\(', line) else 0.0,
                'is_function_def': 1.0 if re.search(r'function\s+\w+', line) else 0.0,
                'has_external_modifier': 1.0 if re.search(r'\b(external|public)\b', line) else 0.0,
            }
            line_features.append(features)

        return line_features

    def get_feature_dim(self) -> int:
        """Return the dimension of the feature vector."""
        return 12
