"""
Slither Static Analysis Feature Extractor.

Runs Slither on a Solidity contract and extracts numeric features
that are concatenated with GraphCodeBERT embeddings before classification.

Falls back gracefully if Slither is not installed.
"""
import os
import re
import json
import tempfile
import subprocess
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Fix for Windows: set VIRTUAL_ENV so solc-select writes to venv dir, not Windows Store Python
import sys
_venv_base = os.path.dirname(os.path.dirname(sys.executable))
if os.path.exists(os.path.join(_venv_base, 'pyvenv.cfg')):
    os.environ.setdefault('VIRTUAL_ENV', _venv_base)

# Check if Slither is available (check both global PATH and venv Scripts/)
def _find_slither_cmd():
    import shutil
    # Try in PATH first
    if shutil.which('slither'):
        return 'slither'
    # Try in venv Scripts directory
    _scripts = os.path.join(os.path.dirname(sys.executable))
    _slither_exe = os.path.join(_scripts, 'slither.exe' if sys.platform == 'win32' else 'slither')
    if os.path.exists(_slither_exe):
        return _slither_exe
    return None

_slither_cmd = _find_slither_cmd()
# Build env with VIRTUAL_ENV set (required for solc-select on Windows)
# VIRTUAL_ENV should be venv root (parent of Scripts/bin dir)
_venv_root = os.path.dirname(os.path.dirname(sys.executable))
_slither_env = {**os.environ, 'VIRTUAL_ENV': _venv_root}
try:
    if _slither_cmd:
        result = subprocess.run(
            [_slither_cmd, '--version'],
            capture_output=True, timeout=5, env=_slither_env
        )
        SLITHER_AVAILABLE = result.returncode == 0
    else:
        SLITHER_AVAILABLE = False
except (FileNotFoundError, subprocess.TimeoutExpired):
    SLITHER_AVAILABLE = False

if not SLITHER_AVAILABLE:
    logger.warning(
        "Slither not found. Install with: pip install slither-analyzer\n"
        "Falling back to regex-based static analysis."
    )


@dataclass
class StaticFeatures:
    """Static analysis features for a contract."""
    # Slither-based (if available)
    num_detectors_triggered: int = 0
    num_high_severity: int = 0
    num_medium_severity: int = 0
    num_low_severity: int = 0
    num_informational: int = 0

    # Regex-based (always available)
    has_reentrancy_pattern: bool = False
    has_tx_origin: bool = False
    has_unchecked_return: bool = False
    has_integer_overflow_risk: bool = False
    has_selfdestruct: bool = False
    has_delegatecall: bool = False
    uses_safematch: bool = False
    has_access_control: bool = False

    def to_vector(self) -> List[float]:
        """Convert to numeric feature vector (12 dims)."""
        return [
            min(self.num_detectors_triggered / 10.0, 1.0),
            min(self.num_high_severity / 5.0, 1.0),
            min(self.num_medium_severity / 5.0, 1.0),
            min(self.num_low_severity / 5.0, 1.0),
            float(self.has_reentrancy_pattern),
            float(self.has_tx_origin),
            float(self.has_unchecked_return),
            float(self.has_integer_overflow_risk),
            float(self.has_selfdestruct),
            float(self.has_delegatecall),
            float(self.uses_safematch),
            float(self.has_access_control),
        ]


class SlitherFeatureExtractor:
    """
    Extracts vulnerability signals from Solidity contracts.
    Uses Slither if available, otherwise falls back to regex patterns.
    """

    # Regex patterns for fallback analysis
    REENTRANCY_PATTERN = re.compile(
        r'\.call\s*\{?\s*value\s*:|msg\.sender\.call\s*\.value',
        re.IGNORECASE
    )
    TX_ORIGIN_PATTERN = re.compile(r'\btx\.origin\b')
    UNCHECKED_RETURN_PATTERN = re.compile(
        r'(\.call|\.send|\.delegatecall)\s*[\(\{][^;]*;(?!\s*require)',
        re.MULTILINE
    )
    SEND_WITHOUT_CHECK = re.compile(r'\.send\s*\([^)]*\)\s*;')
    OVERFLOW_PATTERN = re.compile(r'[\+\-\*]\s*(?!=)(?!.*SafeMath)')
    SELFDESTRUCT_PATTERN = re.compile(r'\b(selfdestruct|suicide)\s*\(')
    DELEGATECALL_PATTERN = re.compile(r'\.delegatecall\s*\(')
    SAFEMATCH_PATTERN = re.compile(r'\bSafeMath\b|pragma solidity \^0\.8|pragma solidity >=0\.8')
    ACCESS_CONTROL_PATTERN = re.compile(
        r'\bonlyOwner\b|\bonlyAdmin\b|\brequire\s*\(\s*msg\.sender\s*==',
        re.IGNORECASE
    )
    STATE_AFTER_CALL = re.compile(
        r'(\.call|\.send)\s*[\(\{][^;]*;\s*\n[^}]*=',
        re.MULTILINE | re.DOTALL
    )

    def __init__(self):
        self.use_slither = SLITHER_AVAILABLE

    def extract(self, code: str, contract_path: Optional[str] = None) -> StaticFeatures:
        """
        Extract static features from contract code.

        Args:
            code: Solidity source code string
            contract_path: Optional path to .sol file (used for Slither)

        Returns:
            StaticFeatures dataclass
        """
        features = StaticFeatures()

        # Always do regex analysis
        self._regex_analysis(code, features)

        # Add Slither analysis if available
        if self.use_slither:
            self._slither_analysis(code, contract_path, features)

        return features

    def _regex_analysis(self, code: str, features: StaticFeatures):
        """Fill features using regex patterns."""
        features.has_reentrancy_pattern = bool(
            self.REENTRANCY_PATTERN.search(code) or
            self.STATE_AFTER_CALL.search(code)
        )
        features.has_tx_origin = bool(self.TX_ORIGIN_PATTERN.search(code))
        features.has_unchecked_return = bool(
            self.UNCHECKED_RETURN_PATTERN.search(code) or
            self.SEND_WITHOUT_CHECK.search(code)
        )
        features.has_integer_overflow_risk = (
            bool(self.OVERFLOW_PATTERN.search(code)) and
            not bool(self.SAFEMATCH_PATTERN.search(code))
        )
        features.has_selfdestruct = bool(self.SELFDESTRUCT_PATTERN.search(code))
        features.has_delegatecall = bool(self.DELEGATECALL_PATTERN.search(code))
        features.uses_safematch = bool(self.SAFEMATCH_PATTERN.search(code))
        features.has_access_control = bool(self.ACCESS_CONTROL_PATTERN.search(code))

    def _slither_analysis(self, code: str, contract_path: Optional[str], features: StaticFeatures):
        """Run Slither and extract detector results."""
        try:
            # Write to temp file if no path provided
            if contract_path is None:
                with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.sol', delete=False
                ) as f:
                    f.write(code)
                    contract_path = f.name
                cleanup = True
            else:
                cleanup = False

            report_path = contract_path + '.slither.json'
            cmd = ['slither', contract_path, '--json', report_path]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )

            if os.path.exists(report_path):
                with open(report_path) as f:
                    report = json.load(f)

                results = report.get('results', {}).get('detectors', [])
                features.num_detectors_triggered = len(results)

                severity_counts = {'High': 0, 'Medium': 0, 'Low': 0, 'Informational': 0}
                for r in results:
                    sev = r.get('impact', 'Informational')
                    if sev in severity_counts:
                        severity_counts[sev] += 1

                features.num_high_severity = severity_counts['High']
                features.num_medium_severity = severity_counts['Medium']
                features.num_low_severity = severity_counts['Low']
                features.num_informational = severity_counts['Informational']

                os.remove(report_path)

            if cleanup and os.path.exists(contract_path):
                os.remove(contract_path)

        except Exception as e:
            logger.warning(f"Slither analysis failed: {e}")
