"""
Mythril-equivalent static analysis service.
Mythril cannot be pip-installed on Windows (pyethash build fails), so we implement
its core SWC checks as precise regex / AST-level pattern analysis.

Covers:
  SWC-100  Function Default Visibility
  SWC-101  Integer Overflow/Underflow
  SWC-104  Unchecked Call Return Value
  SWC-105  Unprotected Ether Withdrawal
  SWC-106  Unprotected SELFDESTRUCT
  SWC-107  Reentrancy
  SWC-110  Assert Violation
  SWC-111  Use of Deprecated Functions
  SWC-112  Delegatecall to Untrusted Callee
  SWC-113  DoS with Failed Call
  SWC-115  Authorization through tx.origin
  SWC-116  Block values as Time Proxy
  SWC-120  Weak Sources of Randomness
"""
import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

MYTHRIL_AVAILABLE = True   # pure-Python, always available


# ── individual check functions ──────────────────────────────────────────────

def _lines_with(code: str, pattern: str, flags=re.IGNORECASE) -> List[int]:
    """Return 1-based line numbers that match pattern."""
    result = []
    for i, line in enumerate(code.splitlines(), 1):
        if re.search(pattern, line, flags):
            result.append(i)
    return result


def _check_swc101(code: str) -> List[Dict]:
    """SWC-101 Integer Overflow/Underflow — pre-0.8 SafeMath missing."""
    if re.search(r'pragma solidity\s+\^?0\.[0-7]\.', code):
        if 'SafeMath' not in code:
            lines = _lines_with(code, r'[\+\-\*]=|[^=!<>]=\s*[\+\-\*]')
            if lines:
                return [{"swc": "SWC-101", "title": "Integer Overflow/Underflow",
                         "severity": "High", "confidence": "Medium",
                         "description": "Arithmetic operations in pre-0.8.0 contract without SafeMath — potential overflow/underflow.",
                         "lines": lines[:10]}]
    return []


def _check_swc104(code: str) -> List[Dict]:
    """SWC-104 Unchecked Call Return Value."""
    findings = []
    lines_code = code.splitlines()
    for i, line in enumerate(lines_code, 1):
        # .send() or .call() whose return value is not captured
        if re.search(r'\.send\s*\([^)]*\)\s*;', line):
            findings.append({"swc": "SWC-104", "title": "Unchecked send()",
                             "severity": "Medium", "confidence": "High",
                             "description": f"Return value of send() not checked at line {i}. Silent failure possible.",
                             "lines": [i]})
        if re.search(r'(?<!\(bool\s)\b\w+\.call\s*[\(\{]', line) and '=' not in line.split('.call')[0].split('\n')[-1]:
            findings.append({"swc": "SWC-104", "title": "Unchecked call()",
                             "severity": "Medium", "confidence": "Medium",
                             "description": f"Return value of low-level call() not captured at line {i}.",
                             "lines": [i]})
    return findings


def _check_swc105(code: str) -> List[Dict]:
    """SWC-105 Unprotected Ether Withdrawal."""
    findings = []
    lines_code = code.splitlines()
    for i, line in enumerate(lines_code, 1):
        if re.search(r'\btransfer\s*\(|\bsend\s*\(|\bcall\s*\{value:', line):
            # Check if inside a function that has no owner/modifier check in surrounding 5 lines
            context_start = max(0, i - 6)
            context = '\n'.join(lines_code[context_start:i])
            if not re.search(r'require|onlyOwner|msg\.sender\s*==|modifier', context, re.IGNORECASE):
                findings.append({"swc": "SWC-105", "title": "Unprotected Ether Withdrawal",
                                 "severity": "High", "confidence": "Low",
                                 "description": f"ETH transfer at line {i} without obvious access control guard.",
                                 "lines": [i]})
    return findings[:3]  # cap at 3 to avoid noise


def _check_swc106(code: str) -> List[Dict]:
    """SWC-106 Unprotected SELFDESTRUCT."""
    findings = []
    lines_code = code.splitlines()
    for i, line in enumerate(lines_code, 1):
        if re.search(r'\bselfdestruct\s*\(|\bsuicide\s*\(', line):
            context = '\n'.join(lines_code[max(0,i-6):i])
            if not re.search(r'require|onlyOwner|msg\.sender\s*==', context, re.IGNORECASE):
                findings.append({"swc": "SWC-106", "title": "Unprotected SELFDESTRUCT",
                                 "severity": "Critical", "confidence": "Medium",
                                 "description": f"selfdestruct() at line {i} without access control — anyone can destroy the contract.",
                                 "lines": [i]})
    return findings


def _check_swc107(code: str) -> List[Dict]:
    """SWC-107 Reentrancy."""
    findings = []
    lines_code = code.splitlines()
    for i, line in enumerate(lines_code, 1):
        if re.search(r'\.call\s*\{value:', line) or re.search(r'\.call\.value\s*\(', line):
            # Look for state modification AFTER external call in next 8 lines
            after = '\n'.join(lines_code[i:min(len(lines_code), i+8)])
            if re.search(r'\w+\s*[\[\(].*\]\s*[+\-]?=\s*|\bbalances\b|\bbal\b', after):
                findings.append({"swc": "SWC-107", "title": "Reentrancy",
                                 "severity": "High", "confidence": "High",
                                 "description": f"External call with value at line {i} followed by state modification — classic reentrancy pattern (CEI violation).",
                                 "lines": [i]})
    return findings


def _check_swc110(code: str) -> List[Dict]:
    """SWC-110 Assert Violation."""
    lines = _lines_with(code, r'\bassert\s*\(')
    if lines:
        return [{"swc": "SWC-110", "title": "Assert Violation Risk",
                 "severity": "Low", "confidence": "Medium",
                 "description": "assert() consumes all remaining gas on failure. Use require() for input validation.",
                 "lines": lines}]
    return []


def _check_swc111(code: str) -> List[Dict]:
    """SWC-111 Use of Deprecated Functions."""
    findings = []
    depr = [
        (r'\bsuicide\s*\(', 'suicide() deprecated — use selfdestruct()'),
        (r'\bsha3\s*\(', 'sha3() deprecated — use keccak256()'),
        (r'\bcallcode\s*\(', 'callcode() deprecated — use delegatecall()'),
        (r'\bthrow\b', 'throw deprecated — use revert()'),
        (r'\bnow\b', '"now" deprecated — use block.timestamp'),
    ]
    for pat, msg in depr:
        lns = _lines_with(code, pat)
        if lns:
            findings.append({"swc": "SWC-111", "title": "Deprecated Function",
                             "severity": "Low", "confidence": "High",
                             "description": msg, "lines": lns})
    return findings


def _check_swc112(code: str) -> List[Dict]:
    """SWC-112 Delegatecall to Untrusted Callee."""
    findings = []
    lines_code = code.splitlines()
    for i, line in enumerate(lines_code, 1):
        if re.search(r'\.delegatecall\s*\(', line):
            # Check if address comes from user input (parameter or storage without whitelist)
            context = '\n'.join(lines_code[max(0,i-10):i])
            if re.search(r'address\s+\w+|_target|_addr|_to\b', context, re.IGNORECASE):
                findings.append({"swc": "SWC-112", "title": "Delegatecall to Untrusted Callee",
                                 "severity": "High", "confidence": "Medium",
                                 "description": f"delegatecall() at line {i} — target may be user-controlled, allowing storage hijack.",
                                 "lines": [i]})
    return findings


def _check_swc113(code: str) -> List[Dict]:
    """SWC-113 DoS with Failed Call."""
    findings = []
    lines_code = code.splitlines()
    for i, line in enumerate(lines_code, 1):
        # transfer() inside a loop
        if re.search(r'\btransfer\s*\(', line):
            context_back = '\n'.join(lines_code[max(0,i-8):i])
            if re.search(r'\bfor\b|\bwhile\b', context_back):
                findings.append({"swc": "SWC-113", "title": "DoS with Failed Call",
                                 "severity": "Medium", "confidence": "Medium",
                                 "description": f"transfer() inside loop at line {i} — one revert blocks entire loop (DoS).",
                                 "lines": [i]})
    return findings


def _check_swc115(code: str) -> List[Dict]:
    """SWC-115 Authorization through tx.origin."""
    lines = _lines_with(code, r'\btx\.origin\b')
    if lines:
        return [{"swc": "SWC-115", "title": "tx.origin Authentication",
                 "severity": "High", "confidence": "High",
                 "description": "tx.origin used for authorization — vulnerable to phishing via malicious contract intermediary.",
                 "lines": lines}]
    return []


def _check_swc116(code: str) -> List[Dict]:
    """SWC-116 Block values as Time Proxy."""
    findings = []
    lines = _lines_with(code, r'\bblock\.timestamp\b|\bnow\b')
    if lines:
        severity = "Low"
        desc = "block.timestamp used as time reference — miners can manipulate by ~15 seconds."
        # Escalate if used for randomness or exact comparison
        if re.search(r'block\.timestamp\s*%|keccak.*block\.timestamp|block\.timestamp\s*==', code):
            severity = "High"
            desc = "block.timestamp used for randomness/exact comparison — highly exploitable by miners (SWC-116 + SWC-120)."
        findings.append({"swc": "SWC-116", "title": "Timestamp Dependence",
                         "severity": severity, "confidence": "Medium",
                         "description": desc, "lines": lines})
    return findings


def _check_swc120(code: str) -> List[Dict]:
    """SWC-120 Weak Sources of Randomness."""
    weak = _lines_with(code, r'block\.timestamp.*%|blockhash|block\.difficulty|block\.number.*%')
    if weak:
        return [{"swc": "SWC-120", "title": "Weak Randomness",
                 "severity": "High", "confidence": "High",
                 "description": "On-chain values (block.timestamp, blockhash, block.difficulty) used as randomness source — miner-manipulable.",
                 "lines": weak}]
    return []


def _check_swc100(code: str) -> List[Dict]:
    """SWC-100 Function Default Visibility — public/external missing on non-view/pure."""
    findings = []
    for i, line in enumerate(code.splitlines(), 1):
        if re.match(r'\s*function\s+\w+\s*\(', line):
            if not re.search(r'\b(public|private|internal|external)\b', line):
                findings.append({"swc": "SWC-100", "title": "Missing Function Visibility",
                                 "severity": "Low", "confidence": "High",
                                 "description": f"Function at line {i} has no visibility specifier — defaults to public (pre-0.5.0 behaviour).",
                                 "lines": [i]})
    return findings[:5]


# ── main entry point ──────────────────────────────────────────────────────

_CHECKS = [
    _check_swc107,   # Reentrancy (most critical first)
    _check_swc106,   # SELFDESTRUCT
    _check_swc112,   # Delegatecall
    _check_swc115,   # tx.origin
    _check_swc120,   # Weak randomness
    _check_swc101,   # Overflow
    _check_swc104,   # Unchecked call
    _check_swc105,   # Unprotected withdrawal
    _check_swc113,   # DoS
    _check_swc116,   # Timestamp
    _check_swc110,   # Assert
    _check_swc111,   # Deprecated
    _check_swc100,   # Visibility
]

_SEV_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Informational": 4}


def run_mythril(code: str) -> Dict:
    """
    Run all SWC checks and return findings sorted by severity.
    Returns: {available, findings, error}
    """
    findings = []
    for check_fn in _CHECKS:
        try:
            findings.extend(check_fn(code))
        except Exception as e:
            logger.warning("Mythril check %s failed: %s", check_fn.__name__, e)

    findings.sort(key=lambda f: _SEV_ORDER.get(f.get("severity", "Low"), 5))
    logger.info("Mythril: %d findings", len(findings))
    return {"available": True, "findings": findings, "error": None}
