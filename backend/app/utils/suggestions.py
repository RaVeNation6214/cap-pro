"""
Suggestions engine for vulnerability detection.
Maps detected vulnerabilities to severity, explanations, and fix suggestions.
"""
from typing import Dict, Optional

# Confidence thresholds for detection
THRESHOLDS = {
    "reentrancy": 0.55,
    "access_control": 0.55,
    "arithmetic": 0.50,
    "unchecked_calls": 0.50,
}

# Detailed vulnerability information
SUGGESTIONS: Dict[str, Dict[str, str]] = {
    "reentrancy": {
        "severity": "Critical",
        "explanation": (
            "External call before state update detected. An attacker can recursively "
            "call back into the contract before balances are updated, draining funds. "
            "This is the vulnerability used in the famous DAO hack ($60M stolen)."
        ),
        "suggestion": (
            "Apply the Checks-Effects-Interactions (CEI) pattern: update all state variables "
            "BEFORE making external calls. Alternatively, use OpenZeppelin's ReentrancyGuard "
            "modifier on all functions that transfer ETH or call external contracts."
        ),
        "fix_example": (
            "// SAFE pattern:\n"
            "function withdraw() public nonReentrant {\n"
            "    uint256 amount = balances[msg.sender];\n"
            "    balances[msg.sender] = 0;  // Effect FIRST\n"
            "    (bool success, ) = msg.sender.call{value: amount}(\"\");  // Interaction LAST\n"
            "    require(success, 'Transfer failed');\n"
            "}"
        ),
        "references": [
            "https://swcregistry.io/docs/SWC-107",
            "https://docs.openzeppelin.com/contracts/4.x/api/security#ReentrancyGuard",
        ],
    },
    "access_control": {
        "severity": "High",
        "explanation": (
            "Authentication or authorization vulnerability detected. Using tx.origin "
            "for access control is dangerous: an attacker can create a malicious contract "
            "that tricks a user into executing it, bypassing your access controls. "
            "Missing modifiers may allow anyone to call privileged functions."
        ),
        "suggestion": (
            "Replace tx.origin with msg.sender for all authentication checks. "
            "Use OpenZeppelin's Ownable or AccessControl libraries for role management. "
            "Add onlyOwner or custom modifier to all privileged functions."
        ),
        "fix_example": (
            "// SAFE pattern:\n"
            "import '@openzeppelin/contracts/access/Ownable.sol';\n\n"
            "contract Safe is Ownable {\n"
            "    function privilegedAction() public onlyOwner {\n"
            "        // Only owner can call\n"
            "    }\n"
            "}"
        ),
        "references": [
            "https://swcregistry.io/docs/SWC-115",
            "https://docs.openzeppelin.com/contracts/4.x/access-control",
        ],
    },
    "arithmetic": {
        "severity": "High",
        "explanation": (
            "Integer overflow or underflow vulnerability detected. In Solidity versions "
            "before 0.8.0, arithmetic operations silently wrap around. For example, "
            "uint8(255) + 1 = 0, causing token balances to overflow to unexpected values. "
            "Attackers can exploit this to mint arbitrary tokens or bypass checks."
        ),
        "suggestion": (
            "Upgrade to Solidity 0.8.0+ which has built-in overflow/underflow protection. "
            "For older code, use OpenZeppelin's SafeMath library for all arithmetic. "
            "Use unchecked{} blocks only when you explicitly want wrap-around behavior."
        ),
        "fix_example": (
            "// Solidity 0.8+ (automatic checks):\n"
            "pragma solidity ^0.8.0;\n\n"
            "// OR use SafeMath for older versions:\n"
            "using SafeMath for uint256;\n"
            "uint256 result = a.add(b);  // Reverts on overflow"
        ),
        "references": [
            "https://swcregistry.io/docs/SWC-101",
            "https://docs.openzeppelin.com/contracts/4.x/api/utils#SafeMath",
        ],
    },
    "unchecked_calls": {
        "severity": "Medium",
        "explanation": (
            "External call return value not checked. The low-level .call(), .send(), "
            "and .delegatecall() return a boolean indicating success/failure. "
            "If not checked, the contract will silently continue on failure, "
            "potentially leaving funds locked or state inconsistent."
        ),
        "suggestion": (
            "Always check return values: require(success, 'Call failed'). "
            "Prefer .transfer() over .send() as it reverts on failure. "
            "Use OpenZeppelin's Address.sendValue() for ETH transfers. "
            "Avoid .delegatecall() to untrusted contracts."
        ),
        "fix_example": (
            "// SAFE pattern:\n"
            "(bool success, bytes memory data) = target.call{value: amount}(\"\");\n"
            "require(success, 'External call failed');\n\n"
            "// OR use Address library:\n"
            "import '@openzeppelin/contracts/utils/Address.sol';\n"
            "Address.sendValue(payable(recipient), amount);"
        ),
        "references": [
            "https://swcregistry.io/docs/SWC-104",
            "https://docs.openzeppelin.com/contracts/4.x/api/utils#Address",
        ],
    },
}


def get_suggestion(vuln_type: str) -> Dict[str, str]:
    """
    Get suggestion details for a vulnerability type.

    Args:
        vuln_type: One of 'reentrancy', 'access_control', 'arithmetic', 'unchecked_calls'

    Returns:
        Dict with severity, explanation, suggestion, fix_example, references
    """
    key = vuln_type.lower().replace(' ', '_').replace('-', '_')
    return SUGGESTIONS.get(key, {
        "severity": "Unknown",
        "explanation": "Unknown vulnerability type.",
        "suggestion": "Consult the SWC Registry for guidance.",
        "fix_example": "",
        "references": ["https://swcregistry.io/"],
    })


def is_detected(probability: float, vuln_type: str) -> bool:
    """Check if vulnerability is detected based on threshold."""
    threshold = THRESHOLDS.get(vuln_type.lower().replace(' ', '_'), 0.5)
    return probability > threshold


def severity_to_score(severity: str) -> float:
    """Convert severity string to numeric weight."""
    return {
        "Critical": 1.0,
        "High": 0.8,
        "Medium": 0.6,
        "Low": 0.4,
        "None": 0.0,
    }.get(severity, 0.5)
