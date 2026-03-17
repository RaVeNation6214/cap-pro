"""
Slither static analysis integration — uses the Python API directly.
Temp files are written to the current working directory to avoid Windows path issues.
"""
import os
import tempfile
import logging
import inspect
from typing import List, Dict

logger = logging.getLogger(__name__)

# Add Scripts dir to PATH so solc.exe is found
_SCRIPTS = r'C:\Users\aksha\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\Scripts'
if os.path.isdir(_SCRIPTS) and _SCRIPTS not in os.environ.get('PATH', ''):
    os.environ['PATH'] = _SCRIPTS + os.pathsep + os.environ.get('PATH', '')

try:
    from slither import Slither
    from slither.detectors import all_detectors
    from slither.detectors.abstract_detector import AbstractDetector, DetectorClassification
    _ALL_DETECTORS = [
        c for _, c in inspect.getmembers(all_detectors, inspect.isclass)
        if issubclass(c, AbstractDetector) and c is not AbstractDetector
    ]
    SLITHER_AVAILABLE = True
    logger.info("Slither available: True  (%d detectors)", len(_ALL_DETECTORS))
except Exception as e:
    SLITHER_AVAILABLE = False
    _ALL_DETECTORS = []
    logger.warning("Slither unavailable: %s", e)


def run_slither(code: str) -> Dict:
    """
    Run Slither Python API on Solidity source.
    Returns: {available, findings, raw_output, error}
    """
    if not SLITHER_AVAILABLE:
        logger.warning("Slither not available — no static analysis performed")
        return {"available": False, "findings": [], "raw_output": "", "error": "slither not installed"}

    tmp_path = None
    try:
        # Write temp file to cwd — Windows path fix
        fd, tmp_path = tempfile.mkstemp(suffix=".sol", dir=".", text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(code)

        sl = Slither(tmp_path)
        for d_cls in _ALL_DETECTORS:
            try:
                sl.register_detector(d_cls)
            except Exception:
                pass

        raw_results = sl.run_detectors()

        findings = []
        seen = set()
        for group in raw_results:
            for item in group:
                if not item:
                    continue
                check = item.get("check", "unknown")
                desc  = item.get("description", "")
                impact = item.get("impact", "Informational")
                conf   = item.get("confidence", "Medium")

                # Deduplicate
                key = (check, desc[:60])
                if key in seen:
                    continue
                seen.add(key)

                lines = []
                for elem in item.get("elements", []):
                    sm = elem.get("source_mapping", {})
                    lines.extend(sm.get("lines", []))

                findings.append({
                    "check": check,
                    "impact": impact,
                    "confidence": conf,
                    "description": desc.strip(),
                    "lines": sorted(set(lines)),
                })

        # Sort by severity
        _order = {"High": 0, "Medium": 1, "Low": 2, "Informational": 3, "Optimization": 4}
        findings.sort(key=lambda f: _order.get(f["impact"], 5))

        logger.info("Slither: %d findings", len(findings))
        return {"available": True, "findings": findings, "raw_output": "", "error": None}

    except Exception as e:
        logger.error("Slither run failed: %s", e)
        return {"available": True, "findings": [], "raw_output": "", "error": str(e)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def build_comparison(model_vulns: List[str], slither_result: Dict) -> Dict:
    """Build a comparison object between model and slither findings."""
    slither_checks = {f["check"].lower() for f in slither_result.get("findings", [])}

    vuln_map = {
        "reentrancy":       {"reentrancy-eth", "reentrancy-no-eth", "reentrancy-benign", "reentrancy-events"},
        "access control":   {"tx-origin", "access-control"},
        "arithmetic":       {"integer-overflow", "tautology", "divide-before-multiply", "weak-prng"},
        "unchecked calls":  {"unchecked-lowlevel", "unchecked-send", "unchecked-transfer", "low-level-calls"},
        "timestamp":        {"timestamp", "block-timestamp"},
    }

    agreement, model_only, slither_only = [], [], []

    for mv in model_vulns:
        keys = vuln_map.get(mv.lower(), set())
        if keys & slither_checks:
            agreement.append(mv)
        else:
            model_only.append(mv)

    covered = set()
    for mv in model_vulns:
        covered |= vuln_map.get(mv.lower(), set())

    for sc in slither_checks:
        if sc not in covered:
            slither_only.append(sc)

    if len(agreement) >= max(len(model_only), len(slither_only)):
        winner = "tie"
    elif len(model_only) > len(slither_only):
        winner = "model"
    else:
        winner = "slither"

    return {
        "slither_available": slither_result.get("available", False),
        "slither_findings":  slither_result.get("findings", []),
        "model_findings":    model_vulns,
        "agreement":         agreement,
        "model_only":        model_only,
        "slither_only":      slither_only,
        "winner":            winner,
    }
