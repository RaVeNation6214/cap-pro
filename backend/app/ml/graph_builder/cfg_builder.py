"""
CFG/AST Builder for Solidity smart contracts.
Builds a control flow graph from Solidity source code.
Used as input to the Graph Attention Network.
"""
import re
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class ContractGraph:
    """Represents a contract as a graph."""
    nodes: List[str] = field(default_factory=list)          # Node names (functions, etc.)
    node_features: List[List[float]] = field(default_factory=list)  # Per-node features
    edges: List[Tuple[int, int]] = field(default_factory=list)      # (src, dst) edges
    labels: Optional[List[float]] = None                   # Vulnerability labels if known

    def num_nodes(self) -> int:
        return len(self.nodes)

    def num_edges(self) -> int:
        return len(self.edges)


class CFGBuilder:
    """
    Builds a simplified Control Flow Graph from Solidity source code.

    Each function becomes a node with feature vector derived from:
    - Cyclomatic complexity (number of branches)
    - External call count (.call, .send, .transfer, .delegatecall)
    - State variable access count
    - Use of tx.origin, block.timestamp
    - Presence of arithmetic ops without SafeMath

    Edges represent:
    - Function calls between functions
    - Inheritance relationships
    - Fallback/receive triggers
    """

    # Patterns for detecting vulnerability-related code
    EXTERNAL_CALL_PATTERN = re.compile(
        r'\.(call|send|transfer|delegatecall|staticcall)\s*[\(\{]',
        re.IGNORECASE
    )
    ARITHMETIC_PATTERN = re.compile(r'[\+\-\*\/]\s*(?!=)')
    STATE_MOD_PATTERN = re.compile(r'\b\w+\s*[\+\-\*\/]?=(?!=)')
    TX_ORIGIN_PATTERN = re.compile(r'\btx\.origin\b')
    TIMESTAMP_PATTERN = re.compile(r'\bblock\.timestamp\b|\bnow\b')
    REQUIRE_PATTERN = re.compile(r'\brequire\s*\(')
    EMIT_PATTERN = re.compile(r'\bemit\b')
    LOOP_PATTERN = re.compile(r'\b(for|while|do)\b')
    SAFEMATCH_PATTERN = re.compile(r'\bSafeMath\b|pragma solidity \^0\.8')

    def __init__(self):
        pass

    def parse_functions(self, code: str) -> List[Dict]:
        """
        Parse Solidity code and extract function definitions with their bodies.
        Returns list of dicts with 'name' and 'body'.
        """
        functions = []

        # Match function definitions including modifiers, constructors, fallback/receive
        func_pattern = re.compile(
            r'(function\s+(\w+)|constructor|fallback|receive)\s*'
            r'\([^)]*\)[^{]*\{',
            re.DOTALL
        )

        for match in func_pattern.finditer(code):
            name_match = re.match(r'function\s+(\w+)', match.group(0))
            if name_match:
                func_name = name_match.group(1)
            elif 'constructor' in match.group(0):
                func_name = '__constructor__'
            elif 'fallback' in match.group(0):
                func_name = '__fallback__'
            elif 'receive' in match.group(0):
                func_name = '__receive__'
            else:
                continue

            # Extract function body by counting braces
            start = match.end()
            body = self._extract_body(code, start - 1)  # include opening brace
            functions.append({'name': func_name, 'body': body, 'start': match.start()})

        # If no functions found (very small contract), treat whole code as one node
        if not functions:
            functions.append({'name': '__contract__', 'body': code, 'start': 0})

        return functions

    def _extract_body(self, code: str, open_brace_pos: int) -> str:
        """Extract function body by counting matching braces."""
        depth = 0
        start = open_brace_pos
        for i in range(open_brace_pos, len(code)):
            if code[i] == '{':
                depth += 1
            elif code[i] == '}':
                depth -= 1
                if depth == 0:
                    return code[start:i + 1]
        return code[open_brace_pos:]

    def extract_node_features(self, func_body: str, has_safematch: bool) -> List[float]:
        """
        Extract a 12-dimensional feature vector for a function node.
        """
        body = func_body

        # 1. External call count
        ext_calls = len(self.EXTERNAL_CALL_PATTERN.findall(body))

        # 2. Arithmetic operation count
        arith_ops = len(self.ARITHMETIC_PATTERN.findall(body))

        # 3. State modification count
        state_mods = len(self.STATE_MOD_PATTERN.findall(body))

        # 4. Uses tx.origin
        uses_tx_origin = 1.0 if self.TX_ORIGIN_PATTERN.search(body) else 0.0

        # 5. Uses block.timestamp
        uses_timestamp = 1.0 if self.TIMESTAMP_PATTERN.search(body) else 0.0

        # 6. Require/assert count
        require_count = len(self.REQUIRE_PATTERN.findall(body))

        # 7. Loop count (complexity indicator)
        loop_count = len(self.LOOP_PATTERN.findall(body))

        # 8. Uses SafeMath / Solidity 0.8+
        safe_math = 1.0 if has_safematch else 0.0

        # 9. Function length (normalized to 0-1 range, capped at 200 lines)
        lines = len(body.split('\n'))
        func_length = min(lines / 200.0, 1.0)

        # 10. Sends ETH (payable call or transfer)
        sends_eth = 1.0 if re.search(r'\.transfer\s*\(|\.send\s*\(|\.call\s*\{.*value', body) else 0.0

        # 11. Has emit (event logging - good security practice)
        has_emit = 1.0 if self.EMIT_PATTERN.search(body) else 0.0

        # 12. Normalized external call ratio (ext_calls / max(lines,1))
        call_ratio = min(ext_calls / max(lines, 1), 1.0)

        return [
            min(ext_calls / 10.0, 1.0),
            min(arith_ops / 20.0, 1.0),
            min(state_mods / 10.0, 1.0),
            uses_tx_origin,
            uses_timestamp,
            min(require_count / 5.0, 1.0),
            min(loop_count / 5.0, 1.0),
            safe_math,
            func_length,
            sends_eth,
            has_emit,
            call_ratio,
        ]

    def build_call_graph(self, functions: List[Dict]) -> List[Tuple[int, int]]:
        """
        Build edges between functions based on internal calls.
        Returns list of (src_idx, dst_idx) pairs.
        """
        edges = []
        func_names = [f['name'] for f in functions]

        for i, caller in enumerate(functions):
            caller_body = caller['body']
            for j, callee in enumerate(functions):
                if i == j:
                    continue
                callee_name = callee['name']
                if callee_name.startswith('__'):
                    continue
                # Check if caller calls callee
                call_pattern = re.compile(r'\b' + re.escape(callee_name) + r'\s*\(')
                if call_pattern.search(caller_body):
                    edges.append((i, j))

        # If no edges, create a chain to ensure graph connectivity
        if not edges and len(functions) > 1:
            for i in range(len(functions) - 1):
                edges.append((i, i + 1))

        return edges

    def build(self, code: str, labels: Optional[List[float]] = None) -> ContractGraph:
        """
        Build a ContractGraph from Solidity source code.

        Args:
            code: Solidity source code string
            labels: Optional vulnerability labels [arith, access, unchecked, reentrancy]

        Returns:
            ContractGraph with nodes, features, edges, and optional labels
        """
        has_safematch = bool(self.SAFEMATCH_PATTERN.search(code))

        # Parse functions
        functions = self.parse_functions(code)

        # Extract node features
        node_features = []
        for func in functions:
            features = self.extract_node_features(func['body'], has_safematch)
            node_features.append(features)

        # Build call graph edges
        edges = self.build_call_graph(functions)

        # Ensure at least one self-loop (prevents empty edge_index)
        if not edges:
            edges = [(0, 0)]

        return ContractGraph(
            nodes=[f['name'] for f in functions],
            node_features=node_features,
            edges=edges,
            labels=labels,
        )

    def to_pyg_data(self, graph: ContractGraph, node_embedding_dim: int = 768):
        """
        Convert ContractGraph to PyTorch Geometric Data object.
        Requires torch and torch_geometric.
        """
        import torch
        from torch_geometric.data import Data

        # Node features (12-dim static) padded/projected to embedding_dim
        x = torch.tensor(graph.node_features, dtype=torch.float)  # [N, 12]

        # Edge index
        if graph.edges:
            edge_index = torch.tensor(graph.edges, dtype=torch.long).t().contiguous()  # [2, E]
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Labels
        y = torch.tensor(graph.labels, dtype=torch.float) if graph.labels else None

        return Data(x=x, edge_index=edge_index, y=y)
