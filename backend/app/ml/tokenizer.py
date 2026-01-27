import re
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter


class SolidityTokenizer:
    """
    Tokenizer for Solidity smart contract code.
    Handles keywords, identifiers, operators, and special tokens.
    """

    # Solidity keywords
    KEYWORDS = {
        'pragma', 'solidity', 'contract', 'interface', 'library', 'abstract',
        'function', 'modifier', 'event', 'struct', 'enum', 'mapping',
        'public', 'private', 'internal', 'external', 'pure', 'view', 'payable',
        'constant', 'immutable', 'storage', 'memory', 'calldata',
        'if', 'else', 'for', 'while', 'do', 'break', 'continue', 'return',
        'try', 'catch', 'revert', 'require', 'assert',
        'new', 'delete', 'this', 'super', 'selfdestruct',
        'true', 'false', 'wei', 'gwei', 'ether', 'seconds', 'minutes', 'hours', 'days', 'weeks',
        'uint', 'uint8', 'uint16', 'uint32', 'uint64', 'uint128', 'uint256',
        'int', 'int8', 'int16', 'int32', 'int64', 'int128', 'int256',
        'bool', 'address', 'string', 'bytes', 'bytes32',
        'msg', 'sender', 'value', 'data', 'sig',
        'block', 'timestamp', 'number', 'gaslimit',
        'tx', 'origin', 'gasprice',
        'abi', 'encode', 'decode', 'encodePacked', 'encodeWithSelector',
        'keccak256', 'sha256', 'ripemd160', 'ecrecover',
        'call', 'delegatecall', 'staticcall', 'send', 'transfer',
        'balance', 'code', 'codehash',
        'emit', 'indexed', 'anonymous',
        'override', 'virtual', 'using', 'is', 'as', 'import', 'from'
    }

    # Special tokens
    SPECIAL_TOKENS = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<CLS>': 2,
        '<SEP>': 3,
        '<MASK>': 4,
        '<NUM>': 5,
        '<ADDR>': 6,
        '<STR>': 7,
    }

    # Token patterns
    PATTERNS = [
        (r'0x[a-fA-F0-9]+', '<ADDR>'),  # Hex address
        (r'\d+', '<NUM>'),  # Numbers
        (r'"[^"]*"', '<STR>'),  # Double-quoted strings
        (r"'[^']*'", '<STR>'),  # Single-quoted strings
        (r'//.*', None),  # Single-line comments (remove)
        (r'/\*[\s\S]*?\*/', None),  # Multi-line comments (remove)
        (r'[a-zA-Z_][a-zA-Z0-9_]*', 'IDENTIFIER'),  # Identifiers
        (r'[+\-*/%&|^~<>=!]+', 'OPERATOR'),  # Operators
        (r'[{}()\[\];,.:?]', 'PUNCTUATION'),  # Punctuation
    ]

    def __init__(self, vocab_size: int = 10000, vocab_path: Optional[Union[str, Path]] = None):
        """
        Initialize tokenizer.

        Args:
            vocab_size: Maximum vocabulary size when building from scratch
            vocab_path: Path to pre-built vocabulary file (.pkl)
        """
        self.vocab_size = vocab_size
        self.word2idx: Dict[str, int] = dict(self.SPECIAL_TOKENS)
        self.idx2word: Dict[int, str] = {v: k for k, v in self.SPECIAL_TOKENS.items()}
        self.vocab_built = False

        if vocab_path is not None:
            self.load_vocab(vocab_path)

    def _preprocess(self, code: str) -> str:
        """Remove comments and normalize whitespace."""
        # Remove single-line comments
        code = re.sub(r'//.*', '', code)
        # Remove multi-line comments
        code = re.sub(r'/\*[\s\S]*?\*/', '', code)
        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code)
        return code.strip()

    def _tokenize_raw(self, code: str) -> List[str]:
        """Tokenize code into raw tokens."""
        code = self._preprocess(code)
        tokens = []

        # Pattern for tokenization
        token_pattern = re.compile(
            r'0x[a-fA-F0-9]+|'  # Hex addresses
            r'\d+|'  # Numbers
            r'"[^"]*"|'  # Double-quoted strings
            r"'[^']*'|"  # Single-quoted strings
            r'[a-zA-Z_][a-zA-Z0-9_]*|'  # Identifiers
            r'[+\-*/%&|^~<>=!]+|'  # Operators
            r'[{}()\[\];,.:?]'  # Punctuation
        )

        for match in token_pattern.finditer(code):
            token = match.group()

            # Classify and normalize token
            if re.match(r'^0x[a-fA-F0-9]+$', token):
                tokens.append('<ADDR>')
            elif re.match(r'^\d+$', token):
                tokens.append('<NUM>')
            elif re.match(r'^["\']', token):
                tokens.append('<STR>')
            elif token.lower() in self.KEYWORDS:
                tokens.append(token.lower())
            else:
                tokens.append(token)

        return tokens

    def build_vocab(self, contracts: List[str]) -> None:
        """Build vocabulary from a list of contracts."""
        counter = Counter()

        for contract in contracts:
            tokens = self._tokenize_raw(contract)
            counter.update(tokens)

        # Add most common tokens to vocabulary
        idx = len(self.SPECIAL_TOKENS)
        for token, _ in counter.most_common(self.vocab_size - len(self.SPECIAL_TOKENS)):
            if token not in self.word2idx:
                self.word2idx[token] = idx
                self.idx2word[idx] = token
                idx += 1

        self.vocab_built = True

    def load_vocab(self, vocab_path: Union[str, Path]) -> None:
        """
        Load vocabulary from a pickle file.

        The file should contain either:
        - A dict mapping tokens to indices
        - A list of tokens (indices will be assigned in order)
        """
        vocab_path = Path(vocab_path)
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)

        # Reset vocabulary to special tokens
        self.word2idx = dict(self.SPECIAL_TOKENS)
        self.idx2word = {v: k for k, v in self.SPECIAL_TOKENS.items()}

        if isinstance(vocab_data, dict):
            # Vocab is already a mapping
            # Shift indices to account for special tokens
            offset = len(self.SPECIAL_TOKENS)
            for token, idx in vocab_data.items():
                if token not in self.word2idx:
                    new_idx = idx + offset
                    self.word2idx[token] = new_idx
                    self.idx2word[new_idx] = token
        elif isinstance(vocab_data, list):
            # Vocab is a list of tokens
            idx = len(self.SPECIAL_TOKENS)
            for token in vocab_data:
                if token not in self.word2idx:
                    self.word2idx[token] = idx
                    self.idx2word[idx] = token
                    idx += 1
        else:
            raise ValueError(f"Unexpected vocabulary format: {type(vocab_data)}")

        self.vocab_built = True
        self.vocab_size = len(self.word2idx)

    def save_vocab(self, vocab_path: Union[str, Path]) -> None:
        """Save vocabulary to a pickle file."""
        vocab_path = Path(vocab_path)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as dict without special tokens (they're added on load)
        vocab_to_save = {
            token: idx - len(self.SPECIAL_TOKENS)
            for token, idx in self.word2idx.items()
            if token not in self.SPECIAL_TOKENS
        }

        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab_to_save, f)

    def encode(self, code: str, max_length: int = 512) -> List[int]:
        """Encode code into token IDs."""
        tokens = self._tokenize_raw(code)

        # Convert to IDs
        ids = [self.SPECIAL_TOKENS['<CLS>']]
        for token in tokens[:max_length - 2]:
            ids.append(self.word2idx.get(token, self.SPECIAL_TOKENS['<UNK>']))
        ids.append(self.SPECIAL_TOKENS['<SEP>'])

        # Pad if necessary
        while len(ids) < max_length:
            ids.append(self.SPECIAL_TOKENS['<PAD>'])

        return ids[:max_length]

    def decode(self, ids: List[int]) -> List[str]:
        """Decode token IDs back to tokens."""
        return [self.idx2word.get(idx, '<UNK>') for idx in ids]

    def tokenize_to_windows(self, code: str, window_size: int = 3) -> List[Tuple[List[int], int, int]]:
        """
        Split code into overlapping windows of lines.
        Returns list of (token_ids, start_line, end_line).
        """
        lines = code.split('\n')
        windows = []

        for i in range(len(lines)):
            # Get window of lines
            start = max(0, i - window_size // 2)
            end = min(len(lines), start + window_size)

            # Adjust start if we're at the end
            if end - start < window_size and start > 0:
                start = max(0, end - window_size)

            window_code = '\n'.join(lines[start:end])
            token_ids = self.encode(window_code)
            windows.append((token_ids, start + 1, end))  # 1-indexed lines

        return windows

    def get_vocab_size(self) -> int:
        """Return the current vocabulary size."""
        return len(self.word2idx)

    def encode_tokens(self, tokens: List[str], max_length: int = 512) -> List[int]:
        """
        Encode a pre-tokenized list of tokens into IDs.

        Args:
            tokens: List of token strings
            max_length: Maximum sequence length

        Returns:
            List of token IDs with CLS, SEP, and padding
        """
        ids = [self.SPECIAL_TOKENS['<CLS>']]
        for token in tokens[:max_length - 2]:
            ids.append(self.word2idx.get(token, self.SPECIAL_TOKENS['<UNK>']))
        ids.append(self.SPECIAL_TOKENS['<SEP>'])

        # Pad if necessary
        while len(ids) < max_length:
            ids.append(self.SPECIAL_TOKENS['<PAD>'])

        return ids[:max_length]

    def encode_window_sequence(
        self,
        window_tokens: List[List[str]],
        max_seq_len: int = 128
    ) -> List[List[int]]:
        """
        Encode a list of windows (each window is a list of tokens).

        Args:
            window_tokens: List of windows, each containing token strings
            max_seq_len: Maximum length per window

        Returns:
            List of token ID sequences for each window
        """
        return [self.encode_tokens(tokens, max_seq_len) for tokens in window_tokens]

    @classmethod
    def from_dataset(cls, data_dir: Union[str, Path]) -> 'SolidityTokenizer':
        """
        Create a tokenizer from the newALLBUGS dataset vocabulary.

        Args:
            data_dir: Path to newALLBUGS directory containing vocab.pkl

        Returns:
            SolidityTokenizer with loaded vocabulary
        """
        data_dir = Path(data_dir)
        vocab_path = data_dir / 'vocab.pkl'

        if not vocab_path.exists():
            raise FileNotFoundError(
                f"vocab.pkl not found in {data_dir}. "
                "Make sure you have the newALLBUGS dataset."
            )

        return cls(vocab_path=vocab_path)
