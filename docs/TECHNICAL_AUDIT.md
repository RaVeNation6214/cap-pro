# Technical Audit: Smart Contract Vulnerability Detector (cap-pro)

**Auditor perspective:** ML engineer + smart contract auditor  
**Scope:** Pipeline, ML architecture, Solidity analysis, dataset, effectiveness  
**Verdict:** See final section.

---

## 1. High-Level Pipeline (From Scratch)

### 1.1 End-to-end flow

```
User (paste/upload Solidity)
    → Frontend POST /api/analyze { code }
    → Backend: DEMO_MODE ? DemoModeAnalyzer.analyze(code) : GNN path
```

**GNN path (when `DEMO_MODE=False` and `models/hybrid_gnn.pt` exists):**

1. **CFGBuilder.build(code)**  
   Regex-based Solidity parser → list of functions (name + body), 12-d static features per function, call-graph edges.

2. **CodeEmbedder**  
   Each function body (truncated to 1000 chars) → GraphCodeBERT CLS → 768-d vector (or fallback: keyword-frequency vector padded to 768).

3. **HybridGNN / SimpleMLP**  
   - **HybridGNN:** Node features = 768-d embeddings; static = mean of 12-d per-node features. GAT(256, 4 heads) → GAT(256) → global mean pool → concat(256 + 12) → MLP → 5 logits.  
   - **SimpleMLP:** Input = mean of 12-d static features only → MLP → 5 logits.

4. **Sigmoid**  
   5 probabilities (reentrancy, arithmetic, access_control, unchecked_calls, timestamp).

5. **DemoModeAnalyzer.analyze_with_probs(code, probs)**  
   Same response shape as demo: line_risks (pattern-based attribution), attention_weights, summary, recommendations. **Line-level attribution is still regex/pattern-based, not from the GNN.**

**Demo path (default, `DEMO_MODE=True`):**

1. **FeatureExtractor.extract(code)**  
   12-d static features (call/send/transfer/delegatecall, tx.origin, arithmetic, state writes, require/assert, reentrancy/unchecked patterns).

2. **DemoModeAnalyzer**  
   Many regex patterns per vulnerability type; weighted score accumulation; safe-pattern dampening (ReentrancyGuard, SafeMath, msg.sender); line attribution; overall risk and recommendations.

So: **two completely separate “engines”—regex demo vs. GNN—with a unified API response.**

---

## 2. ML Architecture in Depth

### 2.1 Graph representation (CFGBuilder)

- **Nodes:** One per function (including `constructor`, `fallback`, `receive`). If no function is found, the whole file is one node `__contract__`.
- **Edges:** Call graph: function A’s body contains `B(` → edge (A, B). If there are no edges, a chain (0→1→2→…) or self-loop (0→0) is added so the graph is never empty.
- **Parsing:** Pure regex (no Solidity compiler):
  - Function detection: `(function \w+|constructor|fallback|receive)\s*\([^)]*\)[^{]*\{`
  - Body: brace-counting from `{` to matching `}`.

**Limitations:**

- Regex can break on nested braces in strings/comments, unusual formatting, or inline assembly.
- No real AST → no type info, no data-flow; only structural and keyword-based signals.

### 2.2 Node features (12-d, per function)

| # | Feature | Rationale |
|---|--------|-----------|
| 1 | External call count (cap 10) | Reentrancy / unchecked calls |
| 2 | Arithmetic op count (cap 20) | Overflow/underflow |
| 3 | State modification count (cap 10) | Reentrancy (state after call) |
| 4 | Uses tx.origin (0/1) | Access control |
| 5 | Uses block.timestamp/now (0/1) | Timestamp dependence |
| 6 | require/assert count (cap 5) | Validation strength |
| 7 | Loop count (cap 5) | Complexity |
| 8 | SafeMath / pragma ^0.8 (0/1) | Arithmetic mitigation |
| 9 | Function length (cap 200 lines) | Complexity |
| 10 | Sends ETH (0/1) | Reentrancy / value flow |
| 11 | Has emit (0/1) | Logging |
| 12 | Call ratio (calls/lines, cap 1) | Call density |

All counts are normalized to [0,1]. Contract-level static input to the classifier is the **mean** of these 12-d vectors over nodes.

### 2.3 Embedding (CodeEmbedder)

- **Primary:** `microsoft/graphcodebert-base` (CodeBERT variant). Input: function body, max_length=512. Output: CLS token → 768-d. Used with `torch.no_grad()` (frozen).
- **Fallback:** If transformers/GraphCodeBERT fails: ~32 keyword counts + 2 line/code-size features, then padded to 768 with a sinusoidal pattern. Not comparable to real code semantics.

### 2.4 HybridGNN (gnn_model.py)

- **Node projection:** Linear(768→256) + LayerNorm + ReLU + Dropout(0.1).
- **GAT:**  
  - Layer 1: GATConv(256 → 256, heads=4) → 256×4 = 1024 → ELU, dropout.  
  - Layer 2: GATConv(1024 → 256, heads=1) → 256 → ELU, dropout.  
- **Readout:** global_mean_pool → one 256-d vector per contract.
- **Classifier:** Concat(256 + 12) = 268 → Linear(268→128) → LayerNorm, ReLU, Dropout → Linear(128→5). Output: 5 logits; loss is BCEWithLogitsLoss (multi-label).

**SimpleMLP fallback:** 12-d static (mean) → 128 → ReLU → 64 → 5. Used when torch_geometric or GraphCodeBERT is missing.

### 2.5 Training (training/train.py)

- **Data:** CSV files: `train.csv`, `val.csv`, `test.csv` with columns `code` (or `path`), `reentrancy`, `arithmetic`, `access_control`, `unchecked_calls`, `timestamp` (binary 0/1).
- **Pipeline:** For each contract: CFGBuilder → graph; GraphCodeBERT embed each function body → node features; mean of node 12-d → static; Batch of PyG Data.
- **Training:** AdamW (lr=1e-4), CosineAnnealingLR, BCEWithLogitsLoss with fixed class weights [2.0, 2.5, 2.5, 1.5, 3.0], grad clip 1.0, early stopping (patience=3), best F1 checkpoint to `models/hybrid_gnn.pt`.
- **Note:** `data/build_dataset.py` is referenced in logs but is **not in the repo**. So you cannot reproduce the CSV splits from raw data without writing this script (e.g. from newALLBUGS or SmartBugs).

---

## 3. Dataset(s) in Depth

### 3.1 newALLBUGS (dataset.py)

- **Purpose:** Used by `VulnerabilityDataset` and `WindowDataset` (hierarchical/transformer training in `app/ml/trainer.py`). **Not** used by the GNN training script `training/train.py`, which uses CSV.
- **Layout:**
  - `contract/<address>.sol` (or `<address>_1.sol`, …) — full source.
  - `threelines-tokenseq/<address>.sol` — 3-line tokenized window around fault line.
  - `ast/<address>.sol` — AST token sequence.
  - `pretrain_label/label190.pkl` — dict: contract address → list of vulnerability IDs (integer).
  - `code_w2i.pkl`, `code_i2w.pkl` — vocabulary (word↔index).

**Vulnerability ID → class (4 classes in dataset.py):**

- Arithmetic: 31, 32, 36, 89  
- Access Control: 39  
- Unchecked Calls: 40, 42, 43, 82  
- Reentrancy: 41  

**Important:** newALLBUGS has **4 classes** and **no “timestamp”** label. The live GNN uses **5 classes** (reentrancy, arithmetic, access_control, unchecked_calls, **timestamp**). So:

- Timestamp is **only** learned from the CSV training data (if present there).
- If CSV was built from newALLBUGS-only, timestamp would be all zeros → model cannot learn timestamp from that dataset.

### 3.2 CSV splits (training/train.py)

- **Expected columns:** `code` (or `path`), `reentrancy`, `arithmetic`, `access_control`, `unchecked_calls`, `timestamp`.
- **Format:** Binary multi-label per row. No schema or sample provided in repo; `build_dataset.py` is missing.
- **Implications:** Without that script or the actual CSVs, we cannot verify balance, leakage, or whether timestamp (or other classes) have enough positives.

### 3.3 What the dataset “looks like”

- **newALLBUGS:** ~190 labeled contracts (from label190.pkl), filenames by address; multi-label; 4 classes; fault windows and AST tokens are optional.
- **CSV (intended):** One row per contract; Solidity in `code` or path; five binary columns. Unknown size and distribution.

---

## 4. Solidity Analysis in Depth

### 4.1 Static analysis

- **CFGBuilder:** Regex-based function extraction + 12-d heuristic features. No solc, no AST, no data-flow.
- **FeatureExtractor (features.py):** Same idea at contract level: booleans and counts (call/send/transfer/delegatecall, tx.origin, arithmetic, state writes, require/assert, reentrancy/unchecked patterns). Used by demo and for line-level features.
- **Slither (slither_features.py):** Optional. If Slither is installed, it runs on a temp file and detector counts/severity are read. There is a regex fallback. Not wired into the GNN path; health check reports `slither_available`.

### 4.2 Demo mode (DemoModeAnalyzer)

- Large list of **VulnerabilityPattern** (regex + type + weight + description).
- For each pattern: find matches in code, accumulate weighted scores per type, attribute to line numbers.
- **Safe-pattern dampening:** ReentrancyGuard/nonReentrant → reentrancy ×0.2; SafeMath/pragma ^0.8 → arithmetic ×0.3; msg.sender without tx.origin → access_control ×0.3.
- Line risks and “attention” are derived from these matches + weights; summary and recommendations are template-based from vulnerability type.

So Solidity “analysis” here is **pattern + heuristics**, not formal verification or true data-flow. It can catch classic patterns (e.g. tx.origin, call before state update) but can miss contextual or cross-contract issues and can false-positive on benign code.

---

## 5. Does It Work Well? Effectiveness and Gaps

### 5.1 Strengths

- **Clear separation:** Demo (regex) vs. GNN path; fallbacks (SimpleMLP, keyword embedding) when deps are missing.
- **Unified API:** Same AnalysisResult shape for both modes; frontend does not care which backend ran.
- **Sensible ML design:** Graph over functions, GraphCodeBERT for code, GAT for propagation, static features alongside embeddings, multi-label BCE with class weights.
- **Operational care:** DEMO_MODE default True so the app runs without a trained model; health endpoint reports model/slither/gemini; lazy loading of heavy components.

### 5.2 Weaknesses and risks

1. **No real Solidity parsing**  
   Regex-based parsing and features can fail or misparse valid Solidity (strings, comments, assembly). No type or data-flow information.

2. **Dataset / training pipeline incomplete**  
   - newALLBUGS is 4-class; GNN is 5-class (timestamp).  
   - CSV pipeline depends on a missing `build_dataset.py`.  
   - No evidence that timestamp (or other rare classes) are present in the training data in a balanced way.

3. **Line-level attribution is not from the model**  
   Even in GNN mode, affected lines and line_risks come from the same regex patterns as demo. The GNN only contributes contract-level probabilities.

4. **Slither not in the model**  
   Slither is only used for health and optional features; it is not part of the GNN input or training.

5. **API inconsistencies**  
   - `/vulnerability-classes` returns 4 classes (no timestamp); model and config use 5.  
   - `suggestions.py` has no entry for `timestamp`; GNN and demo both use timestamp.

6. **Evaluation unknown**  
   No published metrics (precision/recall/F1 per class) or test set results in the repo. Training script logs val F1 and can run on test.csv if present, but we don’t know baseline or threshold tuning.

7. **Model-info typo**  
   Fixed in routes: classifier output was described as “4” instead of “5” classes.

### 5.3 Verdict

- **Technically:** The pipeline is coherent and the ML design (CFG + BERT + GAT + static, multi-label) is sound for a **research-style** tool. Code quality is decent; fallbacks and config are thought through.
- **Practically:**  
  - **Demo mode:** Works as a pattern-based scanner; good for teaching and quick checks; not an audit replacement.  
  - **GNN mode:** Depends on having a proper 5-class CSV dataset and a trained checkpoint. Without `build_dataset.py` and without knowing the label distribution (especially timestamp), we cannot say if the model is “effective” in production.  
- **As an “audit” tool:** It is a **vulnerability detector** (pattern + ML), not a full audit (no formal verification, no full data-flow, no proof of safety). Use it as an **assistant** to flag candidates and explain common issues, not as the sole authority.

---

## 6. Summary Table

| Component | What it is | Does it work? |
|-----------|------------|----------------|
| Pipeline | Frontend → /api/analyze → Demo or GNN → AnalysisResult | Yes |
| CFGBuilder | Regex Solidity → functions, 12-d features, call graph | Yes for typical code; fragile on edge cases |
| GraphCodeBERT | 768-d per function | Yes if transformers available |
| HybridGNN | GAT + pool + MLP, 5 classes | Yes if model file and deps present |
| Training | CSV → CFG + embed → GNN, BCE, early stop | Yes; data pipeline incomplete (no build_dataset.py) |
| newALLBUGS | 4-class, label190, vocab | Used by app/ml datasets only; not by GNN training script |
| Demo mode | Regex patterns + weights + safe dampening | Yes; many classic patterns covered |
| Line attribution | Pattern-based in both modes | Works but not model-driven in GNN mode |
| Slither | Optional extra features / health | Not used in model or training |

**Bottom line:** The system is technically sound and well-structured for a prototype. To judge “does it work well?” in production, you need: (1) a proper 5-class dataset and `build_dataset.py`, (2) reported metrics per class and threshold analysis, and (3) alignment of API/suggestions with 5 classes (including timestamp). For learning and demos, it works; for serious auditing, treat it as an aid, not a replacement for human review and formal tools.
