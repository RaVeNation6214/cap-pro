# Smart Contract Vulnerability Detection

AI-powered smart contract vulnerability detection using a **Hierarchical Transformer** architecture with an interactive React UI.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![React](https://img.shields.io/badge/react-18+-blue.svg)

## Overview

This project implements a novel **Hierarchical Window-to-Contract Transformer** for detecting vulnerabilities in Ethereum smart contracts. It features:

- **4-Class Detection**: Reentrancy, Access Control, Arithmetic, Unchecked Calls
- **Explainable AI**: Attention-based visualization shows risky code regions
- **Line-level Analysis**: Per-line risk scores, not just contract-level predictions
- **Modern UI**: Beautiful React interface with smooth animations

## Project Structure

```
smart-contract-audit/
├── backend/                 # FastAPI + PyTorch backend
│   ├── app/
│   │   ├── api/            # API routes and schemas
│   │   ├── core/           # Configuration
│   │   ├── ml/             # ML model architecture
│   │   └── services/       # Business logic
│   └── requirements.txt
├── frontend/               # React + Tailwind frontend
│   ├── src/
│   │   ├── components/     # Reusable UI components
│   │   ├── pages/          # Application pages
│   │   └── services/       # API service layer
│   └── package.json
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- npm or yarn

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

The UI will be available at `http://localhost:5173`

## Features

### Vulnerability Classes

| Class | Description | Severity |
|-------|-------------|----------|
| **Reentrancy** | External call before state update | Critical |
| **Access Control** | tx.origin authentication | Critical |
| **Unchecked Calls** | Missing return value checks | High |
| **Arithmetic** | Integer overflow/underflow | High |

### Architecture

```
Solidity Contract
       ↓
[Split into 3-line windows]
       ↓
[Window Transformer Encoder]
 - 3 layers, 8 heads, d_model=256
       ↓
[Contract-Level Attention Pooling]
       ↓
[Multi-Label Classifier (4 classes)]
       ↓
Predictions + Attention Heatmap
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/analyze` | Analyze a contract for vulnerabilities |
| GET | `/api/sample-contracts` | Get sample contracts for demo |
| GET | `/api/health` | Health check endpoint |
| GET | `/api/vulnerability-classes` | Get vulnerability class info |

### UI Pages

- **Home**: Landing page with animated hero and feature showcase
- **Analyze**: Code editor with drag-drop upload and sample contracts
- **Results**: Vulnerability cards, risk gauge, and attention heatmap
- **About**: Project information and architecture details

## Demo Mode

The application runs in **demo mode** by default, which simulates model predictions using pattern matching. This allows you to test the full UI without training the model.

To switch to real model inference:
1. Train the model using your dataset
2. Save the model weights to `backend/models/model.pt`
3. Set `DEMO_MODE=False` in `backend/app/core/config.py`

## Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **PyTorch** - Deep learning framework
- **Pydantic** - Data validation

### Frontend
- **React 18** - UI library
- **Vite** - Build tool
- **Tailwind CSS** - Utility-first CSS
- **Framer Motion** - Animation library
- **Recharts** - Charting library
- **Lucide React** - Icon library

## Research Background

This project is inspired by recent advances in smart contract security:

- **SCVDIE (2022)**: Ensemble deep learning for vulnerability detection
- **Lightning Cat (2023)**: CodeBERT with function-level analysis
- **GNNSE (2025)**: GNN + symbolic execution

Our approach offers:
- Lighter architecture (1.5-2M params vs 110M+ for CodeBERT)
- Explainable attention-based predictions
- Fine-grained window-level detection

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- VIT Chennai for capstone project guidance
- OpenZeppelin for security best practices
- Ethereum community for vulnerability datasets

---

**B.Tech Capstone Project 2026**
