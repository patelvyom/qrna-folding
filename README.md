# Quantum RNA Folding

A Streamlit app for predicting RNA secondary structure using QAOA (Quantum Approximate Optimization Algorithm) on PennyLane.

Based on [Jiang et al. (2023)](https://arxiv.org/abs/2305.09561).

## Quick Start

```bash
# Install dependencies
pixi install

# Run the app
pixi run streamlit run qrna_folding/main.py
```

Open **http://localhost:8501** in your browser.

## Usage

The app follows a 3-step workflow:

### 1. Preprocessing

- Enter an RNA sequence using valid bases: `A`, `U`, `G`, `C`
- Set **minimum stem length** (default: 3)
- Click **Perform Preprocessing**
- View the adjacency matrix showing potential base pairs (AU/GU = 2 H-bonds, GC = 3 H-bonds)

### 2. Stem Selection

- Choose **number of stems** to analyze
- Select method:
  - `longest` — prioritize stems with most base pairs
  - `random` — random selection
- Click **Select Stems**
- View network graphs of selected stems

### 3. QAOA Experiment

- Set **QAOA layers** (2–9, default: 5)
- Choose **device**:
  - `default.qubit` — standard PennyLane simulator
  - `lightning.qubit` — high-performance C++ simulator
- Click **Run Experiment**
- Monitor optimization in terminal (100 Adam steps)

## Project Structure

```
qrna_folding/
├── main.py           # Streamlit UI
├── preprocessors.py  # RNA processing, stem detection
├── experiments.py    # QAOA circuit and optimization
├── visualisation.py  # Adjacency matrix and network plots
└── utils.py          # Hamiltonian utilities
```

## Dependencies

Runtime dependencies managed via [Pixi](https://pixi.sh) — see `pixi.toml`.

Dev dependencies defined in `pyproject.toml`:

- `ruff` — linter and formatter
- `pre-commit` — git hooks
