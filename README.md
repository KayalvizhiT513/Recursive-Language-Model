# RLM - Recursive Language Model

An LLM-in-the-loop system for discovering mathematical formulas in tabular data. RLM iteratively refines formula predictions by asking an LLM to propose corrections to residuals (prediction errors).

## Features

- **LLM-Powered Formula Discovery**: Uses OpenAI's API to intelligently propose formulas
- **Residual-Based Refinement**: Iteratively improves predictions by modeling and correcting errors
- **Safe Formula Evaluation**: Evaluates formulas using AST validation (prevents code injection)
- **Automatic Base Discovery**: Starts with strong initial formulas derived from data (no hardcoding)
- **Comprehensive Metrics**: Tracks RMSE, MAE, R², and max error for each candidate
- **Deterministic Fallback**: Includes exhaustive search for residual patterns as backup

## Installation

```bash
pip install openai pandas numpy python-dotenv
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key"
# or create a .env file
echo "OPENAI_API_KEY=your-api-key" > .env
```

## Quick Start

```python
import pandas as pd
from main import rlm_find_formula

# Load your data
df = pd.read_csv("your_data.csv")

# Find formula for target column
best_formula, history = rlm_find_formula(
    df,
    target="Y",
    tol=0.0,
    max_iters=8,
    llm_suggestions=30,
    keep_top=8,
    patience=2,
    verbose=True
)

print(f"Best formula: {best_formula}")
```

## How It Works

1. **Initial Base Discovery**: Generates candidate formulas from primitives (single columns, pairwise operations, affine combinations)
2. **Iteration Loop**:
   - Compute residual: `RESID = Y - BASE`
   - LLM proposes formulas to explain the residual
   - Deterministic search finds additional residual patterns
   - Wrap candidates: `BASE + RESID_FORMULA`
   - Score and keep top performers
3. **Convergence**: Stops when exact match found or patience limit reached

## Parameters

- `target`: Name of the target column to predict
- `tol`: Error tolerance threshold (default: 0.0)
- `max_iters`: Maximum iterations (default: 8)
- `llm_suggestions`: Number of formulas LLM proposes (default: 30)
- `keep_top`: Top candidates to track (default: 8)
- `patience`: Iterations without improvement before stopping (default: 2)
- `model`: OpenAI model to use (default: "gpt-5-mini-2025-08-07")
- `verbose`: Print progress (default: True)

## Allowed Operations

**Operators**: `+`, `-`, `*`, `/`, `//`, `%`, `^` (exponent)

**Functions**: `abs()`, `round()`, `floor()`, `ceil()`, `min()`, `max()`, `clip()`

## Example

See the `__main__` section for a complete working example that discovers the formula `C * B / A` from synthetic data.