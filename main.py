"""
RLM-style table relation finder (LLM-in-the-loop) — UPDATED

Key upgrades (based on your run where LLM didn’t help):
1) No hardcoded initial base:
   - discover_initial_bases() builds a strong starting point from primitives.
2) True residual-target prompting:
   - We ask the LLM to propose formulas for RESID = Y - BASE (NOT for Y directly).
   - We then wrap candidates as: (BASE) + (RESID_FORMULA) (and a few structured variants).
3) Better base discovery:
   - Includes single cols, pairwise + - * /, and affine combos (a*X+b*Z+c) for small integers.
4) More robust iteration:
   - No "stop after 1 no-improvement round". We allow a small patience window.
5) Better prompt shaping:
   - Provide tiny row samples (8–10) + top error rows + residual stats.
6) Optional deterministic residual search:
   - After each iteration, we run a tiny enumerator on RESID to catch obvious corrections.

You must set OPENAI_API_KEY (env var) and have `openai` installed.
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd
from openai import OpenAI

# load env vars from .env if present
from dotenv import load_dotenv
load_dotenv()


# ----------------------------
# Metrics
# ----------------------------
def safe_div(a, b, eps=1e-12):
    b = np.asarray(b, dtype=float)
    return np.asarray(a, dtype=float) / (b + np.where(b == 0, eps, 0.0))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot <= 0:
        return float("-inf")
    return 1.0 - ss_res / ss_tot

def max_abs_err(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.max(np.abs(y_true - y_pred)))

def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ----------------------------
# Safe formula evaluation via AST whitelist
# ----------------------------
_ALLOWED_FUNCS = {
    "abs": np.abs,
    "round": np.round,   # numpy round (banker’s rounding)
    "floor": np.floor,
    "ceil": np.ceil,
    "min": np.minimum,
    "max": np.maximum,
    "clip": np.clip,
    "safe_div": safe_div,   # ✅ allow safe_div in formulas
}

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.keyword,            # ✅ allow keyword nodes
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)

class UnsafeExpression(Exception):
    pass

def _validate_ast(node: ast.AST, allowed_names: set):
    if not isinstance(node, _ALLOWED_NODES):
        raise UnsafeExpression(f"Disallowed AST node: {type(node).__name__}")

    if isinstance(node, ast.Attribute):
        raise UnsafeExpression("Attribute access is not allowed")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise UnsafeExpression("Only direct function calls allowed (e.g., abs(x), floor(x))")
        if node.func.id not in _ALLOWED_FUNCS:
            raise UnsafeExpression(f"Function not allowed: {node.func.id}")
        for arg in node.args:
            _validate_ast(arg, allowed_names)
        for kw in node.keywords:
            _validate_ast(kw.value, allowed_names)

    if isinstance(node, ast.Name):
        if node.id not in allowed_names and node.id not in _ALLOWED_FUNCS:
            raise UnsafeExpression(f"Unknown name: {node.id}")

    for child in ast.iter_child_nodes(node):
        _validate_ast(child, allowed_names)

def eval_formula(expr: str, env: Dict[str, np.ndarray]) -> np.ndarray:
    expr = expr.strip().replace("^", "**")
    if any(tok in expr for tok in ["__", "[", "]", "{", "}", "import", "exec", "eval", "open", "os.", "sys."]):
        raise UnsafeExpression("Unsafe tokens present")

    allowed_names = set(env.keys()) | set(_ALLOWED_FUNCS.keys())
    tree = ast.parse(expr, mode="eval")
    _validate_ast(tree, allowed_names)
    code = compile(tree, "<formula>", "eval")
    scope = {**_ALLOWED_FUNCS, **env}
    return np.asarray(eval(code, {"__builtins__": {}}, scope), dtype=float)


# ----------------------------
# OpenAI LLM hook
# ----------------------------
def query_llm(prompt: str, model: str = "gpt-5-mini-2025-08-07") -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You propose mathematical formulas for tabular data.\n"
                    "Output ONE formula per line. No numbering, no bullets, no explanations.\n"
                    "Use only the allowed operators/functions and provided variables.\n"
                    "Prefer simple residual-correction forms."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=800,
    )
    return resp.choices[0].message.content or ""


# ----------------------------
# Scoring + helpers
# ----------------------------
@dataclass
class CandidateScore:
    expr: str
    rmse: float
    mae: float
    r2: float
    max_err: float
    fails: int
    fail_rows: List[int]

def score_expr(expr: str, df: pd.DataFrame, target: str, tol: float) -> Optional[CandidateScore]:
    env = {c: df[c].to_numpy(dtype=float) for c in df.columns}
    y = env[target]
    try:
        yhat = eval_formula(expr, env)
        if not np.all(np.isfinite(yhat)):
            return None
        errs = np.abs(y - yhat)
        fail_rows = df.index[errs > tol].tolist()
        return CandidateScore(
            expr=expr,
            rmse=rmse(y, yhat),
            mae=mae(y, yhat),
            r2=r2(y, yhat),
            max_err=float(np.max(errs)),
            fails=len(fail_rows),
            fail_rows=fail_rows,
        )
    except Exception:
        return None

def compute_residual(df: pd.DataFrame, target: str, base_expr: str) -> np.ndarray:
    env = {c: df[c].to_numpy(dtype=float) for c in df.columns}
    y = env[target]
    yhat = eval_formula(base_expr, env)
    return y - yhat

def sample_rows_for_prompt(df: pd.DataFrame, cols: List[str], n: int = 10, seed: int = 0) -> pd.DataFrame:
    if len(df) <= n:
        return df[cols].copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(df.index.to_numpy(), size=n, replace=False)
    return df.loc[idx, cols].copy()

def top_fail_rows(df: pd.DataFrame, target: str, expr: str, tol: float, k: int = 6) -> pd.DataFrame:
    env = {c: df[c].to_numpy(dtype=float) for c in df.columns}
    y = env[target]
    yhat = eval_formula(expr, env)
    errs = np.abs(y - yhat)
    top_idx = df.index.to_numpy()[np.argsort(-errs)[:k]]
    out = df.loc[top_idx].copy()
    out["ERR"] = errs[np.argsort(-errs)[:k]]
    return out

def parse_llm_formulas(text: str) -> List[str]:
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        s = re.sub(r"^\s*[\-\*\u2022]\s+", "", s)
        s = re.sub(r"^\s*\d+[\)\.\:]\s*", "", s)
        s = s.strip()
        if len(s) > 220:
            continue
        if not re.match(r"^[A-Za-z0-9_\s\+\-\*\/\%\^\(\)\.\,]+$", s):
            continue
        lines.append(s)
    seen = set()
    out = []
    for s in lines:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ----------------------------
# Base discovery (stronger)
# ----------------------------
def _primitive_candidates(cols: List[str]) -> List[str]:
    cands: List[str] = []
    # singles
    cands += cols
    # small constants
    cands += ["0", "1", "2", "-1", "10"]
    # pairwise ops
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            cands += [f"({a}+{b})", f"({a}-{b})", f"({b}-{a})", f"({a}*{b})", f"({a}/{b})", f"({b}/{a})"]
    return cands

def _affine_candidates(cols: List[str], coeffs: Iterable[int] = (-2, -1, 1, 2), bias: Iterable[int] = (-5, -2, -1, 0, 1, 2, 5)) -> List[str]:
    # a*X + b*Z + c with small integer coefficients
    cands: List[str] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            x, z = cols[i], cols[j]
            for a in coeffs:
                for b in coeffs:
                    for c in bias:
                        # skip too-trivial
                        if a == 0 and b == 0:
                            continue
                        term_x = f"({a}*{x})" if a != 1 else x
                        term_z = f"({b}*{z})" if b != 1 else z
                        expr = f"({term_x}+{term_z}+{c})"
                        cands.append(expr)
    return cands

def discover_initial_bases(df: pd.DataFrame, target: str, tol: float, top_k: int = 8) -> List[CandidateScore]:
    df = normalize_colnames(df)
    cols = [c for c in df.columns if c != target]

    candidates = []
    candidates += _primitive_candidates(cols)
    candidates += _affine_candidates(cols)

    scored: List[CandidateScore] = []
    for expr in candidates:
        s = score_expr(expr, df, target, tol)
        if s:
            scored.append(s)

    scored.sort(key=lambda x: (x.fails, x.rmse, x.max_err, -x.r2))
    return scored[:top_k]


# ----------------------------
# Tiny deterministic residual search (fallback)
# ----------------------------
def residual_brutish_candidates(cols: List[str]) -> List[str]:
    cands = []
    cands += cols
    # remove "0" — it's almost always a dead end and causes repeats
    cands += ["1", "2", "-1", "-2", "10", "-10"]
    for c in cols:
        cands += [f"abs({c})", f"round({c})", f"floor({c})", f"ceil({c})"]
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            cands += [f"({a}+{b})", f"({a}-{b})", f"({a}*{b})", f"({a}/{b})"]
    return cands


# ----------------------------
# Prompt builder: ask for RESID formulas only
# ----------------------------
def build_resid_prompt(
    df: pd.DataFrame,
    target: str,
    base_expr: str,
    sample_df: pd.DataFrame,
    fail_df: pd.DataFrame,
    resid_stats: Dict[str, float],
    best_base: List[str],
    n_suggestions: int,
) -> str:
    colnames = [c for c in df.columns if c != target]

    allowed_ops = "+  -  *  /  //  %  **  ( )"
    allowed_funcs = ", ".join(_ALLOWED_FUNCS.keys())
    print(f"Length of best_base for prompt: {len(best_base)}")

    return f"""
We are fitting a table with target {target}.

Current best base:
BASE = {base_expr}

Define residual:
RESID = {target} - BASE

Best base candidates considered:
{', '.join(best_base)}

Task:
- Propose {n_suggestions} candidate formulas for RESID using the input columns.
- Output ONE RESID formula per line. No explanations.
- Look through the best base candidates and error rows to find patterns in the residuals.
- Never try the one from best base again; instead, look for simple corrections that could fix the residuals.
- Keep formulas simple.
- Allowed operators: {allowed_ops}
- Allowed functions: {allowed_funcs}
- Allowed variables: {', '.join(colnames)}

Residual stats:
mean={resid_stats['mean']:.6g}
std={resid_stats['std']:.6g}
min={resid_stats['min']:.6g}
max={resid_stats['max']:.6g}

Sample rows (includes BASE and RESID):
{sample_df.to_csv(index=False).strip()}

Worst-error rows under BASE (includes BASE and RESID):
{fail_df.to_csv(index=False).strip()}

Now output RESID formulas (one per line):
""".strip()

# ----------------------------
# RLM loop (improved)
# ----------------------------
def rlm_find_formula(
    df: pd.DataFrame,
    target: str,
    tol: float = 0.0,
    max_iters: int = 8,
    llm_suggestions: int = 30,
    keep_top: int = 8,
    patience: int = 2,
    seed: int = 0,
    model: str = "gpt-5-mini-2025-08-07",
    verbose: bool = True,
) -> Tuple[str, List[CandidateScore]]:
    """
    UPDATED:
    - Tracks/respects previously-tried residual formulas (seen_resid) so LLM + deterministic candidates
      don't keep repeating BASE+0 etc.
    - Filters trivial residuals (0, +/-0, (0), etc.)
    - De-dupes wrapped Y candidates
    - Uses a patience window (keeps your existing behavior)
    """
    df = normalize_colnames(df)
    assert target in df.columns, f"Target '{target}' not found. Columns: {list(df.columns)}"

    # 1) Discover starting bases (no hardcode)
    bases = discover_initial_bases(df, target, tol=tol, top_k=keep_top)
    if not bases:
        raise ValueError("Could not discover any valid initial base formulas.")

    best_score = bases[0]
    best_expr = best_score.expr

    history: List[CandidateScore] = [best_score]
    stall = 0

    # Keep a human-readable list for prompt context (optional)
    best_base: List[str] = []
    if verbose:
        print("Initial base candidates:")
        for k, b in enumerate(bases[:min(5, len(bases))], 1):
            print(f"{k:02d}  {b.expr}  fails={b.fails} rmse={b.rmse:.6g} max_err={b.max_err:.6g} r2={b.r2:.4f}")
            best_base.append(f"{b.expr} (fails={b.fails} rmse={b.rmse:.6g} max_err={b.max_err:.6g} r2={b.r2:.4f})")
        print(f"\nStarting with: {best_expr}\n")

    cols = [c for c in df.columns if c != target]
    return

    # NEW: track residual formulas we've already tried across iterations
    seen_resid: set[str] = set()

    def is_trivial_resid(r: str) -> bool:
        rr = r.replace(" ", "")
        return rr in {"0", "+0", "-0", "(0)", "((0))", "((+0))", "((-0))"}

    for it in range(1, max_iters + 1):
        # 2) Compute residual and build prompt tables
        env_base = {c: df[c].to_numpy(dtype=float) for c in df.columns}
        base_pred = eval_formula(best_expr, env_base)
        resid = df[target].to_numpy(dtype=float) - base_pred

        resid_stats = {
            "mean": float(np.mean(resid)),
            "std": float(np.std(resid)),
            "min": float(np.min(resid)),
            "max": float(np.max(resid)),
        }

        work = df.copy()
        work["BASE"] = base_pred
        work["RESID"] = resid

        cols_for_prompt = [c for c in df.columns] + ["BASE", "RESID"]
        sample_df = sample_rows_for_prompt(work, cols_for_prompt, n=10, seed=seed + it)

        # worst rows under BASE (not BASE+RESID)
        fail_df = top_fail_rows(work, target, "BASE", tol=tol, k=6)[cols_for_prompt + ["ERR"]]

        prompt = build_resid_prompt(
            df=df,
            target=target,
            base_expr=best_expr,
            sample_df=sample_df,
            fail_df=fail_df.drop(columns=["ERR"], errors="ignore"),
            resid_stats=resid_stats,
            best_base=best_base,          # if you keep this in the prompt
            n_suggestions=llm_suggestions,
        )

        # 3) Get RESID candidates from LLM
        llm_text = query_llm(prompt, model=model)
        resid_cands = parse_llm_formulas(llm_text)

        # 4) Add deterministic residual candidates
        resid_cands += residual_brutish_candidates(cols)

        # 5) De-dup and filter:
        #    - remove trivial residuals like 0
        #    - remove residuals already tried in previous rounds
        resid_seen_local = set()
        resid_filtered: List[str] = []
        for r in resid_cands:
            r = r.strip()
            if not r:
                continue
            if r in resid_seen_local:
                continue
            resid_seen_local.add(r)

            if is_trivial_resid(r):
                continue
            if r in seen_resid:
                continue

            resid_filtered.append(r)

        resid_cands = resid_filtered

        # NEW: mark these residuals as attempted (so they won't repeat next iter)
        for r in resid_cands:
            seen_resid.add(r)

        # 6) Wrap into Y candidates (BASE + RESID_FORMULA)
        y_cands: List[str] = []
        for r in resid_cands:
            y_cands.append(f"({best_expr})+({r})")
            y_cands.append(f"round(({best_expr})+({r}))")
            y_cands.append(f"({best_expr})-({r})")

        # De-dupe wrapped Y candidates (NEW)
        y_seen = set()
        y_cands_uniq = []
        for e in y_cands:
            e = e.strip()
            if e and e not in y_seen:
                y_seen.add(e)
                y_cands_uniq.append(e)
        y_cands = y_cands_uniq

        # 7) Score candidates
        scored: List[CandidateScore] = []
        for e in y_cands:
            s = score_expr(e, df, target, tol=tol)
            if s is not None:
                scored.append(s)

        # Always keep current best
        scored.append(best_score)
        scored.sort(key=lambda x: (x.fails, x.max_err, x.rmse, -x.r2))

        top = scored[:keep_top]
        new_best = top[0]

        if verbose:
            print(f"\n=== Iteration {it} ===")
            print(f"Current best: {best_score.expr} | fails={best_score.fails} rmse={best_score.rmse:.6g} max_err={best_score.max_err:.6g} r2={best_score.r2:.4f}")
            print("Top candidates:")
            for k, t in enumerate(top, 1):
                print(f"{k:02d}  {t.expr}  fails={t.fails}  rmse={t.rmse:.6g}  max_err={t.max_err:.6g}  r2={t.r2:.4f}")
                # optional: grow "best_base" prompt context
                best_base.append(f"{t.expr} (fails={t.fails} rmse={t.rmse:.6g} max_err={t.max_err:.6g} r2={t.r2:.4f})")

        history.append(new_best)

        # Stop if exact
        if new_best.fails == 0:
            if verbose:
                print("\n✅ Found exact match within tolerance.")
            return new_best.expr, history

        # Improvement test
        improved = (new_best.fails < best_score.fails) or (
            new_best.fails == best_score.fails and new_best.max_err < best_score.max_err - 1e-12
        ) or (
            new_best.fails == best_score.fails and abs(new_best.max_err - best_score.max_err) < 1e-12 and new_best.rmse < best_score.rmse - 1e-12
        )

        if improved:
            best_score = new_best
            best_expr = new_best.expr
            stall = 0
        else:
            stall += 1
            if verbose:
                print(f"\n⚠️ No improvement this round. stall={stall}/{patience}")
            if stall >= patience:
                if verbose:
                    print("\nStopping due to patience limit.")
                return best_expr, history

    return best_expr, history

# ----------------------------
# Example
# ----------------------------
if __name__ == "__main__":
    # Replace with your real data:
    # df = pd.read_csv("your_table.csv")
    # target = "Y"

    # Demo synthetic:
    rng = np.random.default_rng(0)
    size = 500
    df = pd.DataFrame({
        "A": rng.integers(1, 10, size=size),
        "B": rng.integers(1, 10, size=size),
        "C": rng.integers(1, 10, size=size),
        "D": rng.integers(1, 10, size=size),
        "E": rng.integers(1, 10, size=size),
        "F": rng.integers(1, 10, size=size),
    })
    # hidden formula
    df["Y"] = df["C"] * df["B"] / df["A"]

    print(f"Data sample:\n{df.head()}\n")
    
    best, hist = rlm_find_formula(
        df,
        target="Y",
        tol=0.0,
        max_iters=8,
        llm_suggestions=5,
        keep_top=8,
        patience=5,
        model="gpt-5-mini-2025-08-07",
        verbose=True,
    )
    print("\nBest formula:", best)
