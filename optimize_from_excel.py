"""
optimize_from_excel_cleaned.py
--------------------------------
Excel‑driven Bayesian optimization (ProcessOptimizer) with a pivot RUNS sheet.

Key cleanups vs. the original:
- Removed unused SETUP‑CONST reader (constants were never used).
- Simplified RUNS parsing; read_runs_pivot() now returns only what we actually use.
- print_loss_report() reuses the already‑read RUNS dataframe to avoid re‑reading.
- Added automatic .pkl compatibility / invalidation via a space signature. If the Excel
  variable space changes (names, types, bounds/choices), the old pickle is ignored so
  you *don’t* have to delete it manually anymore.
- Tightened helpers; added type hints and minimal docstrings.
- Kept behavior and sheet layouts intact; zero Excel schema changes required.

Sheets layout:
  TEMPLATE      (ignored)
  SETUP‑VAR     section | name | type | low | high | choices | unit | active
  SETUP‑OUT     section | name | kind | target | weight
  SETUP‑OPT     section | name | value
  RUNS          pivot:
      Row "vars", then variable rows; row "outputs", then output rows; optional "loss" row.
      Experiments are columns (B..).

Suggestions are printed to console.
Loss values are COMPUTED and PRINTED to console (not written to Excel).
"""
from __future__ import annotations

from pathlib import Path
import pickle
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import ProcessOptimizer as po
from ProcessOptimizer.space import Real, Integer, Categorical

# plotting (non-interactive backend)
try:
    import matplotlib
    matplotlib.use("Agg")  # safe for headless runs
    import matplotlib.pyplot as plt
except Exception:
    matplotlib = None
    plt = None

# -------------------- CONFIG --------------------
EXCEL_PATH = Path("Excel/opica_tp4_optimizer.xlsx")  # adjust path/name
STATE_PKL  = EXCEL_PATH.with_suffix(".pkl")
VERBOSE_WARMSTART = True
MAX_DIVERSITY_TRIES = 500
# ------------------------------------------------

# ----------------- helpers ----------------------
def _norm_cols_inplace(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _as_bool(x: Any, default: bool = True) -> bool:
    s = str(x).strip().lower()
    if s in ("true","1","yes","y"): return True
    if s in ("false","0","no","n"): return False
    return default

def _num_from_cell(val: Any) -> Optional[float]:
    if pd.isna(val) or val == "":
        return None
    s = str(val).strip().replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def _int_from_cell(val: Any) -> Optional[int]:
    f = _num_from_cell(val)
    if f is None:
        return None
    try:
        return int(round(f))
    except Exception:
        return None

# --------------- SETUP readers ------------------
def read_setup_var(path: Path) -> List[Dict[str, Any]]:
    """Read active variables from SETUP‑VAR.

    Returns list of dicts with keys depending on type: Real/Integer/Categorical.
    """
    df = pd.read_excel(path, sheet_name="SETUP-VAR").fillna("")
    _norm_cols_inplace(df)
    for need in ["name","type","low","high","choices","unit","active"]:
        if need not in df.columns:
            df[need] = ""
    if "section" in df.columns:
        df = df[df["section"].astype(str).str.strip().str.lower() == "var"]
    df["name"]   = df["name"].astype(str).str.strip()
    df["type"]   = df["type"].astype(str).str.strip().str.capitalize()
    df["unit"]   = df["unit"].astype(str).str.strip()
    df["active"] = df["active"].apply(_as_bool)
    df = df[df["active"] == True]

    variables: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        name = r["name"]
        vtype = r["type"] or "Real"
        unit = r["unit"]
        if vtype == "Categorical":
            choices = [c.strip() for c in str(r.get("choices", "")).split(",") if c.strip()]
            if not choices:
                raise ValueError(f"SETUP-VAR: categorical var '{name}' needs 'choices'.")
            variables.append({"type": "Categorical", "name": name, "choices": choices, "unit": unit})
        elif vtype == "Integer":
            lo = _int_from_cell(r.get("low", ""))
            hi = _int_from_cell(r.get("high", ""))
            if lo is None or hi is None:
                raise ValueError(f"SETUP-VAR: integer var '{name}' needs numeric low/high.")
            variables.append({"type": "Integer", "name": name, "low": lo, "high": hi, "unit": unit})
        else:  # Real
            lo = _num_from_cell(r.get("low", ""))
            hi = _num_from_cell(r.get("high", ""))
            if lo is None or hi is None:
                raise ValueError(f"SETUP-VAR: real var '{name}' needs numeric low/high.")
            variables.append({"type": "Real", "name": name, "low": lo, "high": hi, "unit": unit})
    return variables


def read_setup_out(path: Path) -> List[Dict[str, Any]]:
    """Read outputs (objectives) from SETUP‑OUT.

    kind ∈ {min, max, target}; default target=100, weight=1.
    """
    df = pd.read_excel(path, sheet_name="SETUP-OUT").fillna("")
    _norm_cols_inplace(df)
    for need in ["name","kind","target","weight"]:
        if need not in df.columns:
            df[need] = ""
    if "section" in df.columns:
        df = df[df["section"].astype(str).str.strip().str.lower() == "out"]
    df["name"] = df["name"].astype(str).str.strip()
    df["kind"] = df["kind"].astype(str).str.strip().str.lower()

    outputs: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        tgt = _num_from_cell(r.get("target", "")); tgt = 100.0 if tgt is None else tgt
        w   = _num_from_cell(r.get("weight", "")); w   = 1.0   if w   is None else w
        kind = r.get("kind", "min") or "min"
        outputs.append({"name": r["name"], "kind": kind, "target": float(tgt), "weight": float(w)})
    return outputs


def read_setup_opt(path: Path) -> Dict[str, Any]:
    """Read optimization options from SETUP‑OPT with robust header handling.

    Recognized options (name → value):
      - acq_func (str)
      - n_initial_points (int)
      - batch_size (int)
      - diversity_eps (float)
      - plot_objective (bool; default False)
      - plot_path (str; default "objective.png")
      - plot_dpi (int; default 150)
    """
    df = pd.read_excel(path, sheet_name="SETUP-OPT").fillna("")
    try:
        _norm_cols_inplace(df)
    except Exception:
        pass
    if "name" not in df.columns or "value" not in df.columns:
        df = pd.read_excel(path, sheet_name="SETUP-OPT", header=None).fillna("")
        if df.shape[1] >= 3:
            df.columns = ["section","name","value"] + [f"extra_{i}" for i in range(3, df.shape[1])]
        else:
            df.columns = ["name","value"] + [f"extra_{i}" for i in range(2, df.shape[1])]
    _norm_cols_inplace(df)

    opts: Dict[str, Any] = {
        "acq_func": "EI",
        "n_initial_points": 6,
        "batch_size": 1,
        "diversity_eps": 0.05,
        # plotting
        "plot_objective": False,
        "plot_path": "objective.png",
        "plot_dpi": 150,
    }
    for _, r in df.iterrows():
        n = str(r.get("name", "")).strip().lower()
        v = r.get("value", "")
        if n == "acq_func":
            opts["acq_func"] = str(v)
        elif n == "n_initial_points":
            iv = _int_from_cell(v);  opts["n_initial_points"] = iv if iv is not None else opts["n_initial_points"]
        elif n == "batch_size":
            iv = _int_from_cell(v);  opts["batch_size"]       = iv if iv is not None else opts["batch_size"]
        elif n == "diversity_eps":
            fv = _num_from_cell(v);  opts["diversity_eps"]    = fv if fv is not None else opts["diversity_eps"]
        elif n == "plot_objective":
            opts["plot_objective"] = _as_bool(v, default=False)
        elif n == "plot_path":
            opts["plot_path"] = str(v).strip() or opts["plot_path"]
        elif n == "plot_dpi":
            iv = _int_from_cell(v);  opts["plot_dpi"] = iv if iv is not None else opts["plot_dpi"]
    return opts

# --------------- RUNS (pivot) reader ---------------
def read_runs_pivot(path: Path) -> Tuple[List[Optional[Dict[str, Any]]], pd.DataFrame, List[str]]:
    """Parse RUNS sheet and construct experiments.

    Returns (experiments, raw_dataframe, var_names_order).
    - experiments is a list aligned to experiment columns (may include None entries for incomplete rows).
    - var_names_order are the variable row labels in appearance order under the 'vars' header.
    """
    df = pd.read_excel(path, sheet_name="RUNS", header=None)
    if df.empty:
        return [], df, []

    # Find 'vars' header row
    vars_header_idx = df.index[df.iloc[:,0].astype(str).str.strip().str.lower() == "vars"]
    if len(vars_header_idx) == 0:
        return [], df, []
    vh = vars_header_idx[0]

    # variable rows
    r = vh + 1
    var_rows: List[int] = []
    while r < len(df):
        first = str(df.iloc[r,0]).strip()
        if first == "" or first.lower() == "outputs":
            break
        var_rows.append(r)
        r += 1

    # outputs rows
    outs_header_idx = df.index[df.iloc[:,0].astype(str).str.strip().str.lower() == "outputs"]
    out_rows: List[int] = []
    if len(outs_header_idx):
        oh = outs_header_idx[0]
        r2 = oh + 1
        while r2 < len(df):
            first = str(df.iloc[r2,0]).strip()
            if first == "":
                break
            out_rows.append(r2)
            r2 += 1

    # experiment columns (headers on vh row)
    exp_cols: List[int] = []
    for j in range(1, df.shape[1]):
        hdr = str(df.iloc[vh, j]).strip()
        if hdr != "":
            exp_cols.append(j)

    # build experiments
    experiments: List[Optional[Dict[str, Any]]] = []
    for j in exp_cols:
        e: Dict[str, Any] = {}
        good = True
        for rr in var_rows:
            key = str(df.iloc[rr,0]).strip()
            val = df.iloc[rr, j]
            if pd.isna(val) or str(val).strip() == "":
                good = False
                break
            e[key] = val
        if not good:
            experiments.append(None)  # keep alignment
            continue
        for rr in out_rows:
            key = str(df.iloc[rr,0]).strip()
            if key.lower() == "loss":  # ignore recorded loss
                continue
            val = df.iloc[rr, j]
            if pd.isna(val) or str(val).strip() == "":
                continue
            e[key] = val
        experiments.append(e)

    var_names_order = [str(df.iloc[rr,0]).strip() for rr in var_rows]
    return experiments, df, var_names_order

# --------------- loss & space -------------------
def compute_loss(outputs_cfg: Sequence[Dict[str, Any]], row_dict: Dict[str, Any]) -> Optional[float]:
    """Compute scalar loss from outputs according to SETUP‑OUT config."""
    total = 0.0
    for cfg in outputs_cfg:
        n = cfg["name"]
        kind = cfg.get("kind", "min")
        w = float(cfg.get("weight", 1.0))
        tgt = float(cfg.get("target", 100.0))
        if n not in row_dict:
            return None
        val = _num_from_cell(row_dict[n])
        if val is None:
            return None
        if kind == "min":
            pen = val
        elif kind == "max":
            # Penalize shortfall from target for max problems.
            pen = max(0.0, tgt - val)
        elif kind == "target":
            pen = abs(val - tgt)
        else:
            raise ValueError(f"Unknown kind '{kind}' for output '{n}'")
        total += w * pen
    return total


def build_space(variables: Sequence[Dict[str, Any]]) -> po.Space:
    dims = []
    for v in variables:
        if v["type"] == "Real":
            dims.append(Real(v["low"], v["high"], name=v["name"]))
        elif v["type"] == "Integer":
            dims.append(Integer(v["low"], v["high"], name=v["name"]))
        elif v["type"] == "Categorical":
            dims.append(Categorical(v["choices"], name=v["name"]))
    return po.Space(dims)


def vec_to_named(x_vec: Sequence[Any], variables: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for i, v in enumerate(variables):
        t = v["type"]; n = v["name"]
        if t == "Categorical":
            out[n] = x_vec[i]
        elif t == "Real":
            out[n] = float(x_vec[i])
        else:
            out[n] = int(x_vec[i])
    return out


def normalized_distance(a: Dict[str, Any], b: Dict[str, Any], variables: Sequence[Dict[str, Any]]) -> float:
    num = 0.0; cnt = 0
    for v in variables:
        t = v["type"]; n = v["name"]
        va, vb = a[n], b[n]
        if t == "Categorical":
            num += 0.0 if va == vb else 1.0; cnt += 1
        else:
            span = float(v["high"] - v["low"]) if t != "Categorical" else 1.0
            if span <= 0:
                continue
            num += abs((float(va) - float(vb)) / span); cnt += 1
    return num / max(cnt, 1)


def ask_batch(opt: po.Optimizer, k: int, variables: Sequence[Dict[str, Any]], diversity_eps: float) -> List[Dict[str, Any]]:
    if k <= 1:
        return [vec_to_named(opt.ask(), variables)]
    batch: List[Dict[str, Any]] = []
    tries = 0
    while len(batch) < k and tries < MAX_DIVERSITY_TRIES:
        xn = vec_to_named(opt.ask(), variables)
        if batch:
            dmin = min(normalized_distance(xn, b, variables) for b in batch)
            if dmin < diversity_eps:
                tries += 1
                continue
        batch.append(xn)
    return batch or [vec_to_named(opt.ask(), variables)]

# --------------- collect categorical values from RUNS ---------------
def observed_categorical_values_from_runs(pdf: pd.DataFrame, var_names_order: Sequence[str]) -> Dict[str, set]:
    """Scan RUNS to collect string values seen for each variable (categorical tokens)."""
    seen: Dict[str, set] = {}
    if pdf.empty or not var_names_order:
        return seen

    # map variable name -> row index in pdf
    name_to_row: Dict[str, int] = {}
    for i in range(pdf.shape[0]):
        label = str(pdf.iloc[i,0]).strip()
        if label in var_names_order:
            name_to_row[label] = i

    # Find the 'vars' header row to know which columns are experiments
    vars_header_idx = pdf.index[pdf.iloc[:,0].astype(str).str.strip().str.lower() == "vars"]
    if len(vars_header_idx) == 0:
        return seen
    vh = vars_header_idx[0]

    exp_cols = [j for j in range(1, pdf.shape[1]) if str(pdf.iloc[vh, j]).strip() != ""]
    for name in var_names_order:
        r = name_to_row.get(name, None)
        if r is None:
            continue
        vals: set = set()
        for j in exp_cols:
            val = pdf.iloc[r, j]
            if pd.isna(val) or str(val).strip() == "":
                continue
            if _num_from_cell(val) is None:  # treat as categorical token
                vals.add(str(val).strip())
        if vals:
            seen[name] = vals
    return seen

# --------------- warm start --------------------
def warmstart_from_runs(opt: po.Optimizer, experiments: Sequence[Optional[Dict[str, Any]]], variables: Sequence[Dict[str, Any]], outputs_cfg: Sequence[Dict[str, Any]]) -> None:
    if not experiments:
        if VERBOSE_WARMSTART:
            print("[warmstart] no experiment columns found.")
        return
    fed = 0
    for idx, e in enumerate(experiments, start=1):
        if not e:
            if VERBOSE_WARMSTART:
                print(f"[warmstart] skip column {idx}: missing at least one variable value.")
            continue
        x: List[Any] = []
        ok = True
        for v in variables:
            name, t = v["name"], v["type"]
            if name not in e:
                ok = False
                if VERBOSE_WARMSTART:
                    print(f"[warmstart] skip column {idx}: variable '{name}' missing.")
                break
            if t == "Categorical":
                x.append(str(e[name]))
            elif t == "Real":
                num = _num_from_cell(e[name])
                if num is None:
                    ok = False
                    if VERBOSE_WARMSTART:
                        print(f"[warmstart] skip column {idx}: variable '{name}' is non-numeric.")
                    break
                x.append(num)
            else:
                iv = _int_from_cell(e[name])
                if iv is None:
                    ok = False
                    if VERBOSE_WARMSTART:
                        print(f"[warmstart] skip column {idx}: variable '{name}' is non-integer.")
                    break
                x.append(iv)
        if not ok:
            continue
        y = compute_loss(outputs_cfg, e)
        if y is None:
            if VERBOSE_WARMSTART:
                print(f"[warmstart] skip column {idx}: at least one output missing or non-numeric.")
            continue
        try:
            opt.tell(x, y)
            fed += 1
        except Exception as err:
            if VERBOSE_WARMSTART:
                print(f"[warmstart] skip column {idx}: opt.tell failed ({err}).")
            continue
    print(f"[warmstart] fed {fed} completed run(s) into the optimizer.")

# --------------- loss report --------------------
def print_loss_report(runs_df: pd.DataFrame, outputs_cfg: Sequence[Dict[str, Any]]) -> None:
    if runs_df.empty:
        print("\n[loss] RUNS sheet is empty.")
        return

    # locate headers
    vars_header_idx = runs_df.index[runs_df.iloc[:,0].astype(str).str.strip().str.lower() == "vars"]
    if not len(vars_header_idx):
        print("\n[loss] No 'vars' header found in RUNS.")
        return
    vh = vars_header_idx[0]

    outs_header_idx = runs_df.index[runs_df.iloc[:,0].astype(str).str.strip().str.lower() == "outputs"]
    if not len(outs_header_idx):
        print("\n[loss] No 'outputs' header found in RUNS.")
        return
    oh = outs_header_idx[0]

    # gather output rows
    out_rows: List[int] = []
    r2 = oh + 1
    while r2 < len(runs_df):
        first = str(runs_df.iloc[r2,0]).strip()
        if first == "":
            break
        out_rows.append(r2)
        r2 += 1

    # experiment columns + names
    exp_cols: List[int] = []
    exp_names: List[str] = []
    for j in range(1, runs_df.shape[1]):
        hdr = str(runs_df.iloc[vh, j]).strip()
        if hdr != "":
            exp_cols.append(j)
            exp_names.append(hdr)

    print("\n=== Loss report (copy manually into RUNS → 'loss' row) ===")
    any_row = False
    for j, name in zip(exp_cols, exp_names):
        row_dict: Dict[str, Any] = {}
        have_all = True
        missing: List[str] = []
        for rr in out_rows:
            key = str(runs_df.iloc[rr,0]).strip()
            if key.lower() == "loss":
                continue
            val = runs_df.iloc[rr, j]
            if pd.isna(val) or str(val).strip() == "":
                have_all = False
                missing.append(key)
            else:
                row_dict[key] = val
        if have_all:
            y = compute_loss(outputs_cfg, row_dict)
            if y is None:
                print(f"  {name:>20}: cannot compute (non-numeric output)")
            else:
                print(f"  {name:>20}: loss = {y:.6g}")
                any_row = True
        else:
            print(f"  {name:>20}: missing outputs -> {', '.join(missing)}")
    if not any_row:
        print("  (no complete experiments with all outputs present)")

# --------------- persistence helpers --------------------

def space_signature(variables: Sequence[Dict[str, Any]]) -> Tuple:
    """Create a hashable signature of the current factor space."""
    sig: List[Tuple] = []
    for v in variables:
        if v["type"] == "Categorical":
            sig.append((v["name"], v["type"], tuple(sorted(map(str, v["choices"])))))
        elif v["type"] in ("Real", "Integer"):
            sig.append((v["name"], v["type"], float(v["low"]), float(v["high"])) )
        else:
            sig.append((v["name"], v["type"]))
    return tuple(sig)


def load_or_init_optimizer(space: po.Space, opts: Dict[str, Any], variables: Sequence[Dict[str, Any]]) -> po.Optimizer:
    """Load optimizer from STATE_PKL if compatible with the current space; otherwise init new."""
    sig = space_signature(variables)
    if STATE_PKL.exists():
        try:
            with open(STATE_PKL, "rb") as f:
                payload = pickle.load(f)
            # Backward compatibility: payload may be a bare Optimizer or a dict
            if isinstance(payload, po.Optimizer):
                print("[state] found legacy optimizer pickle; compatibility not guaranteed with changed space.")
                # If the space length differs, drop it; else reuse.
                if len(payload.space.dimensions) != len(space.dimensions):
                    print("[state] variable count changed → reinitializing optimizer.")
                else:
                    return payload
            elif isinstance(payload, dict) and "opt" in payload and "space_sig" in payload:
                if payload["space_sig"] == sig:
                    return payload["opt"]
                else:
                    print("[state] space changed since last run → reinitializing optimizer.")
        except Exception as e:
            print(f"[state] failed to load pickle ({e}); reinitializing optimizer.")

    # fresh optimizer
    return po.Optimizer(space,
                        base_estimator="GP",
                        n_initial_points=opts["n_initial_points"],
                        acq_func=opts["acq_func"])


def save_optimizer(opt: po.Optimizer, variables: Sequence[Dict[str, Any]]) -> None:
    payload = {"opt": opt, "space_sig": space_signature(variables)}
    with open(STATE_PKL, "wb") as f:
        pickle.dump(payload, f)

# ------------------ main -----------------------

def _maybe_plot_objective(opt: po.Optimizer, opts: Dict[str, Any]) -> None:
    """Optionally generate an objective plot from the current optimizer state.

    Saves to opts['plot_path'] with dpi=opts['plot_dpi'] using a non-interactive backend.
    """
    if not opts.get("plot_objective", False):
        return
    if matplotlib is None or plt is None:
        print("[plot] matplotlib unavailable; skipping plot_objective.")
        return
    try:
        result = opt.get_result()
        # ProcessOptimizer returns axes; grab current figure
        po.plot_objective(result)
        fig = plt.gcf()
        out_path = Path(opts.get("plot_path", "objective.png"))
        # If relative, drop next to the Excel file for convenience
        if not out_path.is_absolute():
            out_path = EXCEL_PATH.parent / out_path
        fig.savefig(out_path, dpi=int(opts.get("plot_dpi", 150)), bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] objective saved to: {out_path}")
    except Exception as e:
        print(f"[plot] failed to create objective plot: {e}")

def main() -> None:
    if not EXCEL_PATH.exists():
        sys.exit(f"Excel not found: {EXCEL_PATH}")

    # Read setup
    variables = read_setup_var(EXCEL_PATH)
    outputs   = read_setup_out(EXCEL_PATH)
    opts      = read_setup_opt(EXCEL_PATH)

    # Read RUNS early to discover categorical values actually used (e.g., 'precipitation')
    experiments, runs_df, var_names_order = read_runs_pivot(EXCEL_PATH)

    # Auto‑union categorical choices with observed tokens in RUNS
    observed = observed_categorical_values_from_runs(runs_df, var_names_order)
    changed_any = False
    for v in variables:
        if v["type"] == "Categorical":
            name = v["name"]
            seen_vals = set([c.strip() for c in v["choices"]])
            extra = {tok for tok in observed.get(name, set()) if tok not in seen_vals}
            if extra:
                v["choices"] = list(seen_vals.union(extra))
                changed_any = True
                print(f"[setup] extended categorical '{name}' choices with values from RUNS: {sorted(list(extra))}")
    if changed_any:
        print("[setup] categorical spaces updated to include all values seen in RUNS.")

    # Build space after union
    space = build_space(variables)

    # Load/init optimizer with compatibility check
    opt = load_or_init_optimizer(space, opts, variables)

    # Warm start from RUNS (only columns with all outputs)
    warmstart_from_runs(opt, experiments, variables, outputs)

    # Optional objective plot of the current GP mean surface
    _maybe_plot_objective(opt, opts)

    # Ask next suggestion(s)
    suggestions = ask_batch(opt, opts["batch_size"], variables, opts["diversity_eps"])

    print("\n=== Suggested conditions ===")
    for i, s in enumerate(suggestions, 1):
        print(f"\n-- Suggestion {i}/{len(suggestions)} --")
        for v in variables:
            name, unit = v["name"], v.get("unit", "")
            print(f"{name:>24}: {s[name]} {unit}".rstrip())

    # Persist optimizer state (with space signature)
    save_optimizer(opt, variables)

    # Print per‑experiment loss report (no Excel write)
    print_loss_report(runs_df, outputs)

    print(f"\nPrinted {len(suggestions)} suggestion(s).")
    print("Paste one suggestion into RUNS (new experiment column under 'vars'), run it, fill outputs, then re‑run.")
    print("Copy the 'loss' values from the Loss report into the RUNS sheet manually.")


if __name__ == "__main__":
    main()
