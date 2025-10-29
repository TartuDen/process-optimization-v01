"""
optimizer_template.py

General human-in-the-loop Bayesian optimizer using ProcessOptimizer.
- Configure CONSTANTS, VARIABLES, OUTPUTS, and LOSS at the top.
- The script suggests conditions, you run the experiment, you type measured outputs.
- It logs to CSV and persists optimizer state so you can stop/resume any time.
"""

from pathlib import Path
import csv
import pickle
import math
import random
import ProcessOptimizer as po
from ProcessOptimizer.space import Real, Integer, Categorical

# ========================= USER CONFIG =========================

# 1) CONSTANTS: put anything you want to keep fixed here
CONSTANTS = {
    "substrate_g": 11.0,
    "reactor_max_bar": 25.0,
    "reactor_max_temp_C": 120.0,
    # Add your own: "substrate_mw": 250.31, "catalyst_grade": "10% Pd/C (wet)", ...
}

# 2) VARIABLES: define your decision variables (bounds, units, etc.)
# Types supported: Real, Integer, Categorical
VARIABLES = [
    # Typical organic synthesis knobs (edit/delete/add as needed)
    {"type": "Real",     "name": "T_C",             "low": 20.0,  "high": 95.0,  "unit": "°C"},
    {"type": "Real",     "name": "H2_bar",          "low": 1.0,   "high": 25.0,  "unit": "bar"},
    {"type": "Real",     "name": "time_h",          "low": 0.25,  "high": 24.0,  "unit": "h"},
    {"type": "Real",     "name": "reagent_eq",      "low": 0.5,   "high": 3.0,   "unit": "equiv"},
    {"type": "Real",     "name": "conc_mol_L",      "low": 0.05,  "high": 2.0,   "unit": "mol/L"},
    {"type": "Real",     "name": "feed_rate_mL_min","low": 0.1,   "high": 10.0,  "unit": "mL/min"},
    {"type": "Real",     "name": "solv_mL_per_g",   "low": 5.0,   "high": 40.0,  "unit": "mL/g"},
    {"type": "Integer",  "name": "rpm",             "low": 200,   "high": 400,   "unit": "rpm"},  # or step multiples
    {"type": "Categorical","name": "solvent",       "choices": ["MeOH","MeCN","IPA","THF"]},
    # Add others commonly used in org synth:
    # {"type":"Real","name":"pH_set","low":1.0,"high":12.0,"unit":"pH"},
    # {"type":"Real","name":"pressure_bar","low":1.0,"high":10.0,"unit":"bar"},
    # {"type":"Categorical","name":"base","choices":["Et3N","DIPEA","Na2CO3"]},
]

# 3) OUTPUTS: what you will type in after each run (measured)
# For each output, define how it contributes to the loss we minimize.
# kind: "min" (lower is better), "max" (higher is better), or "target" (closer to target is better)
# target: only used when kind == "target" OR to define a “good” reference for % scales
# weight: relative importance in the final loss (weights are linear)
OUTPUTS = [
    {"name": "yield_pct",       "kind": "max",    "target": 100.0, "weight": 0.20},
    {"name": "purity_pct",      "kind": "max",    "target": 100.0, "weight": 0.20},
    {"name": "imp_RRT119_pct",  "kind": "min",                    "weight": 0.30},
    {"name": "imp_RRT120_pct",  "kind": "min",                    "weight": 0.20},
    # Examples you can add:
    # {"name":"residual_solvent_ppm","kind":"min","weight":0.10},
    # {"name":"cycle_time_h","kind":"min","weight":0.10},
    # {"name":"assay_pct","kind":"max","target":100.0,"weight":0.10},
    # {"name":"pH_final","kind":"target","target":7.0,"weight":0.10},
]

# Loss settings
ACQ_FUNC = "EI"              # "EI" is robust; others: "LCB", "PI"
N_INITIAL_POINTS = 6         # initial space-filling before BO
BATCH_SIZE = 1               # set >1 to propose multiple suggestions per run
DIVERSITY_EPS = 0.05         # minimum normalized distance between batch points (0..1), if BATCH_SIZE > 1

# Feasibility / safety constraints (all must return True)
def _guard_basic(_x):  # example: keep under hardware limits
    return (_x["T_C"] <= CONSTANTS["reactor_max_temp_C"]) and (_x["H2_bar"] <= CONSTANTS["reactor_max_bar"])

def _guard_interactions(_x):  # example: avoid too hot & too concentrated simultaneously
    return not (_x["T_C"] > 90 and _x["conc_mol_L"] > 1.5)

def _guard_practical(_x):  # example: rpm discrete steps of 25
    return (int(_x["rpm"]) % 25 == 0)

CONSTRAINTS = [_guard_basic, _guard_interactions, _guard_practical]

# Files
HISTORY_CSV = Path("history.csv")
STATE_PKL   = Path("optimizer.pkl")

# ====================== END USER CONFIG =======================


# -------- Helper: build ProcessOptimizer Space from VARIABLES
def build_space(variables):
    dims = []
    for v in variables:
        if v["type"] == "Real":
            dims.append(Real(v["low"], v["high"], name=v["name"]))
        elif v["type"] == "Integer":
            dims.append(Integer(v["low"], v["high"], name=v["name"]))
        elif v["type"] == "Categorical":
            dims.append(Categorical(v["choices"], name=v["name"]))
        else:
            raise ValueError(f"Unknown variable type: {v['type']}")
    return po.Space(dims)

SPACE = build_space(VARIABLES)


# -------- Helper: convert vector x -> dict with names
def vec_to_named(x_vec):
    out = {}
    for i, v in enumerate(VARIABLES):
        out[v["name"]] = x_vec[i]
        # optional rounding for pretty printing
        if v["type"] in ("Real", "Integer") and isinstance(out[v["name"]], float):
            out[v["name"]] = float(out[v["name"]])
    return out


# -------- Helper: constraints
def is_feasible(x_dict):
    return all(fn(x_dict) for fn in CONSTRAINTS)


# -------- Helper: normalized distance between two points for diversity
def normalized_distance(xa, xb):
    # Scale numeric dims 0..1, categoricals 0 (same) or 1 (diff)
    num = 0.0; cnt = 0
    for v in VARIABLES:
        name = v["name"]
        a, b = xa[name], xb[name]
        if v["type"] == "Categorical":
            num += 0.0 if a == b else 1.0
            cnt += 1
        elif v["type"] in ("Real","Integer"):
            span = (v["high"] - v["low"])
            if span <= 0: continue
            num += abs((float(a) - float(b)) / span)
            cnt += 1
    return num / max(cnt, 1)


# -------- Suggest K feasible, diverse points
def ask_batch(opt, k):
    batch = []
    tries = 0
    while len(batch) < k and tries < 200:
        x = opt.ask()
        x_named = vec_to_named(x)
        if not is_feasible(x_named):
            tries += 1
            continue
        if batch:
            # keep some diversity
            dmin = min(normalized_distance(x_named, b) for b in batch)
            if dmin < DIVERSITY_EPS:
                tries += 1
                continue
        batch.append(x_named)
    # if we failed to fill the batch, just return whatever we got (maybe 0..k)
    return batch


# -------- Persistence
def load_or_init_optimizer():
    if STATE_PKL.exists():
        with open(STATE_PKL, "rb") as f:
            return pickle.load(f)
    return po.Optimizer(SPACE, base_estimator="GP", n_initial_points=N_INITIAL_POINTS, acq_func=ACQ_FUNC)

def warm_start(opt):
    if not HISTORY_CSV.exists():
        return
    X, y = [], []
    with open(HISTORY_CSV, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            x = []
            for v in VARIABLES:
                if v["type"] == "Categorical":
                    x.append(row[v["name"]])
                else:
                    x.append(float(row[v["name"]]))
            X.append(x)
            y.append(float(row["loss"]))
    if X:
        opt.tell(X, y)


# -------- Derived/auxiliary calculations (optional, for display/logging)
def compute_derived(x):
    """Return a dict of extra fields you want to log (human-readable helpers)."""
    d = {}
    # Example: total solvent volume
    if "solv_mL_per_g" in x and "substrate_g" in CONSTANTS:
        d["total_solvent_mL"] = float(x["solv_mL_per_g"]) * float(CONSTANTS["substrate_g"])
    # Example: simple space-time yield if user later provides yield and time
    return d


# -------- Loss function factory
def compute_loss_from_outputs(outputs_dict):
    """
    Turn user-entered outputs into a single scalar loss.
    Lower is better.
    Implements three behaviors:
      - kind == "min":   loss += weight * value
      - kind == "max":   loss += weight * max(0, target - value)      (target defaults to 100 if not set)
      - kind == "target":loss += weight * abs(value - target)
    Tip: For ppm/area% impurities, use kind="min".
         For % yield/purity, use kind="max", target=100.
         For setpoints (e.g., pH_final), use kind="target", target=<desired>.
    """
    total = 0.0
    for cfg in OUTPUTS:
        name   = cfg["name"]
        kind   = cfg.get("kind", "min")
        weight = float(cfg.get("weight", 1.0))
        target = float(cfg.get("target", 100.0))  # sensible default for percentages

        if name not in outputs_dict:
            raise ValueError(f"Missing output '{name}' in entered results.")

        val = float(outputs_dict[name])

        if kind == "min":
            pen = val
        elif kind == "max":
            pen = max(0.0, target - val)
        elif kind == "target":
            pen = abs(val - target)
        else:
            raise ValueError(f"Unknown kind '{kind}' for output '{name}'")

        total += weight * pen
    return total


# -------- I/O helpers
def prompt_float(label):
    while True:
        try:
            return float(input(f"{label}: ").strip())
        except Exception:
            print("Please enter a number, e.g., 75.6")

def prompt_outputs():
    print("\nEnter measured outcomes:")
    out = {}
    for cfg in OUTPUTS:
        n = cfg["name"]
        out[n] = prompt_float(f"  {n}")
    return out


# -------- CSV logging
def append_history(row):
    # header includes variables, constants (optional), derived, outputs, loss
    var_names = [v["name"] for v in VARIABLES]
    out_names = [o["name"] for o in OUTPUTS]
    header = var_names + out_names + ["loss"]

    # include derived columns deterministically (keys sorted for stable CSV)
    derived_keys = sorted([k for k in row.keys() if k not in header and k not in var_names])
    header = var_names + derived_keys + out_names + ["loss"]

    new = not HISTORY_CSV.exists()
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})


# -------- Main loop
def main():
    opt = load_or_init_optimizer()
    warm_start(opt)

    # Propose a batch of K points
    suggestions = ask_batch(opt, BATCH_SIZE) or []
    if not suggestions:
        # Fallback single suggestion if batch selection failed (rare)
        suggestions = [vec_to_named(opt.ask())]

    print("\n=== Suggested conditions ===")
    for i, s in enumerate(suggestions, 1):
        print(f"\n-- Suggestion {i}/{len(suggestions)} --")
        for v in VARIABLES:
            name = v["name"]
            val  = s[name]
            unit = v.get("unit", "")
            if unit:
                print(f"{name:>18}: {val} {unit}")
            else:
                print(f"{name:>18}: {val}")
        derived = compute_derived(s)
        for k, v in derived.items():
            print(f"{k:>18}: {v}")

    # For each suggestion, collect outcomes and tell optimizer
    for i, s in enumerate(suggestions, 1):
        print(f"\n=== Record outcomes for suggestion {i} ===")
        measured = prompt_outputs()
        loss = compute_loss_from_outputs(measured)

        # Tell optimizer
        x_vec = []
        for v in VARIABLES:
            name = v["name"]
            if v["type"] == "Categorical":
                x_vec.append(s[name])
            else:
                x_vec.append(float(s[name]))
        opt.tell(x_vec, loss)

        # Persist state
        with open(STATE_PKL, "wb") as f:
            pickle.dump(opt, f)

        # Log
        row = {}
        for v in VARIABLES:
            row[v["name"]] = s[v["name"]]
        derived = compute_derived(s)
        row.update(derived)
        for k, v in measured.items():
            row[k] = v
        row["loss"] = loss
        append_history(row)

        res = opt.get_result()
        print("\nLogged. Current best (lowest loss):")
        print(f"  x* = {res.x}")
        print(f"  loss* = {res.fun:.4f}")

    print("\nRun the script again to get the next suggestion batch.")


if __name__ == "__main__":
    main()
