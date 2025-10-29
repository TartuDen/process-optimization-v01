"""
opicaTP.1_optimizer.py — fixed for your current case

Variables:
- T_C (80..85 °C)
- NH2OH_HCl_eq (2.0..3.0 equiv)
- EtOH_vol (2..10 arbitrary "volume units")
- H2O_vol (0..10 arbitrary "volume units")
- solvent (ETOH or H2O)  <-- keep only if you truly need a categorical switch

Outputs:
- yield_pct (maximize)
- purity_pct (maximize)
- imp_RRT3_16_pct (minimize)   # renamed to avoid dot in key name
"""

from pathlib import Path
import csv
import pickle
import ProcessOptimizer as po
from ProcessOptimizer.space import Real, Integer, Categorical

# ========================= USER CONFIG =========================

CONSTANTS = {
    "substrate_g": 10.0,
    # add hardware limits here if you want to enforce them, e.g.:
    # "reactor_max_temp_C": 120.0,
}

VARIABLES = [
    {"type": "Real", "name": "T_C",             "low": 80.0, "high": 85.0, "unit": "°C"},
    {"type": "Real", "name": "NH2OH_HCl_eq",    "low": 2.0,  "high": 3.0,  "unit": "equiv"},
    {"type": "Real", "name": "EtOH_vol",        "low": 2.0,  "high": 10.0, "unit": "vol"},
    {"type": "Real", "name": "H2O_vol",         "low": 0.0,  "high": 10.0, "unit": "vol"},
    {"type": "Categorical","name": "solvent",   "choices": ["ETOH","H2O"]},
]

OUTPUTS = [
    {"name": "yield_pct",       "kind": "max", "target": 100.0, "weight": 0.10},
    {"name": "purity_pct",      "kind": "max", "target": 100.0, "weight": 0.10},
    {"name": "imp_RRT3_16_pct", "kind": "min",                 "weight": 0.80},
]

ACQ_FUNC = "EI"
N_INITIAL_POINTS = 6
BATCH_SIZE = 1
DIVERSITY_EPS = 0.05

# -------- Feasibility / safety constraints (edit to taste)
def _guard_basic(x):
    # Example: keep temperature within given bounds (already enforced by Space, but belt & suspenders)
    if not (80.0 <= float(x["T_C"]) <= 85.0):
        return False
    # If you add hardware limits later, check here (e.g., "reactor_max_temp_C" in CONSTANTS)
    return True

def _guard_volumes(x):
    # avoid both solvents being zero; also keep sum in a reasonable window if you want
    etoh = float(x["EtOH_vol"])
    h2o  = float(x["H2O_vol"])
    if etoh <= 0.0 and h2o <= 0.0:
        return False
    # Example optional: cap total volume to reduce extreme dilution
    # if etoh + h2o > 18.0: return False
    return True

def _guard_consistency(x):
    # If you keep 'solvent' categorical, you may want it consistent with dominant volume
    # Not required, but here's an example rule:
    if x["solvent"] == "ETOH" and float(x["H2O_vol"]) > float(x["EtOH_vol"]):
        return False
    if x["solvent"] == "H2O" and float(x["EtOH_vol"]) > float(x["H2O_vol"]):
        return False
    return True

CONSTRAINTS = [_guard_basic, _guard_volumes, _guard_consistency]

# Files
HISTORY_CSV = Path("opicaTP.1_history.csv")
STATE_PKL   = Path("opicaTP.1_optimizer.pkl")

# ====================== END USER CONFIG =======================

# -------- Build Space
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

# -------- Helpers
def vec_to_named(x_vec):
    out = {}
    for i, v in enumerate(VARIABLES):
        out[v["name"]] = x_vec[i]
        if v["type"] in ("Real", "Integer"):
            out[v["name"]] = float(out[v["name"]])
    return out

def is_feasible(x_dict):
    return all(fn(x_dict) for fn in CONSTRAINTS)

def normalized_distance(xa, xb):
    num = 0.0; cnt = 0
    for v in VARIABLES:
        name = v["name"]
        a, b = xa[name], xb[name]
        if v["type"] == "Categorical":
            num += 0.0 if a == b else 1.0
            cnt += 1
        else:
            span = (v["high"] - v["low"])
            if span <= 0: continue
            num += abs((float(a) - float(b)) / span)
            cnt += 1
    return num / max(cnt, 1)

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
            dmin = min(normalized_distance(x_named, b) for b in batch)
            if dmin < DIVERSITY_EPS:
                tries += 1
                continue
        batch.append(x_named)
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

# -------- Derived (optional)
def compute_derived(x):
    d = {}
    # If later you add mL/g variables, compute totals here
    return d

# -------- Loss
def compute_loss_from_outputs(outputs_dict):
    total = 0.0
    for cfg in OUTPUTS:
        name   = cfg["name"]
        kind   = cfg.get("kind", "min")
        weight = float(cfg.get("weight", 1.0))
        target = float(cfg.get("target", 100.0))
        if name not in outputs_dict:
            raise ValueError(f"Missing output '{name}'")
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

# -------- I/O
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
    var_names = [v["name"] for v in VARIABLES]
    out_names = [o["name"] for o in OUTPUTS]
    header = var_names + out_names + ["loss"]
    derived_keys = sorted([k for k in row.keys() if k not in header and k not in var_names])
    header = var_names + derived_keys + out_names + ["loss"]
    new = not HISTORY_CSV.exists()
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new: w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})

# -------- Main
def main():
    opt = load_or_init_optimizer()
    warm_start(opt)

    suggestions = ask_batch(opt, BATCH_SIZE) or [vec_to_named(opt.ask())]

    print("\n=== Suggested conditions ===")
    for i, s in enumerate(suggestions, 1):
        print(f"\n-- Suggestion {i}/{len(suggestions)} --")
        for v in VARIABLES:
            name = v["name"]; val = s[name]; unit = v.get("unit", "")
            print(f"{name:>18}: {val} {unit}".rstrip())
        derived = compute_derived(s)
        for k, v in derived.items():
            print(f"{k:>18}: {v}")

    for i, s in enumerate(suggestions, 1):
        print(f"\n=== Record outcomes for suggestion {i} ===")
        measured = prompt_outputs()
        loss = compute_loss_from_outputs(measured)

        x_vec = []
        for v in VARIABLES:
            name = v["name"]
            x_vec.append(s[name] if v["type"] == "Categorical" else float(s[name]))
        opt.tell(x_vec, loss)

        with open(STATE_PKL, "wb") as f:
            pickle.dump(opt, f)

        row = {v["name"]: s[v["name"]] for v in VARIABLES}
        row.update(compute_derived(s))
        row.update(measured)
        row["loss"] = loss
        append_history(row)

        res = opt.get_result()
        print("\nLogged. Current best (lowest loss):")
        print(f"  x* = {res.x}")
        print(f"  loss* = {res.fun:.4f}")

    print("\nRun the script again to get the next suggestion batch.")

if __name__ == "__main__":
    main()
