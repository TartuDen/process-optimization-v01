"""
Universal human-in-the-loop optimizer using ProcessOptimizer.

- Edit CONSTANTS, VARIABLES, OUTPUTS only.
- Constraints are optional; by default they are disabled or auto-skipped if ill-defined.
- Logs to CSV, persists state, supports batch suggestions with diversity.
"""

from pathlib import Path
import csv
import pickle
import ProcessOptimizer as po
from ProcessOptimizer.space import Real, Integer, Categorical

# ========================= USER CONFIG =========================

# 1) CONSTANTS (purely for your reference/derived calcs; not used by the optimizer)
CONSTANTS = {
    "substrate_g": 10.0,
    # Add more if useful, e.g.:
    # "reactor_max_bar": 25.0,
    # "reactor_max_temp_C": 120.0,
}

# 2) VARIABLES (only change this list to define your search space)
#   Supported types: "Real", "Integer", "Categorical"
#   Pick names that are simple (letters, numbers, underscores) for cleaner CSVs.
VARIABLES = [
    {"type": "Real", "name": "T_C",          "low": 70.0, "high": 95.0, "unit": "°C"},
    {"type": "Real", "name": "time_h",       "low": 1.0,  "high": 24.0, "unit": "h"},
    {"type": "Real", "name": "reagent_eq",   "low": 0.5,  "high": 3.0,  "unit": "equiv"},
    {"type": "Integer", "name": "rpm",       "low": 200,  "high": 400,  "unit": "rpm"},
    {"type": "Categorical", "name": "solvent", "choices": ["MeOH", "MeCN", "IPA"]},
]

# 3) OUTPUTS (what you type in after each experiment)
#   kind: "min" (lower better), "max" (higher better), "target" (closer to target better)
#   weight: importance in total loss (we minimize the weighted sum of penalties)
OUTPUTS = [
    {"name": "yield_pct",  "kind": "max",    "target": 100.0, "weight": 0.25},
    {"name": "purity_pct", "kind": "max",    "target": 100.0, "weight": 0.25},
    {"name": "imp_A_pct",  "kind": "min",                     "weight": 0.30},
    {"name": "imp_B_pct",  "kind": "min",                     "weight": 0.20},
]

# Optimizer & batching
ACQ_FUNC = "EI"             # "EI" is a robust default
N_INITIAL_POINTS = 6        # space-filling points before BO
BATCH_SIZE = 1              # set to 3–6 to get multiple diverse suggestions per bench run
DIVERSITY_EPS = 0.05        # min normalized distance between suggestions if BATCH_SIZE>1 (0..1)

# ---------- CONSTRAINTS (optional) ----------
# If you want constraints, define them **here** as callables: f(x, const) -> bool.
# Each function receives:
#   x:     dict of {"var_name": value} for current suggestion
#   const: CONSTANTS dict
# Return True if feasible, False if infeasible.
# IMPORTANT: These are optional. By default they are DISABLED (see flags below).

def example_guard_temperature(x, const):
    # keep temperature under a hardware max if present
    maxT = const.get("reactor_max_temp_C", None)
    return True if maxT is None else float(x.get("T_C", -1e9)) <= maxT

def example_guard_discrete_rpm(x, const):
    # only multiples of 25 rpm (works even if rpm not present)
    rpm = x.get("rpm", None)
    return True if rpm is None else (int(rpm) % 25 == 0)

CONSTRAINTS = [
    # example_guard_temperature,
    # example_guard_discrete_rpm,
]

# Constraint behavior flags:
ENABLE_CONSTRAINTS   = False   # set True to enforce constraints
STRICT_CONSTRAINTS   = False   # when True: missing keys in a rule => rule fails; when False: skip that rule
VERBOSE_CONSTRAINTS  = False   # print why a rule was skipped/failed

# Files
HISTORY_CSV = Path("history.csv")
STATE_PKL   = Path("optimizer.pkl")

# ====================== END USER CONFIG =======================


# -------- Build optimization Space
def build_space(variables):
    dims = []
    for v in variables:
        t = v["type"]
        if t == "Real":
            dims.append(Real(v["low"], v["high"], name=v["name"]))
        elif t == "Integer":
            dims.append(Integer(v["low"], v["high"], name=v["name"]))
        elif t == "Categorical":
            dims.append(Categorical(v["choices"], name=v["name"]))
        else:
            raise ValueError(f"Unknown variable type: {t}")
    return po.Space(dims)

SPACE = build_space(VARIABLES)


# -------- Utilities
def vec_to_named(x_vec):
    out = {}
    for i, v in enumerate(VARIABLES):
        name, t = v["name"], v["type"]
        out[name] = x_vec[i]
        if t in ("Real", "Integer"):
            out[name] = float(out[name]) if t == "Real" else int(out[name])
    return out

def normalized_distance(a, b):
    num = 0.0; cnt = 0
    for v in VARIABLES:
        name, t = v["name"], v["type"]
        va, vb = a[name], b[name]
        if t == "Categorical":
            num += 0.0 if va == vb else 1.0
            cnt += 1
        else:
            span = float(v["high"] - v["low"])
            if span <= 0: continue
            num += abs((float(va) - float(vb)) / span)
            cnt += 1
    return num / max(cnt, 1)

def run_constraints(x_dict):
    """Universal, non-crashy constraint runner."""
    if not ENABLE_CONSTRAINTS or not CONSTRAINTS:
        return True
    for fn in CONSTRAINTS:
        try:
            ok = fn(x_dict, CONSTANTS)
        except KeyError as e:
            if STRICT_CONSTRAINTS:
                if VERBOSE_CONSTRAINTS: print(f"[constraint FAIL missing key] {fn.__name__}: {e}")
                return False
            else:
                if VERBOSE_CONSTRAINTS: print(f"[constraint SKIP missing key] {fn.__name__}: {e}")
                continue
        except Exception as e:
            # Any other runtime error: strict->fail, non-strict->skip
            if STRICT_CONSTRAINTS:
                if VERBOSE_CONSTRAINTS: print(f"[constraint FAIL error] {fn.__name__}: {e}")
                return False
            else:
                if VERBOSE_CONSTRAINTS: print(f"[constraint SKIP error] {fn.__name__}: {e}")
                continue
        if not ok:
            if VERBOSE_CONSTRAINTS: print(f"[constraint REJECT] {fn.__name__}")
            return False
    return True

def ask_batch(opt, k):
    if k <= 1:
        # Single suggestion, retry a few times to find a feasible point
        for _ in range(50):
            xn = vec_to_named(opt.ask())
            if run_constraints(xn):
                return [xn]
        return [vec_to_named(opt.ask())]  # last resort
    # Batch with diversity
    batch = []
    tries = 0
    while len(batch) < k and tries < 500:
        xn = vec_to_named(opt.ask())
        if not run_constraints(xn):
            tries += 1
            continue
        if batch:
            dmin = min(normalized_distance(xn, b) for b in batch)
            if dmin < DIVERSITY_EPS:
                tries += 1
                continue
        batch.append(xn)
    return batch or [vec_to_named(opt.ask())]

def load_or_init_optimizer():
    if STATE_PKL.exists():
        with open(STATE_PKL, "rb") as f:
            return pickle.load(f)
    return po.Optimizer(SPACE, base_estimator="GP", n_initial_points=N_INITIAL_POINTS, acq_func=ACQ_FUNC)

def warm_start(opt):
    if not HISTORY_CSV.exists():
        return
    try:
        with open(HISTORY_CSV, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            X, y = [], []
            for row in r:
                x = []
                for v in VARIABLES:
                    name, t = v["name"], v["type"]
                    if t == "Categorical":
                        x.append(row.get(name))
                    elif row.get(name) not in (None, ""):
                        x.append(float(row[name]) if t == "Real" else int(float(row[name])))
                    else:
                        # Missing value -> skip this row
                        x = None; break
                if x is None:
                    continue
                if "loss" in row and row["loss"] not in (None, ""):
                    X.append(x); y.append(float(row["loss"]))
            if X:
                opt.tell(X, y)
    except Exception:
        # If warm start fails for any reason, just ignore and start fresh
        pass

def compute_derived(x):
    """Optional derived columns. Keep it safe (check keys)."""
    d = {}
    if "solv_mL_per_g" in x and "substrate_g" in CONSTANTS:
        d["total_solvent_mL"] = float(x["solv_mL_per_g"]) * float(CONSTANTS["substrate_g"])
    return d

def compute_loss(outputs):
    """Aggregate outputs into a scalar loss (lower is better)."""
    total = 0.0
    for cfg in OUTPUTS:
        n = cfg["name"]; kind = cfg.get("kind", "min")
        w = float(cfg.get("weight", 1.0))
        tgt = float(cfg.get("target", 100.0))
        if n not in outputs:
            raise ValueError(f"Missing output '{n}'")
        val = float(outputs[n])
        if kind == "min":
            pen = val
        elif kind == "max":
            pen = max(0.0, tgt - val)
        elif kind == "target":
            pen = abs(val - tgt)
        else:
            raise ValueError(f"Unknown kind '{kind}' for output '{n}'")
        total += w * pen
    return total

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
        out[cfg["name"]] = prompt_float(f"  {cfg['name']}")
    return out

def append_history(row):
    var_names = [v["name"] for v in VARIABLES]
    out_names = [o["name"] for o in OUTPUTS]
    # include derived columns deterministically
    derived_keys = sorted([k for k in row.keys() if k not in var_names + out_names + ["loss"]])
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

    suggestions = ask_batch(opt, BATCH_SIZE)

    print("\n=== Suggested conditions ===")
    for i, s in enumerate(suggestions, 1):
        print(f"\n-- Suggestion {i}/{len(suggestions)} --")
        for v in VARIABLES:
            name, unit = v["name"], v.get("unit", "")
            val = s[name]
            print(f"{name:>18}: {val} {unit}".rstrip())
        for k, v in compute_derived(s).items():
            print(f"{k:>18}: {v}")

    for i, s in enumerate(suggestions, 1):
        print(f"\n=== Record outcomes for suggestion {i} ===")
        measured = prompt_outputs()
        loss = compute_loss(measured)

        x_vec = []
        for v in VARIABLES:
            name, t = v["name"], v["type"]
            x_vec.append(s[name] if t == "Categorical" else (float(s[name]) if t == "Real" else int(s[name])))
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
