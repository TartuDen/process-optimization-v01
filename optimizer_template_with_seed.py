"""
optimizer_template_with_seed.py

Universal human-in-the-loop optimizer (ProcessOptimizer) with:
- CONFIG blocks for CONSTANTS, VARIABLES, OUTPUTS
- Embedded prior runs (EMBEDDED_PAST_RUNS) that are appended once to history.csv
- De-duplication so embedded runs are not re-added on subsequent executions
- CSV logging, resume/warm-start, optional batch suggestions, safe (non-crashy) constraints

Run:
  python "optimizer_template_with_seed.py"
"""

from pathlib import Path
import csv
import pickle
import ProcessOptimizer as po
from ProcessOptimizer.space import Real, Integer, Categorical

# ========================= USER CONFIG =========================

# 1) CONSTANTS (for your reference/derived calcs; not used by the optimizer directly)
CONSTANTS = {
    "substrate_g": 10.0,
    # "reactor_max_temp_C": 95.0,
    # "reactor_max_bar": 25.0,
}

# 2) VARIABLES (define your search space)
#   Choose simple names (letters/numbers/underscores) to keep CSV clean.
VARIABLES = [
    {"type": "Real",     "name": "T_C",        "low": 70.0, "high": 95.0, "unit": "°C"},
    {"type": "Real",     "name": "time_h",     "low": 1.0,  "high": 24.0, "unit": "h"},
    {"type": "Real",     "name": "reagent_eq", "low": 0.5,  "high": 3.0,  "unit": "equiv"},
    {"type": "Integer",  "name": "rpm",        "low": 200,  "high": 400,  "unit": "rpm"},
    {"type": "Categorical","name": "solvent",  "choices": ["MeOH","MeCN","IPA"]},
]

# 3) OUTPUTS (you will type these after each experiment)
#   kind: "min" (lower better), "max" (higher better), "target" (closer to target better)
OUTPUTS = [
    {"name": "yield_pct",  "kind": "max",    "target": 100.0, "weight": 0.25},
    {"name": "purity_pct", "kind": "max",    "target": 100.0, "weight": 0.25},
    {"name": "imp_A_pct",  "kind": "min",                     "weight": 0.30},
    {"name": "imp_B_pct",  "kind": "min",                     "weight": 0.20},
]

# 4) EMBEDDED PAST RUNS (optional) — add your previously-conducted experiments here.
#    Each item has "vars" (exactly the VARIABLE names) and "outputs" (OUTPUT names).
#    On first run, any *new* rows append to history.csv; on later runs they’re skipped.
EMBEDDED_PAST_RUNS = [
    # EXAMPLE rows — replace with your data or leave empty list [].
    # {
    #   "vars":    {"T_C": 80.0, "time_h": 6.0, "reagent_eq": 1.2, "rpm": 300, "solvent": "MeOH"},
    #   "outputs": {"yield_pct": 72.5, "purity_pct": 93.1, "imp_A_pct": 1.8, "imp_B_pct": 0.6}
    # },
    # {
    #   "vars":    {"T_C": 82.0, "time_h": 8.0, "reagent_eq": 1.0, "rpm": 350, "solvent": "MeCN"},
    #   "outputs": {"yield_pct": 74.2, "purity_pct": 95.0, "imp_A_pct": 1.1, "imp_B_pct": 0.4}
    # },
]

# Optimizer & batching
ACQ_FUNC = "EI"
N_INITIAL_POINTS = 6
BATCH_SIZE = 1               # set 3–6 to get multiple diverse suggestions per run
DIVERSITY_EPS = 0.05         # min normalized distance between batch points (0..1)

# (Optional) constraints — disabled by default and safe if left empty
ENABLE_CONSTRAINTS  = False
STRICT_CONSTRAINTS  = False
VERBOSE_CONSTRAINTS = False

def example_guard_discrete_rpm(x, const):
    rpm = x.get("rpm")
    return True if rpm is None else (int(rpm) % 25 == 0)

CONSTRAINTS = [
    # example_guard_discrete_rpm,
]

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
        if t == "Real":
            out[name] = float(out[name])
        elif t == "Integer":
            out[name] = int(out[name])
    return out

def compute_loss(outputs):
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
    return {cfg["name"]: prompt_float(f"  {cfg['name']}") for cfg in OUTPUTS}

def append_history(row):
    var_names = [v["name"] for v in VARIABLES]
    out_names = [o["name"] for o in OUTPUTS]
    derived_keys = sorted([k for k in row.keys() if k not in var_names + out_names + ["loss"]])
    header = var_names + derived_keys + out_names + ["loss"]
    new = not HISTORY_CSV.exists()
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new: w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})

# ---- Safe constraints runner
def run_constraints(x_dict):
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

# ---- Diversity metric for batch selection
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

def ask_batch(opt, k):
    if k <= 1:
        for _ in range(50):
            xn = vec_to_named(opt.ask())
            if run_constraints(xn):
                return [xn]
        return [vec_to_named(opt.ask())]
    batch = []
    tries = 0
    while len(batch) < k and tries < 500:
        xn = vec_to_named(opt.ask())
        if not run_constraints(xn):
            tries += 1; continue
        if batch:
            dmin = min(normalized_distance(xn, b) for b in batch)
            if dmin < DIVERSITY_EPS:
                tries += 1; continue
        batch.append(xn)
    return batch or [vec_to_named(opt.ask())]

# ---- Persistence
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
                    val = row.get(name, "")
                    if val == "":
                        x = None; break
                    if t == "Categorical":
                        x.append(val)
                    elif t == "Real":
                        x.append(float(val))
                    else:
                        x.append(int(float(val)))
                if x is None: continue
                if "loss" in row and row["loss"] not in ("", None):
                    X.append(x); y.append(float(row["loss"]))
            if X: opt.tell(X, y)
    except Exception:
        pass

# ---- Embedded seeding with de-duplication
def _key_from_vars(var_dict, decimals=8):
    """Create a hashable key from variable values in canonical order."""
    key = []
    for v in VARIABLES:
        name, t = v["name"], v["type"]
        val = var_dict.get(name, None)
        if t == "Categorical":
            key.append(("C", str(val)))
        elif t == "Real":
            key.append(("R", round(float(val), decimals)))
        else:
            key.append(("I", int(val)))
    return tuple(key)

def _existing_keys_from_history():
    keys = set()
    if not HISTORY_CSV.exists():
        return keys
    with open(HISTORY_CSV, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            present = True
            vd = {}
            for v in VARIABLES:
                name, t = v["name"], v["type"]
                if row.get(name, "") == "":
                    present = False; break
                if t == "Categorical":
                    vd[name] = row[name]
                elif t == "Real":
                    vd[name] = float(row[name])
                else:
                    vd[name] = int(float(row[name]))
            if present:
                keys.add(_key_from_vars(vd))
    return keys

def seed_embedded_runs():
    """Append embedded runs once; skip duplicates based on VARIABLES only."""
    if not EMBEDDED_PAST_RUNS:
        return 0
    existing = _existing_keys_from_history()
    added = 0
    for item in EMBEDDED_PAST_RUNS:
        vars_dict = item.get("vars", {})
        outs_dict = item.get("outputs", {})
        # verify all required names exist
        if any(v["name"] not in vars_dict for v in VARIABLES):
            continue
        if any(o["name"] not in outs_dict for o in OUTPUTS):
            continue
        key = _key_from_vars(vars_dict)
        if key in existing:
            continue  # already in CSV
        # compute loss and append
        loss = compute_loss(outs_dict)
        row = {}
        # variables
        for v in VARIABLES:
            name, t = v["name"], v["type"]
            val = vars_dict[name]
            if t == "Categorical":
                row[name] = str(val)
            elif t == "Real":
                row[name] = float(val)
            else:
                row[name] = int(val)
        # (optional) derived fields could be added here if you like
        # outputs + loss
        for o in OUTPUTS:
            row[o["name"]] = float(outs_dict[o["name"]])
        row["loss"] = loss
        append_history(row)
        added += 1
        existing.add(key)
    # warm-start optimizer with newly added rows (only the new ones)
    if added > 0:
        opt = load_or_init_optimizer()
        warm_start(opt)
        with open(STATE_PKL, "wb") as f:
            pickle.dump(opt, f)
    return added

# -------- MAIN
def main():
    # 1) Seed embedded runs once (de-duplicated)
    added = seed_embedded_runs()
    if added:
        print(f"[seed] Added {added} embedded past run(s) to history.csv")

    # 2) Load/init optimizer & warm start from history
    opt = load_or_init_optimizer()
    warm_start(opt)

    # 3) Ask for next suggestion(s)
    suggestions = ask_batch(opt, BATCH_SIZE)

    print("\n=== Suggested conditions ===")
    for i, s in enumerate(suggestions, 1):
        print(f"\n-- Suggestion {i}/{len(suggestions)} --")
        for v in VARIABLES:
            name, unit = v["name"], v.get("unit", "")
            val = s[name]
            print(f"{name:>18}: {val} {unit}".rstrip())

    # 4) Collect outcomes and update model
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

        # log row
        row = {v["name"]: s[v["name"]] for v in VARIABLES}
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
