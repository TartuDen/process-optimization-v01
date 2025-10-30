# optimize_from_excel.py
# Excel-driven Bayesian optimization (ProcessOptimizer) with a pivot RUNS sheet.
# Layout:
#   TEMPLATE      (ignored)
#   SETUP-CONST   section | name | value
#   SETUP-VAR     section | name | type | low | high | choices | unit | active
#   SETUP-OUT     section | name | kind | target | weight
#   SETUP-OPT     section | name | value
#   RUNS          pivot:
#       Row "vars", then variable rows; row "outputs", then output rows; optional "loss" row.
#       Experiments are columns (B..).
#
# Suggestions are printed to console.
# Loss values are COMPUTED and PRINTED to console (not written to Excel).

from pathlib import Path
import pickle
import sys
import pandas as pd
import ProcessOptimizer as po
from ProcessOptimizer.space import Real, Integer, Categorical

# -------------------- CONFIG --------------------
EXCEL_PATH = Path("Excel/opica_tp4_optimizer.xlsx")  # adjust path/name
STATE_PKL  = EXCEL_PATH.with_suffix(".pkl")
# ------------------------------------------------

# ----------------- helpers ----------------------
def _norm_cols_inplace(df):
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _as_bool(x, default=True):
    s = str(x).strip().lower()
    if s in ("true","1","yes","y"): return True
    if s in ("false","0","no","n"): return False
    return default

def _num_from_cell(val):
    if pd.isna(val) or val == "": return None
    s = str(val).strip().replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def _int_from_cell(val):
    f = _num_from_cell(val)
    if f is None: return None
    try:
        return int(round(f))
    except Exception:
        return None

# --------------- SETUP readers ------------------
def read_setup_const(path: Path):
    df = pd.read_excel(path, sheet_name="SETUP-CONST").fillna("")
    _norm_cols_inplace(df)
    constants = {}
    if {"name","value"}.issubset(df.columns):
        for _, r in df.iterrows():
            nm = str(r.get("name","")).strip()
            if nm:
                constants[nm] = r.get("value","")
    return constants

def read_setup_var(path: Path):
    df = pd.read_excel(path, sheet_name="SETUP-VAR").fillna("")
    _norm_cols_inplace(df)
    for need in ["name","type","low","high","choices","unit","active"]:
        if need not in df.columns:
            df[need] = ""
    if "section" in df.columns:
        df = df[df["section"].astype(str).str.strip().str.lower()=="var"]
    df["name"]  = df["name"].astype(str).str.strip()
    df["type"]  = df["type"].astype(str).str.strip().str.capitalize()
    df["unit"]  = df["unit"].astype(str).str.strip()
    df["active"]= df["active"].apply(_as_bool)
    df = df[df["active"]==True]

    variables = []
    for _, r in df.iterrows():
        name = r["name"]; vtype = r["type"] or "Real"; unit = r["unit"]
        if vtype == "Categorical":
            choices = [c.strip() for c in str(r.get("choices","")).split(",") if c.strip()]
            if not choices:
                raise ValueError(f"SETUP-VAR: categorical var '{name}' needs 'choices'.")
            variables.append({"type":"Categorical","name":name,"choices":choices,"unit":unit})
        elif vtype == "Integer":
            lo = _int_from_cell(r.get("low","")); hi = _int_from_cell(r.get("high",""))
            if lo is None or hi is None:
                raise ValueError(f"SETUP-VAR: integer var '{name}' needs numeric low/high.")
            variables.append({"type":"Integer","name":name,"low":lo,"high":hi,"unit":unit})
        else:  # Real
            lo = _num_from_cell(r.get("low","")); hi = _num_from_cell(r.get("high",""))
            if lo is None or hi is None:
                raise ValueError(f"SETUP-VAR: real var '{name}' needs numeric low/high.")
            variables.append({"type":"Real","name":name,"low":lo,"high":hi,"unit":unit})
    return variables

def read_setup_out(path: Path):
    df = pd.read_excel(path, sheet_name="SETUP-OUT").fillna("")
    _norm_cols_inplace(df)
    for need in ["name","kind","target","weight"]:
        if need not in df.columns: df[need] = ""
    if "section" in df.columns:
        df = df[df["section"].astype(str).str.strip().str.lower()=="out"]
    df["name"] = df["name"].astype(str).str.strip()
    df["kind"] = df["kind"].astype(str).str.strip().str.lower()

    outputs = []
    for _, r in df.iterrows():
        tgt = _num_from_cell(r.get("target","")); tgt = 100.0 if tgt is None else tgt
        w   = _num_from_cell(r.get("weight","")); w   = 1.0   if w   is None else w
        kind = r.get("kind","min") or "min"
        outputs.append({"name": r["name"], "kind": kind, "target": float(tgt), "weight": float(w)})
    return outputs

def read_setup_opt(path: Path):
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

    opts = {"acq_func":"EI","n_initial_points":6,"batch_size":1,"diversity_eps":0.05}
    for _, r in df.iterrows():
        n = str(r.get("name","")).strip().lower()
        v = r.get("value","")
        if n == "acq_func":
            opts["acq_func"] = str(v)
        elif n == "n_initial_points":
            iv = _int_from_cell(v);  opts["n_initial_points"] = iv if iv is not None else opts["n_initial_points"]
        elif n == "batch_size":
            iv = _int_from_cell(v);  opts["batch_size"]       = iv if iv is not None else opts["batch_size"]
        elif n == "diversity_eps":
            fv = _num_from_cell(v);  opts["diversity_eps"]    = fv if fv is not None else opts["diversity_eps"]
    return opts

# --------------- RUNS (pivot) reader ---------------
def read_runs_pivot(path: Path):
    df = pd.read_excel(path, sheet_name="RUNS", header=None)
    if df.empty: return [], [], []

    # find 'vars' header row
    vars_header_idx = df.index[df.iloc[:,0].astype(str).str.strip().str.lower()=="vars"]
    if len(vars_header_idx)==0: return [], [], []
    vh = vars_header_idx[0]

    # variable rows
    r = vh + 1
    var_rows = []
    while r < len(df):
        first = str(df.iloc[r,0]).strip()
        if first == "" or first.lower() == "outputs": break
        var_rows.append(r); r += 1

    # outputs rows
    outs_header_idx = df.index[df.iloc[:,0].astype(str).str.strip().str.lower()=="outputs"]
    out_rows = []
    if len(outs_header_idx):
        oh = outs_header_idx[0]
        r2 = oh + 1
        while r2 < len(df):
            first = str(df.iloc[r2,0]).strip()
            if first == "": break
            out_rows.append(r2); r2 += 1

    # experiment columns (headers on vh row)
    exp_cols = []
    exp_names = []
    for j in range(1, df.shape[1]):
        hdr = str(df.iloc[vh, j]).strip()
        if hdr != "":
            exp_cols.append(j)
            exp_names.append(hdr)

    # build experiments
    experiments = []
    for j in exp_cols:
        e = {}
        good = True
        for rr in var_rows:
            key = str(df.iloc[rr,0]).strip()
            val = df.iloc[rr, j]
            if pd.isna(val) or str(val).strip() == "":
                good = False; break
            e[key] = val
        if not good: 
            experiments.append(None)  # keep alignment with exp_cols
            continue
        for rr in out_rows:
            key = str(df.iloc[rr,0]).strip()
            if key.lower() == "loss":  # ignore recorded loss
                continue
            val = df.iloc[rr, j]
            if pd.isna(val) or str(val).strip() == "": continue
            e[key] = val
        experiments.append(e)

    return experiments, exp_cols, exp_names

# --------------- loss & space -------------------
def compute_loss(outputs_cfg, row_dict):
    total = 0.0
    for cfg in outputs_cfg:
        n = cfg["name"]; kind = cfg.get("kind","min")
        w = float(cfg.get("weight",1.0))
        tgt = float(cfg.get("target",100.0))
        if n not in row_dict: return None
        val = _num_from_cell(row_dict[n])
        if val is None: return None
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

def build_space(variables):
    dims = []
    for v in variables:
        if v["type"] == "Real":
            dims.append(Real(v["low"], v["high"], name=v["name"]))
        elif v["type"] == "Integer":
            dims.append(Integer(v["low"], v["high"], name=v["name"]))
        elif v["type"] == "Categorical":
            dims.append(Categorical(v["choices"], name=v["name"]))
    return po.Space(dims)

def vec_to_named(x_vec, variables):
    out = {}
    for i, v in enumerate(variables):
        t = v["type"]; n = v["name"]
        if t == "Categorical": out[n] = x_vec[i]
        elif t == "Real":      out[n] = float(x_vec[i])
        else:                  out[n] = int(x_vec[i])
    return out

def normalized_distance(a, b, variables):
    num = 0.0; cnt = 0
    for v in variables:
        t = v["type"]; n = v["name"]
        va, vb = a[n], b[n]
        if t == "Categorical":
            num += 0.0 if va == vb else 1.0; cnt += 1
        else:
            span = float(v["high"] - v["low"])
            if span <= 0: continue
            num += abs((float(va) - float(vb)) / span); cnt += 1
    return num / max(cnt, 1)

def ask_batch(opt, k, variables, diversity_eps):
    if k <= 1:
        return [vec_to_named(opt.ask(), variables)]
    batch, tries = [], 0
    while len(batch) < k and tries < 500:
        xn = vec_to_named(opt.ask(), variables)
        if batch:
            dmin = min(normalized_distance(xn, b, variables) for b in batch)
            if dmin < diversity_eps:
                tries += 1; continue
        batch.append(xn)
    return batch or [vec_to_named(opt.ask(), variables)]

# --------------- warm start --------------------
def warmstart_from_runs(opt, experiments, variables, outputs_cfg):
    if not experiments: return
    fed = 0
    for e in experiments:
        if not e: 
            continue
        x, ok = [], True
        for v in variables:
            name, t = v["name"], v["type"]
            if name not in e: ok = False; break
            if t == "Categorical":
                x.append(str(e[name]))
            elif t == "Real":
                num = _num_from_cell(e[name]);   
                if num is None: ok = False; break
                x.append(num)
            else:
                iv  = _int_from_cell(e[name]);   
                if iv  is None: ok = False; break
                x.append(iv)
        if not ok: continue
        y = compute_loss(outputs_cfg, e)
        if y is None: continue
        opt.tell(x, y); fed += 1
    print(f"[warmstart] fed {fed} completed run(s) into the optimizer.")

# --------------- compute+print loss report ---------------
def print_loss_report(path: Path, outputs_cfg):
    """
    Reads RUNS and prints a loss line per experiment column:
      - experiment header
      - loss value (if all outputs present), or a short reason why not computed
    """
    df = pd.read_excel(path, sheet_name="RUNS", header=None)
    if df.empty:
        print("\n[loss] RUNS sheet is empty.")
        return

    # locate headers
    vars_header_idx = df.index[df.iloc[:,0].astype(str).str.strip().str.lower()=="vars"]
    if not len(vars_header_idx):
        print("\n[loss] No 'vars' header found in RUNS.")
        return
    vh = vars_header_idx[0]

    # outputs rows
    outs_header_idx = df.index[df.iloc[:,0].astype(str).str.strip().str.lower()=="outputs"]
    if not len(outs_header_idx):
        print("\n[loss] No 'outputs' header found in RUNS.")
        return
    oh = outs_header_idx[0]
    r2 = oh + 1
    out_rows = []
    while r2 < len(df):
        first = str(df.iloc[r2,0]).strip()
        if first == "": break
        out_rows.append(r2); r2 += 1

    # experiment cols + names
    exp_cols = []
    exp_names = []
    for j in range(1, df.shape[1]):
        hdr = str(df.iloc[vh, j]).strip()
        if hdr != "":
            exp_cols.append(j)
            exp_names.append(hdr)

    print("\n=== Loss report (copy manually into RUNS → 'loss' row) ===")
    any_row = False
    for j, name in zip(exp_cols, exp_names):
        row_dict, have_all = {}, True
        missing = []
        for rr in out_rows:
            key = str(df.iloc[rr,0]).strip()
            if key.lower() == "loss":
                continue
            val = df.iloc[rr, j]
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

# ------------------ main -----------------------
def main():
    if not EXCEL_PATH.exists():
        sys.exit(f"Excel not found: {EXCEL_PATH}")

    constants = read_setup_const(EXCEL_PATH)
    variables = read_setup_var(EXCEL_PATH)
    outputs   = read_setup_out(EXCEL_PATH)
    opts      = read_setup_opt(EXCEL_PATH)

    space = build_space(variables)

    if STATE_PKL.exists():
        with open(STATE_PKL, "rb") as f:
            opt = pickle.load(f)
    else:
        opt = po.Optimizer(space,
                           base_estimator="GP",
                           n_initial_points=opts["n_initial_points"],
                           acq_func=opts["acq_func"])

    experiments, _, _ = read_runs_pivot(EXCEL_PATH)
    warmstart_from_runs(opt, experiments, variables, outputs)

    suggestions = ask_batch(opt, opts["batch_size"], variables, opts["diversity_eps"])

    print("\n=== Suggested conditions ===")
    for i, s in enumerate(suggestions, 1):
        print(f"\n-- Suggestion {i}/{len(suggestions)} --")
        for v in variables:
            name, unit = v["name"], v.get("unit","")
            print(f"{name:>24}: {s[name]} {unit}".rstrip())

    with open(STATE_PKL, "wb") as f:
        pickle.dump(opt, f)

    # Print per-experiment loss report (no Excel write)
    print_loss_report(EXCEL_PATH, outputs)

    print(f"\nPrinted {len(suggestions)} suggestion(s).")
    print("Paste one suggestion into RUNS (new experiment column under 'vars'), run it, fill outputs, then re-run.")
    print("Copy the 'loss' values from the Loss report into the RUNS sheet manually.")
    
if __name__ == "__main__":
    main()
