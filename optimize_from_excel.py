# optimize_from_excel.py
# Excel-driven, human-in-the-loop optimization using ProcessOptimizer
# - TEMPLATE sheet: ignored
# - SETUP sheet: sections CONST / VAR / OUT / OPT (any order). Case & whitespace tolerant.
# - RUNS sheet: pivot layout:
#     Row with "vars", then variable rows; a row "outputs", then output rows; columns = experiments.
# - Suggestions printed to console only.
# - Missing 'loss' computed (if all outputs present) and written back if the file isn't locked.

from pathlib import Path
import pickle
import pandas as pd
import ProcessOptimizer as po
from ProcessOptimizer.space import Real, Integer, Categorical

# -------------------- CONFIG --------------------
EXCEL_PATH = Path("Excel/opica_tp1_optimizer.xlsx")  # adjust if needed
STATE_PKL  = EXCEL_PATH.with_suffix(".pkl")          # persisted optimizer state per workbook
# ------------------------------------------------

# ---------- small helpers ----------
def _norm_cols(df):
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _as_bool(x, default=True):
    s = str(x).strip().lower()
    if s in ("true", "1", "yes", "y"): return True
    if s in ("false", "0", "no", "n"): return False
    return default

def _num_from_cell(val):
    """Coerce Excel cell to float, allowing comma decimals and trimming spaces."""
    if pd.isna(val) or val == "":
        return None
    s = str(val).strip().replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def _int_from_cell(val):
    f = _num_from_cell(val)
    if f is None:
        return None
    try:
        return int(round(f))
    except Exception:
        return None

# ---------- SETUP parsing (robust) ----------
def read_setup(excel_path: Path):
    df = pd.read_excel(excel_path, sheet_name="SETUP").fillna("")
    _norm_cols(df)

    # constants
    cdf = df[df.get("section", "").astype(str).str.strip().str.lower() == "const"].copy()
    _norm_cols(cdf)
    constants = {}
    for _, r in cdf.iterrows():
        nm = str(r.get("name", "")).strip()
        if nm:
            constants[nm] = r.get("value", "")

    # variables
    vdf = df[df.get("section", "").astype(str).str.strip().str.lower() == "var"].copy()
    _norm_cols(vdf)
    for need in ["name", "type", "low", "high", "choices", "unit", "active"]:
        if need not in vdf.columns:
            vdf[need] = ""
    vdf["name"] = vdf["name"].astype(str).str.strip()
    vdf["type"] = vdf["type"].astype(str).str.strip().str.capitalize()
    vdf["unit"] = vdf["unit"].astype(str).str.strip()
    vdf["active"] = vdf["active"].apply(_as_bool)

    vdf = vdf[vdf["active"] == True]

    variables = []
    for _, r in vdf.iterrows():
        name = r["name"]
        vtype = r["type"] or "Real"
        unit  = r["unit"]
        if vtype == "Categorical":
            choices = [c.strip() for c in str(r.get("choices", "")).split(",") if c.strip()]
            if not choices:
                raise ValueError(f"SETUP: categorical var '{name}' needs 'choices'.")
            variables.append({"type": "Categorical", "name": name, "choices": choices, "unit": unit})
        elif vtype == "Integer":
            lo = _int_from_cell(r.get("low", ""))
            hi = _int_from_cell(r.get("high", ""))
            if lo is None or hi is None:
                raise ValueError(f"SETUP: integer var '{name}' needs numeric low/high.")
            variables.append({"type": "Integer", "name": name, "low": lo, "high": hi, "unit": unit})
        else:  # Real (default)
            lo = _num_from_cell(r.get("low", ""))
            hi = _num_from_cell(r.get("high", ""))
            if lo is None or hi is None:
                raise ValueError(f"SETUP: real var '{name}' needs numeric low/high.")
            variables.append({"type": "Real", "name": name, "low": lo, "high": hi, "unit": unit})

    # outputs
    odf = df[df.get("section", "").astype(str).str.strip().str.lower() == "out"].copy()
    _norm_cols(odf)
    for need in ["name", "kind", "target", "weight"]:
        if need not in odf.columns:
            odf[need] = ""
    odf["name"] = odf["name"].astype(str).str.strip()
    odf["kind"] = odf["kind"].astype(str).str.strip().str.lower().replace({"": "min"})

    outputs = []
    for _, r in odf.iterrows():
        tgt = _num_from_cell(r.get("target", ""))
        if tgt is None: tgt = 100.0
        w   = _num_from_cell(r.get("weight", ""))
        if w is None: w = 1.0
        outputs.append({
            "name": r["name"],
            "kind": r.get("kind", "min"),
            "target": float(tgt),
            "weight": float(w),
        })

    # options
    opts = {"acq_func": "EI", "n_initial_points": 6, "batch_size": 1, "diversity_eps": 0.05}
    optdf = df[df.get("section", "").astype(str).str.strip().str.lower() == "opt"].copy()
    _norm_cols(optdf)
    for _, r in optdf.iterrows():
        n = str(r.get("name", "")).strip().lower()
        v = r.get("value", "")
        if n == "acq_func":
            opts["acq_func"] = str(v)
        elif n == "n_initial_points":
            iv = _int_from_cell(v)
            if iv is not None: opts["n_initial_points"] = iv
        elif n == "batch_size":
            iv = _int_from_cell(v)
            if iv is not None: opts["batch_size"] = iv
        elif n == "diversity_eps":
            fv = _num_from_cell(v)
            if fv is not None: opts["diversity_eps"] = fv

    return constants, variables, outputs, opts

# ---------- RUNS (pivot) parsing ----------
def read_runs_pivot(excel_path: Path):
    df = pd.read_excel(excel_path, sheet_name="RUNS", header=None)
    if df.empty:
        return [], []

    # find 'vars' row
    vars_row_idx = df.index[df.iloc[:, 0].astype(str).str.strip().str.lower() == "vars"]
    if len(vars_row_idx) == 0:
        return [], []
    vars_row = vars_row_idx[0]

    # variable rows
    r = vars_row + 1
    var_rows = []
    while r < len(df):
        first = str(df.iloc[r, 0]).strip()
        if first == "" or first.lower() == "outputs":
            break
        var_rows.append(r)
        r += 1

    # outputs rows
    outs_header_idx = df.index[df.iloc[:, 0].astype(str).str.strip().str.lower() == "outputs"]
    out_rows = []
    if len(outs_header_idx):
        outs_row = outs_header_idx[0]
        r2 = outs_row + 1
        while r2 < len(df):
            first = str(df.iloc[r2, 0]).strip()
            if first == "":
                break
            out_rows.append(r2)
            r2 += 1

    # experiment columns (headers at vars row)
    exp_cols = []
    for j in range(1, df.shape[1]):
        hdr = str(df.iloc[vars_row, j]).strip()
        if hdr != "":
            exp_cols.append(j)

    experiments = []
    for j in exp_cols:
        e = {}
        # variables
        bad = False
        for rr in var_rows:
            key = str(df.iloc[rr, 0]).strip()
            val = df.iloc[rr, j]
            if pd.isna(val) or str(val).strip() == "":
                bad = True; break
            e[key] = val
        if bad:
            continue
        # outputs (optional)
        for rr in out_rows:
            key = str(df.iloc[rr, 0]).strip()
            val = df.iloc[rr, j]
            if pd.isna(val) or str(val).strip() == "":
                continue
            e[key] = val
        experiments.append(e)

    return experiments, exp_cols

# ---------- Loss & space helpers ----------
def compute_loss(outputs_cfg, row_dict):
    total = 0.0
    for cfg in outputs_cfg:
        n = cfg["name"]; kind = cfg.get("kind", "min")
        w = float(cfg.get("weight", 1.0))
        tgt = float(cfg.get("target", 100.0))
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
        if t == "Categorical":
            out[n] = x_vec[i]
        elif t == "Real":
            out[n] = float(x_vec[i])
        else:
            out[n] = int(x_vec[i])
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

# ---------- Warm start ----------
def warmstart_from_runs(opt, experiments, variables, outputs_cfg):
    if not experiments:
        return
    for e in experiments:
        x, ok = [], True
        for v in variables:
            name, t = v["name"], v["type"]
            if name not in e: ok = False; break
            if t == "Categorical":
                x.append(str(e[name]))
            elif t == "Real":
                num = _num_from_cell(e[name])
                if num is None: ok = False; break
                x.append(num)
            else:
                iv = _int_from_cell(e[name])
                if iv is None: ok = False; break
                x.append(iv)
        if not ok: continue
        y = compute_loss(outputs_cfg, e)
        if y is None: continue
        opt.tell(x, y)

# ---------- Write/Update 'loss' back into RUNS (pivot) ----------
def write_loss_into_runs(excel_path: Path, outputs_cfg):
    """
    Compute 'loss' for experiment columns that already have all outputs present,
    and write/overwrite a 'loss' row under the 'outputs' block.
    """
    try:
        df = pd.read_excel(excel_path, sheet_name="RUNS", header=None)
    except Exception:
        return
    if df.empty: return

    # locate headers/blocks
    vars_header_idx = df.index[df.iloc[:, 0].astype(str).str.strip().str.lower() == "vars"]
    if not len(vars_header_idx): return
    vh = vars_header_idx[0]

    r = vh + 1
    var_rows = []
    while r < len(df):
        first = str(df.iloc[r, 0]).strip()
        if first == "" or first.lower() == "outputs":
            break
        var_rows.append(r)
        r += 1

    outs_header_idx = df.index[df.iloc[:, 0].astype(str).str.strip().str.lower() == "outputs"]
    if not len(outs_header_idx): return
    oh = outs_header_idx[0]
    r2 = oh + 1
    out_rows = []
    while r2 < len(df):
        first = str(df.iloc[r2, 0]).strip()
        if first == "": break
        out_rows.append(r2)
        r2 += 1

    # find/create 'loss' row
    loss_row_idx = None
    for rr in out_rows:
        if str(df.iloc[rr, 0]).strip().lower() == "loss":
            loss_row_idx = rr; break
    if loss_row_idx is None:
        loss_row_idx = r2
        needed = loss_row_idx - (len(df) - 1)
        if needed > 0:
            for _ in range(needed):
                df.loc[len(df)] = [None] * df.shape[1]
        df.iloc[loss_row_idx, 0] = "loss"

    changed = False
    # iterate experiment columns
    for j in range(1, df.shape[1]):
        hdr = str(df.iloc[vh, j]).strip()
        if hdr == "": continue
        # collect outputs for this column
        row_dict, have_all = {}, True
        for rr in out_rows:
            key = str(df.iloc[rr, 0]).strip()
            val = df.iloc[rr, j]
            if pd.isna(val) or str(val).strip() == "":
                have_all = False; break
            row_dict[key] = val
        if not have_all: continue
        y = compute_loss(outputs_cfg, row_dict)
        if y is None: continue
        df.iloc[loss_row_idx, j] = y
        changed = True

    if changed:
        try:
            with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as wr:
                df.to_excel(wr, sheet_name="RUNS", header=False, index=False)
        except PermissionError:
            print("[warn] Excel is open: couldn't write computed loss back to RUNS. Close Excel and re-run to save.")

# ---------- main ----------
def main():
    if not EXCEL_PATH.exists():
        raise SystemExit(f"Excel not found: {EXCEL_PATH}")

    constants, variables, outputs, opts = read_setup(EXCEL_PATH)
    space = build_space(variables)

    # load or init optimizer
    if STATE_PKL.exists():
        with open(STATE_PKL, "rb") as f:
            opt = pickle.load(f)
    else:
        opt = po.Optimizer(space,
                           base_estimator="GP",
                           n_initial_points=opts["n_initial_points"],
                           acq_func=opts["acq_func"])

    # read runs and warm-start (only columns with full outputs)
    experiments, _exp_cols = read_runs_pivot(EXCEL_PATH)
    warmstart_from_runs(opt, experiments, variables, outputs)

    # propose next batch
    suggestions = ask_batch(opt, opts["batch_size"], variables, opts["diversity_eps"])

    # print suggestions to console
    print("\n=== Suggested conditions ===")
    for i, s in enumerate(suggestions, 1):
        print(f"\n-- Suggestion {i}/{len(suggestions)} --")
        for v in variables:
            name, unit = v["name"], v.get("unit", "")
            print(f"{name:>24}: {s[name]} {unit}".rstrip())

    # persist optimizer state
    with open(STATE_PKL, "wb") as f:
        pickle.dump(opt, f)

    # compute/write loss back for columns with all outputs (if Excel isn't open)
    write_loss_into_runs(EXCEL_PATH, outputs)

    print(f"\nPrinted {len(suggestions)} suggestion(s).")
    print("Paste one suggestion into RUNS (new experiment column under 'vars'),")
    print("run the experiment, fill outputs, then re-run this script.")

if __name__ == "__main__":
    main()
