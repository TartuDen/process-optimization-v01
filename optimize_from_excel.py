# optimize_from_excel.py
# Excel-driven, human-in-the-loop optimization using ProcessOptimizer
# - TEMPLATE sheet: ignored
# - SETUP sheet: sections CONST / VAR / OUT / OPT (mixed blocks), tolerant to case/whitespace
# - RUNS sheet: pivot layout:
#       Row with "vars" in A1, experiment headers across row (B1..)
#       Variable rows below ("NH2OH_eq", "Et3N_eq", ...), values per experiment column
#       A row with "outputs"
#       Output rows below (yield_pct, purity_pct, imp_..., ...), values per experiment column
# - Suggestions are PRINTED TO CONSOLE ONLY (no Excel write).
# - Missing 'loss' is computed when all outputs for a column are present and written back if file is not open.

from pathlib import Path
import pickle
import pandas as pd
import ProcessOptimizer as po
from ProcessOptimizer.space import Real, Integer, Categorical

# -------------------- CONFIG --------------------
EXCEL_PATH = Path("opica_tp1_optimizer.xlsx")  # change path/name as needed
STATE_PKL  = EXCEL_PATH.with_suffix(".pkl")          # persisted optimizer state per workbook
# ------------------------------------------------

# ---------- SETUP parsing (robust) ----------
def read_setup(excel_path: Path):
    df = pd.read_excel(excel_path, sheet_name="SETUP").fillna("")
    # normalize headers
    df.columns = [str(c).strip().lower() for c in df.columns]

    def col_or(name, default_series=None):
        return df[name] if name in df.columns else (default_series if default_series is not None else "")

    # ---- constants
    cdf = df[col_or("section").astype(str).str.strip().str.lower() == "const"].copy()
    cdf.columns = [str(c).strip().lower() for c in cdf.columns]
    constants = {}
    if not cdf.empty:
        for _, r in cdf.iterrows():
            nm = str(r.get("name", "")).strip()
            if nm:
                constants[nm] = r.get("value", "")

    # ---- variables
    vdf = df[col_or("section").astype(str).str.strip().str.lower() == "var"].copy()
    vdf.columns = [str(c).strip().lower() for c in vdf.columns]
    # ensure columns exist
    for need in ["name", "type", "low", "high", "choices", "unit", "active"]:
        if need not in vdf.columns:
            vdf[need] = ""
    vdf["name"] = vdf["name"].astype(str).str.strip()
    vdf["type"] = vdf["type"].astype(str).str.strip().str.capitalize()
    vdf["unit"] = vdf["unit"].astype(str).str.strip()

    def to_bool(x):
        s = str(x).strip().lower()
        return s in ["true", "1", "yes", "y", ""]  # default TRUE if blank
    vdf["active"] = vdf["active"].apply(to_bool)
    vdf = vdf[vdf["active"] == True]

    variables = []
    for _, r in vdf.iterrows():
        name = r["name"]
        vtype = r["type"] or "Real"
        unit = r["unit"]
        if vtype == "Categorical":
            choices = [c.strip() for c in str(r.get("choices", "")).split(",") if c.strip() != ""]
            if not choices:
                raise ValueError(f"Categorical var '{name}' needs 'choices' in SETUP.")
            variables.append({"type": "Categorical", "name": name, "choices": choices, "unit": unit})
        elif vtype == "Integer":
            try:
                lo = int(float(r.get("low", 0)))
                hi = int(float(r.get("high", 0)))
            except Exception:
                raise ValueError(f"Integer var '{name}' needs numeric low/high.")
            variables.append({"type": "Integer", "name": name, "low": lo, "high": hi, "unit": unit})
        else:  # Real (default)
            try:
                lo = float(r.get("low", 0.0))
                hi = float(r.get("high", 0.0))
            except Exception:
                raise ValueError(f"Real var '{name}' needs numeric low/high.")
            variables.append({"type": "Real", "name": name, "low": lo, "high": hi, "unit": unit})

    # ---- outputs
    odf = df[col_or("section").astype(str).str.strip().str.lower() == "out"].copy()
    odf.columns = [str(c).strip().lower() for c in odf.columns]
    for need in ["name", "kind", "target", "weight"]:
        if need not in odf.columns:
            odf[need] = ""
    odf["name"] = odf["name"].astype(str).str.strip()
    odf["kind"] = odf["kind"].astype(str).str.strip().str.lower().replace({"": "min"})
    outputs = []
    for _, r in odf.iterrows():
        tgt = r.get("target", "")
        outputs.append({
            "name": r["name"],
            "kind": r.get("kind", "min"),
            "target": float(tgt) if str(tgt).strip() != "" else 100.0,
            "weight": float(r.get("weight", 1.0)),
        })

    # ---- options
    opts = {"acq_func": "EI", "n_initial_points": 6, "batch_size": 1, "diversity_eps": 0.05}
    optdf = df[col_or("section").astype(str).str.strip().str.lower() == "opt"].copy()
    optdf.columns = [str(c).strip().lower() for c in optdf.columns]
    for _, r in optdf.iterrows():
        n = str(r.get("name", "")).strip().lower()
        v = r.get("value", "")
        if n == "acq_func":
            opts["acq_func"] = str(v)
        elif n == "n_initial_points":
            opts["n_initial_points"] = int(float(v))
        elif n == "batch_size":
            opts["batch_size"] = int(float(v))
        elif n == "diversity_eps":
            opts["diversity_eps"] = float(v)

    return constants, variables, outputs, opts

# ---------- RUNS (pivot) parsing ----------
def read_runs_pivot(excel_path: Path):
    """
    Returns:
      experiments: list of dicts (variables + any available outputs) per experiment col
      exp_col_names: user-visible column headers on the 'vars' header row
    """
    df = pd.read_excel(excel_path, sheet_name="RUNS", header=None)
    if df.empty:
        return [], []

    # find 'vars' row
    vars_row_idx = df.index[df.iloc[:, 0].astype(str).str.strip().str.lower() == "vars"]
    if len(vars_row_idx) == 0:
        return [], []
    vars_row = vars_row_idx[0]

    # collect variable rows
    r = vars_row + 1
    var_rows = []
    while r < len(df):
        first = str(df.iloc[r, 0]).strip()
        if first == "" or first.lower() == "outputs":
            break
        var_rows.append(r)
        r += 1

    # find 'outputs' row
    outs_header_idx = df.index[df.iloc[:, 0].astype(str).str.strip().str.lower() == "outputs"]
    out_rows = []
    outs_row = None
    if len(outs_header_idx):
        outs_row = outs_header_idx[0]
        r2 = outs_row + 1
        while r2 < len(df):
            first = str(df.iloc[r2, 0]).strip()
            if first == "":
                break
            out_rows.append(r2)
            r2 += 1

    # experiment columns (headers are on the vars header row)
    exp_cols = []
    for j in range(1, df.shape[1]):
        hdr = str(df.iloc[vars_row, j]).strip()
        if hdr != "":
            exp_cols.append((j, hdr))

    experiments = []
    for (j, _hdr) in exp_cols:
        e = {}
        # variables
        bad = False
        for rr in var_rows:
            key = str(df.iloc[rr, 0]).strip()
            val = df.iloc[rr, j]
            if pd.isna(val) or str(val).strip() == "":
                bad = True
                break
            e[key] = val
        if bad:
            continue
        # outputs (optional; OK if missing)
        for rr in out_rows:
            key = str(df.iloc[rr, 0]).strip()
            val = df.iloc[rr, j]
            if pd.isna(val) or str(val).strip() == "":
                continue
            e[key] = val
        experiments.append(e)

    return experiments, [c for _, c in exp_cols]

# ---------- Loss & space helpers ----------
def compute_loss(outputs_cfg, row_dict):
    total = 0.0
    for cfg in outputs_cfg:
        n = cfg["name"]
        kind = cfg.get("kind", "min")
        w = float(cfg.get("weight", 1.0))
        tgt = float(cfg.get("target", 100.0))
        if n not in row_dict:
            return None
        try:
            val = float(row_dict[n])
        except Exception:
            return None
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
            num += 0.0 if va == vb else 1.0
            cnt += 1
        else:
            span = float(v["high"] - v["low"])
            if span <= 0: continue
            num += abs((float(va) - float(vb)) / span)
            cnt += 1
    return num / max(cnt, 1)

def ask_batch(opt, k, variables, diversity_eps):
    if k <= 1:
        return [vec_to_named(opt.ask(), variables)]
    batch = []
    tries = 0
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
        # build X vector in variable order; skip if any var missing
        x = []
        good = True
        for v in variables:
            name, t = v["name"], v["type"]
            if name not in e:
                good = False; break
            val = e[name]
            if t == "Categorical":
                x.append(str(val))
            elif t == "Real":
                x.append(float(val))
            else:
                x.append(int(float(val)))
        if not good:
            continue
        y = compute_loss(outputs_cfg, e)
        if y is None:
            continue
        opt.tell(x, y)

# ---------- Write/Update 'loss' row back into RUNS (pivot) ----------
def write_loss_into_runs(excel_path: Path, outputs_cfg, variables):
    """
    Re-open RUNS and write a 'loss' row under outputs for columns where all outputs are present.
    Safe if Excel is open: if file is locked, we just warn and skip writing.
    """
    try:
        df = pd.read_excel(excel_path, sheet_name="RUNS", header=None)
    except Exception:
        return

    if df.empty:
        return

    # locate vars header row
    vars_header_idx = df.index[df.iloc[:, 0].astype(str).str.strip().str.lower() == "vars"]
    if len(vars_header_idx) == 0:
        return
    vh = vars_header_idx[0]

    # gather variable and output rows
    r = vh + 1
    var_rows = []
    while r < len(df):
        first = str(df.iloc[r, 0]).strip()
        if first == "" or first.lower() == "outputs":
            break
        var_rows.append(r)
        r += 1

    outs_header_idx = df.index[df.iloc[:, 0].astype(str).str.strip().str.lower() == "outputs"]
    if len(outs_header_idx) == 0:
        return
    oh = outs_header_idx[0]
    r2 = oh + 1
    out_rows = []
    while r2 < len(df):
        first = str(df.iloc[r2, 0]).strip()
        if first == "":
            break
        out_rows.append(r2)
        r2 += 1

    # find or create 'loss' row
    loss_row_idx = None
    for rr in out_rows:
        if str(df.iloc[rr, 0]).strip().lower() == "loss":
            loss_row_idx = rr; break
    if loss_row_idx is None:
        loss_row_idx = r2  # append at first blank after outputs
        needed = loss_row_idx - (len(df) - 1)
        if needed > 0:
            for _ in range(needed):
                df.loc[len(df)] = [None] * df.shape[1]
        df.iloc[loss_row_idx, 0] = "loss"

    # iterate experiment columns and compute loss where possible
    changed = False
    for j in range(1, df.shape[1]):
        # skip empty experiment headers
        hdr = str(df.iloc[vh, j]).strip()
        if hdr == "":
            continue

        row_dict = {}
        # collect outputs
        have_all = True
        for rr in out_rows:
            key = str(df.iloc[rr, 0]).strip()
            val = df.iloc[rr, j]
            if pd.isna(val) or str(val).strip() == "":
                have_all = False
                break
            row_dict[key] = val
        if not have_all:
            continue

        y = compute_loss(outputs_cfg, row_dict)
        if y is None:
            continue
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

    # load existing optimizer or create new
    if STATE_PKL.exists():
        with open(STATE_PKL, "rb") as f:
            opt = pickle.load(f)
    else:
        opt = po.Optimizer(space,
                          base_estimator="GP",
                          n_initial_points=opts["n_initial_points"],
                          acq_func=opts["acq_func"])

    # read runs (pivot), warmstart if outputs are present
    experiments, exp_cols = read_runs_pivot(EXCEL_PATH)
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

    # attempt to compute/write loss back to RUNS for fully-observed experiments
    write_loss_into_runs(EXCEL_PATH, outputs, variables)

    print(f"\nPrinted {len(suggestions)} suggestion(s).")
    print("Copy one suggestion into a new experiment column under RUNS → vars, perform the run,")
    print("then fill outputs. You can keep Excel open to read; close it if you want loss written back.")

if __name__ == "__main__":
    main()
