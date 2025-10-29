# optimize_from_excel_pivot.py
# Works with your 3-sheet workbook:
# - TEMPLATE (ignored)
# - SETUP: mixed sections (CONST / VAR / OUT / OPT) exactly as you showed
# - RUNS: pivoted: "vars" block (rows=variable names) + "outputs" block (rows=output names), columns = experiments

from pathlib import Path
import pickle
import pandas as pd
import ProcessOptimizer as po
from ProcessOptimizer.space import Real, Integer, Categorical

# --------- USER: set workbook path ----------
EXCEL_PATH = Path("Excel/optimize_from_excel.xlsx")  # change to your file path
STATE_PKL  = EXCEL_PATH.with_suffix(".pkl")          # state persisted per workbook

# --------- SETUP parsing ----------
def read_setup(excel_path: Path):
    df = pd.read_excel(excel_path, sheet_name="SETUP").fillna("")

    # constants
    const = {}
    for _, r in df[df["section"].astype(str).str.upper()=="CONST"].iterrows():
        const[str(r["name"]).strip()] = r["value"]

    # variables
    vdf = df[df["section"].astype(str).str.upper()=="VAR"].copy()
    vdf["name"] = vdf["name"].astype(str).str.strip()
    vdf["type"] = vdf["type"].astype(str).str.strip().str.capitalize()
    vdf["unit"] = vdf.get("unit","")
    if "active" in vdf:
        vdf["active"] = vdf["active"].astype(str).str.upper().isin(["TRUE","1","YES","Y"])
    else:
        vdf["active"] = True
    vdf = vdf[vdf["active"]==True]

    variables = []
    for _, r in vdf.iterrows():
        t = r["type"]; name = r["name"]; unit = r.get("unit","")
        if t == "Real":
            variables.append({"type":"Real","name":name,"low":float(r["low"]),"high":float(r["high"]),"unit":unit})
        elif t == "Integer":
            variables.append({"type":"Integer","name":name,"low":int(float(r["low"])),"high":int(float(r["high"])),"unit":unit})
        elif t == "Categorical":
            choices = [c.strip() for c in str(r.get("choices","")).split(",") if c.strip()!=""]
            if not choices:
                raise ValueError(f"Categorical var '{name}' needs choices in SETUP.")
            variables.append({"type":"Categorical","name":name,"choices":choices,"unit":unit})
        else:
            raise ValueError(f"Unknown var type '{t}' for '{name}'")

    # outputs
    odf = df[df["section"].astype(str).str.upper()=="OUT"].copy()
    odf["name"] = odf["name"].astype(str).str.strip()
    odf["kind"] = odf["kind"].astype(str).str.lower().str.strip()
    outputs = []
    for _, r in odf.iterrows():
        outputs.append({
            "name": r["name"],
            "kind": r.get("kind","min"),
            "target": float(r["target"]) if str(r.get("target","")).strip()!="" else 100.0,
            "weight": float(r.get("weight",1.0)),
        })

    # options
    opts = {"acq_func":"EI","n_initial_points":6,"batch_size":1,"diversity_eps":0.05}
    optdf = df[df["section"].astype(str).str.upper()=="OPT"].copy()
    for _, r in optdf.iterrows():
        n = str(r["name"]).strip().lower()
        v = r["value"]
        if n in ["n_initial_points","batch_size"]:
            opts[n] = int(v)
        elif n in ["diversity_eps"]:
            opts[n] = float(v)
        elif n == "acq_func":
            opts[n] = str(v)

    return const, variables, outputs, opts

# --------- RUNS pivot parsing ----------
def read_runs_pivot(excel_path: Path):
    """
    Returns:
      experiments: list of dicts {<var1>:..., <var2>:..., <out1>:..., ...}
      cols: list of experiment column names
      frames: (vars_block_df, outs_block_df) for possible writing back
    """
    df = pd.read_excel(excel_path, sheet_name="RUNS", header=None)
    if df.empty:
        return [], [], (None, None)

    # locate 'vars' row
    vars_row_idx = df.index[df.iloc[:,0].astype(str).str.lower()=="vars"]
    if len(vars_row_idx)==0:
        # empty / not set up yet
        return [], [], (None, None)
    vars_row = vars_row_idx[0]

    # rows for variables start at vars_row+1 until we hit blank or 'outputs'
    r = vars_row + 1
    var_rows = []
    while r < len(df):
        first = str(df.iloc[r,0]).strip()
        if first=="" or first.lower()=="outputs":
            break
        var_rows.append(r)
        r += 1

    # 'outputs' row
    outs_header_row = df.index[df.iloc[:,0].astype(str).str.lower()=="outputs"]
    if len(outs_header_row):
        outs_row = outs_header_row[0]
        # outputs start at outs_row+1 until blank or end
        r2 = outs_row + 1
        out_rows = []
        while r2 < len(df):
            first = str(df.iloc[r2,0]).strip()
            if first=="":
                break
            out_rows.append(r2)
            r2 += 1
    else:
        out_rows = []

    # experiments columns are from column 1 onwards where headers look like 'experiment #1' etc — we just take all non-empty
    # we will keep column labels as-is (from the header lines: on vars row, df.iloc[vars_row, 1:])
    exp_cols = []
    for j in range(1, df.shape[1]):
        header_val = str(df.iloc[vars_row, j]).strip()
        if header_val != "":
            exp_cols.append((j, header_val))

    # build list of experiments dicts
    experiments = []
    for (j, colname) in exp_cols:
        e = {}
        # vars
        for r in var_rows:
            key = str(df.iloc[r,0]).strip()
            val = df.iloc[r, j]
            if pd.isna(val) or str(val).strip()=="":
                # missing variable value — skip this experiment entirely
                e = None; break
            e[key] = val
        if e is None:
            continue
        # outputs (optional; it's fine if blank)
        for r in out_rows:
            key = str(df.iloc[r,0]).strip()
            val = df.iloc[r, j]
            if pd.isna(val) or str(val).strip()=="":
                # leave absent; we compute loss only if all outputs present
                continue
            e[key] = val
        e["_exp_col_index"] = j    # keep excel column index for potential write-back
        experiments.append(e)

    # store the blocks for writing back loss later (we’ll create a 'loss' row under outputs if missing)
    vars_block = df.iloc[[vars_row]+var_rows, :].copy()
    outs_block = None
    if len(out_rows)>0:
        outs_block = df.iloc[[outs_header_row[0]]+out_rows, :].copy()
    else:
        outs_block = pd.DataFrame()

    return experiments, [c for _, c in exp_cols], (vars_block, outs_block)

# --------- Loss and space helpers ----------
def compute_loss(outputs_cfg, row_dict):
    # returns float or None if some outputs are missing
    total = 0.0
    for cfg in outputs_cfg:
        n = cfg["name"]; kind = cfg.get("kind","min")
        w = float(cfg.get("weight",1.0))
        tgt = float(cfg.get("target",100.0))
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
        total += w*pen
    return total

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
    return po.Space(dims)

def vec_to_named(x_vec, variables):
    out = {}
    for i, v in enumerate(variables):
        name, t = v["name"], v["type"]
        if t == "Categorical":
            out[name] = x_vec[i]
        elif t == "Real":
            out[name] = float(x_vec[i])
        else:
            out[name] = int(x_vec[i])
    return out

def normalized_distance(a, b, variables):
    num = 0.0; cnt = 0
    for v in variables:
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

# --------- Warm start from RUNS ----------
def warmstart_from_runs_pivot(opt, experiments, variables, outputs_cfg):
    if not experiments:
        return
    var_names = [v["name"] for v in variables]
    for e in experiments:
        # build X vector in variable order; skip experiment if any var missing
        missing = False
        x = []
        for v in variables:
            name, t = v["name"], v["type"]
            if name not in e:
                missing = True; break
            val = e[name]
            if t == "Categorical":
                x.append(str(val))
            elif t == "Real":
                x.append(float(val))
            else:
                x.append(int(float(val)))
        if missing:
            continue
        # build y (loss) only if outputs present
        y = compute_loss(outputs_cfg, e)
        if y is None:
            continue
        opt.tell(x, y)

# --------- Write SUGGESTIONS sheet ----------
def write_suggestions_sheet(excel_path: Path, suggestions):
    if not suggestions:
        return
    df = pd.DataFrame(suggestions)
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as wr:
        df.to_excel(wr, sheet_name="SUGGESTIONS", index=False)

# --------- Optional: write/append 'loss' row under outputs ----------
def write_loss_into_runs(excel_path: Path, variables, outputs_cfg, experiments, cols):
    """
    Adds/updates a 'loss' row under the 'outputs' block (pivot layout).
    Only writes for experiments that have all outputs present.
    """
    if not experiments:
        return
    # Load raw RUNS again to preserve formatting as much as possible
    df = pd.read_excel(excel_path, sheet_name="RUNS", header=None)
    # find outputs header row
    outs_header_idx = df.index[df.iloc[:,0].astype(str).str.lower()=="outputs"]
    if len(outs_header_idx)==0:
        # create outputs header under current vars block end
        # locate vars header and end
        vars_header_idx = df.index[df.iloc[:,0].astype(str).str.lower()=="vars"]
        if not len(vars_header_idx):
            return
        # find end of var rows
        r = vars_header_idx[0] + 1
        while r < len(df) and str(df.iloc[r,0]).strip() not in ["","outputs"]:
            r += 1
        # insert 'outputs' header + 'loss' (only) — we will not attempt to rebuild full outputs here
        # safer: if no outputs exist, skip writing loss altogether
        return

    # outputs block exists -> see if 'loss' row exists; if not, append at bottom
    ob = outs_header_idx[0]
    # collect out rows indices
    r = ob + 1
    out_rows = []
    while r < len(df) and str(df.iloc[r,0]).strip() != "":
        out_rows.append(r)
        r += 1
    # is there a 'loss' row?
    loss_row_idx = None
    for rr in out_rows:
        if str(df.iloc[rr,0]).strip().lower() == "loss":
            loss_row_idx = rr; break
    if loss_row_idx is None:
        # append one more row
        loss_row_idx = r
        # ensure df has that row
        needed = loss_row_idx - (len(df)-1)
        if needed > 0:
            # pad with empty rows
            for _ in range(needed):
                df.loc[len(df)] = [None]*df.shape[1]
        df.iloc[loss_row_idx, 0] = "loss"

    # map experiment col index by column header at vars header row
    vars_header_idx = df.index[df.iloc[:,0].astype(str).str.lower()=="vars"]
    if not len(vars_header_idx):
        return
    vh = vars_header_idx[0]
    exp_col_map = {}
    for j in range(1, df.shape[1]):
        hdr = str(df.iloc[vh, j]).strip()
        if hdr:
            exp_col_map[hdr] = j

    # write loss values for experiments with full outputs
    changed = False
    for e in experiments:
        y = compute_loss(outputs_cfg, e)
        if y is None:
            continue
        # get experiment header (original col label)
        # if your headers are "experiment #1" style, they’re already stored in cols list
        # but we don't have the header text inside 'e' — we can map using stored column index during read
        j = e.get("_exp_col_index", None)
        if j is None:
            continue
        df.iloc[loss_row_idx, j] = y
        changed = True

    if changed:
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as wr:
            df.to_excel(wr, sheet_name="RUNS", header=False, index=False)

# --------- main ----------
def main():
    if not EXCEL_PATH.exists():
        raise SystemExit(f"Excel not found: {EXCEL_PATH}")

    constants, variables, outputs, opts = read_setup(EXCEL_PATH)
    space = build_space(variables)

    # load RUNS (pivot)
    experiments, exp_col_names, (vars_block, outs_block) = read_runs_pivot(EXCEL_PATH)

    # init/warmstart
    if STATE_PKL.exists():
        with open(STATE_PKL, "rb") as f:
            opt = pickle.load(f)
    else:
        opt = po.Optimizer(space, base_estimator="GP",
                           n_initial_points=opts["n_initial_points"],
                           acq_func=opts["acq_func"])
    warmstart_from_runs_pivot(opt, experiments, variables, outputs)

    # ask next batch
    suggestions = ask_batch(opt, opts["batch_size"], variables, opts["diversity_eps"])

    # print suggestions
    print("\n=== Suggested conditions ===")
    for i, s in enumerate(suggestions, 1):
        print(f"\n-- Suggestion {i}/{len(suggestions)} --")
        for v in variables:
            name, unit = v["name"], v.get("unit","")
            print(f"{name:>22}: {s[name]} {unit}".rstrip())

    # write suggestions sheet
    write_suggestions_sheet(EXCEL_PATH, suggestions)

    # persist state
    with open(STATE_PKL, "wb") as f:
        pickle.dump(opt, f)

    # write/update 'loss' row in RUNS (if outputs are present)
    write_loss_into_runs(EXCEL_PATH, variables, outputs, experiments, exp_col_names)

    print(f"\nWrote {len(suggestions)} suggestion(s) to SUGGESTIONS sheet.")
    print("Copy a row into RUNS (vars block) as a new experiment column, run it, fill outputs block; rerun this script.")

if __name__ == "__main__":
    main()
