# optimize_hydrogenation.py  (rpm added as a factor)
import ProcessOptimizer as po
from ProcessOptimizer.space import Real  # use Integer if rpm has discrete steps
import pickle, csv
from pathlib import Path

# ===================== USER TUNABLES =====================
SUBSTRATE_G = 11.0

SPACE = po.Space([
    Real(70.0, 95.0,   name="T_C"),            # °C
    Real(10.0, 25.0,   name="H2_bar"),         # bar
    Real(6.0,  24.0,   name="time_h"),         # h
    Real(5.0,  15.0,   name="PdC_wtpc"),       # wt% 10% Pd/C vs substrate (wet)
    Real(0.30, 0.70,   name="MSA_frac"),       # MSA mass fraction (w/w)
    Real(10.0, 30.0,   name="solv_mL_per_g"),  # mL/g substrate
    Real(200.0, 400.0, name="rpm")             # stirring speed
])

# Single-objective loss (MINIMIZE)
W_RRT      = 0.60
W_OFFSPEC  = 0.25
W_YIELD    = 0.15
# =========================================================

HISTORY = Path("hydro_history.csv")
STATE   = Path("optimizer.pkl")

def load_or_init_optimizer():
    if STATE.exists():
        with open(STATE, "rb") as f:
            return pickle.load(f)
    return po.Optimizer(SPACE, base_estimator="GP", n_initial_points=5, acq_func="EI")

def warm_start(opt):
    if not HISTORY.exists():
        return
    X, y = [], []
    with open(HISTORY, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            x = [float(row[n]) for n in [
                "T_C","H2_bar","time_h","PdC_wtpc","MSA_frac","solv_mL_per_g","rpm"
            ]]
            X.append(x)
            y.append(float(row["loss"]))
    if X:
        opt.tell(X, y)

def respects_constraints(x):
    T_C, H2_bar, time_h, PdC_wtpc, MSA_frac, solv_mL_per_g, rpm = x
    # base bounds (belt & suspenders with Space bounds)
    if not (70.0 <= T_C <= 95.0): return False
    if not (10.0 <= H2_bar <= 25.0): return False
    if not (6.0  <= time_h <= 24.0): return False
    if not (5.0  <= PdC_wtpc <= 15.0): return False
    if not (0.30 <= MSA_frac <= 0.70): return False
    if not (10.0 <= solv_mL_per_g <= 30.0): return False
    if not (200.0 <= rpm <= 400.0): return False

    # example interaction guards (tweak to your SOPs):
    if T_C > 92.0 and MSA_frac > 0.60:
        return False
    # if you observe attrition/foaming at high rpm + high gas load, guard here:
    # if rpm > 380 and H2_bar > 22: return False

    return True

def ask_feasible(opt):
    for _ in range(25):
        x = opt.ask()
        if respects_constraints(x):
            return x
    return x  # fallback

def compute_loss(hplc_purity_pct, rrt119_pct, rrt120_pct, isolated_yield_pct):
    offspec   = max(0.0, 100.0 - hplc_purity_pct)
    yield_pen = max(0.0, 100.0 - isolated_yield_pct)
    return (W_RRT * (rrt119_pct + rrt120_pct)
            + W_OFFSPEC * offspec
            + W_YIELD * yield_pen)

def append_history(row_dict):
    header = [
        "T_C","H2_bar","time_h","PdC_wtpc","MSA_frac","solv_mL_per_g","rpm",
        "calc_total_mL","hplc_purity_pct","rrt119_pct","rrt120_pct","isolated_yield_pct","loss"
    ]
    is_new = not HISTORY.exists()
    with open(HISTORY, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if is_new: w.writeheader()
        w.writerow(row_dict)

def main():
    opt = load_or_init_optimizer()
    warm_start(opt)

    x = ask_feasible(opt)
    T_C, H2_bar, time_h, PdC_wtpc, MSA_frac, solv_mL_per_g, rpm = x
    calc_total_mL = solv_mL_per_g * SUBSTRATE_G

    print("\n=== Suggested hydrogenation conditions ===")
    print(f"Substrate   : {SUBSTRATE_G:.2f} g IP.4")
    print(f"T           : {T_C:.1f} °C")
    print(f"H2          : {H2_bar:.1f} bar")
    print(f"Time        : {time_h:.1f} h")
    print(f"Pd/C        : {PdC_wtpc:.1f} wt% of substrate (10% Pd/C, wet)")
    print(f"MSA fraction: {MSA_frac:.2f} (w/w in MSA/H2O)")
    print(f"Solvent     : {solv_mL_per_g:.1f} mL/g → total ≈ {calc_total_mL:.0f} mL")
    print(f"Stir speed  : {rpm:.0f} rpm")

    print("\nEnter measured outcomes (percentages, e.g., 75.6)")
    def ask_float(label):
        while True:
            try: return float(input(f"{label}: ").strip())
            except Exception: print("Please enter a number, e.g., 75.6")

    hplc_purity_pct    = ask_float("HPLC purity of IP.5 (%)")
    rrt119_pct         = ask_float("Impurity RRT 1.19 (%)")
    rrt120_pct         = ask_float("Impurity RRT 1.20 (%)")
    isolated_yield_pct = ask_float("Isolated yield (%)")

    loss = compute_loss(hplc_purity_pct, rrt119_pct, rrt120_pct, isolated_yield_pct)

    opt.tell([T_C, H2_bar, time_h, PdC_wtpc, MSA_frac, solv_mL_per_g, rpm], loss)

    with open(STATE, "wb") as f:
        pickle.dump(opt, f)

    append_history({
        "T_C":T_C, "H2_bar":H2_bar, "time_h":time_h, "PdC_wtpc":PdC_wtpc,
        "MSA_frac":MSA_frac, "solv_mL_per_g":solv_mL_per_g, "rpm": rpm,
        "calc_total_mL": calc_total_mL,
        "hplc_purity_pct": hplc_purity_pct,
        "rrt119_pct": rrt119_pct,
        "rrt120_pct": rrt120_pct,
        "isolated_yield_pct": isolated_yield_pct,
        "loss": loss
    })

    res = opt.get_result()
    print("\nLogged. Current best (lowest loss):")
    print(f"x* = {res.x}")
    print(f"loss* = {res.fun:.3f}")
    print("Run the script again for the next suggested experiment.")

if __name__ == "__main__":
    main()
