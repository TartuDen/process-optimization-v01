import ProcessOptimizer as po
from ProcessOptimizer.space import Real
import pickle, csv
from pathlib import Path

# ===================== USER TUNABLES =====================
# Scale constants for your current batch (used only for logging/intuition)
SUBSTRATE_G = 11.0           # grams of IP.4
DEFAULT_STIR_RPM = 300       # if you keep it constant, it's meta-info only

# Factor space (bounds chosen to stay near your current recipe but allow useful moves)
SPACE = po.Space([
    Real(70.0, 95.0,   name="T_C"),          # °C
    Real(10.0, 25.0,   name="H2_bar"),       # hydrogen pressure, bar
    Real(6.0,  24.0,   name="time_h"),       # hours
    Real(5.0,  15.0,   name="PdC_wtpc"),     # wt% 10% Pd/C vs substrate (as "wet" basis)
    Real(0.30, 0.70,   name="MSA_frac"),     # MSA mass fraction in MSA/water (w/w)
    Real(10.0, 30.0,   name="solv_mL_per_g") # total solvent volume per gram substrate
])

# Objective weights (sum ≈ 1). We MINIMIZE this loss.
# Heavier emphasis on RRT 1.19 + 1.20 (you can tweak after a few runs):
W_RRT      = 0.60   # weight on (RRT1.19 + RRT1.20)
W_OFFSPEC  = 0.25   # penalty for (100 - purity)
W_YIELD    = 0.15   # penalty for (100 - isolated_yield)
# =========================================================

HISTORY = Path("hydro_history.csv")
STATE   = Path("optimizer.pkl")

def load_or_init_optimizer():
    if STATE.exists():
        with open(STATE, "rb") as f:
            opt = pickle.load(f)
    else:
        opt = po.Optimizer(
            SPACE,
            base_estimator="GP",
            n_initial_points=5,
            acq_func="EI",
        )
    return opt

def warm_start(opt):
    if not HISTORY.exists():
        return
    X, y = [], []
    with open(HISTORY, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            x = [float(row[n]) for n in ["T_C","H2_bar","time_h","PdC_wtpc","MSA_frac","solv_mL_per_g"]]
            loss = float(row["loss"])
            X.append(x); y.append(loss)
    if X:
        opt.tell(X, y)

def respects_constraints(x):
    T_C, H2_bar, time_h, PdC_wtpc, MSA_frac, solv_mL_per_g = x
    # Example feasibility/safety sanity checks; edit to your SOPs
    if T_C > 95.0 or T_C < 70.0: return False
    if H2_bar < 8.0 or H2_bar > 30.0: return False
    if time_h < 2.0 or time_h > 30.0: return False
    if PdC_wtpc < 3.0 or PdC_wtpc > 20.0: return False
    if MSA_frac < 0.25 or MSA_frac > 0.80: return False
    if solv_mL_per_g < 8.0 or solv_mL_per_g > 40.0: return False
    # Gentle interaction rule example: very high T with very high acid → avoid
    if T_C > 92.0 and MSA_frac > 0.6: return False
    return True

def ask_feasible(opt):
    x = opt.ask()
    # Try a few times to hit a feasible point (GP + bounds usually OK)
    for _ in range(20):
        if respects_constraints(x):
            return x
        x = opt.ask()
    return x  # last resort

def compute_loss(hplc_purity_pct, rrt119_pct, rrt120_pct, isolated_yield_pct):
    # All inputs expected as percentages 0..100
    offspec = max(0.0, 100.0 - hplc_purity_pct)   # penalty if purity < 100
    yield_pen = max(0.0, 100.0 - isolated_yield_pct)
    return (W_RRT * (rrt119_pct + rrt120_pct)
            + W_OFFSPEC * offspec
            + W_YIELD * yield_pen)

def append_history(row_dict):
    header = [
        "T_C","H2_bar","time_h","PdC_wtpc","MSA_frac","solv_mL_per_g",
        "calc_total_mL","hplc_purity_pct","rrt119_pct","rrt120_pct",
        "isolated_yield_pct","loss"
    ]
    is_new = not HISTORY.exists()
    with open(HISTORY, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if is_new: w.writeheader()
        w.writerow(row_dict)

def main():
    opt = load_or_init_optimizer()
    warm_start(opt)

    # === Suggest next run ===
    x = ask_feasible(opt)
    T_C, H2_bar, time_h, PdC_wtpc, MSA_frac, solv_mL_per_g = x
    calc_total_mL = solv_mL_per_g * SUBSTRATE_G

    print("\n=== Suggested hydrogenation conditions ===")
    print(f"Substrate: {SUBSTRATE_G:.2f} g IP.4    (stir {DEFAULT_STIR_RPM} rpm if constant)")
    print(f"T         : {T_C:.1f} °C")
    print(f"H2        : {H2_bar:.1f} bar")
    print(f"Time      : {time_h:.1f} h")
    print(f"Pd/C      : {PdC_wtpc:.1f} wt% of substrate (10% Pd/C, wet)")
    print(f"MSA frac  : {MSA_frac:.2f} (w/w in MSA/H2O)")
    print(f"Solvent   : {solv_mL_per_g:.1f} mL per g substrate → total ≈ {calc_total_mL:.0f} mL")

    # === Enter measured outcomes after you run it ===
    print("\nEnter measured outcomes (percentages). Use decimal numbers, e.g., 75.6")
    def ask_float(label):
        while True:
            try:
                return float(input(f"{label}: ").strip())
            except Exception:
                print("Please enter a number, e.g., 75.6")

    hplc_purity_pct   = ask_float("HPLC purity of IP.5 (%)")
    rrt119_pct        = ask_float("Impurity RRT 1.19 (%)")
    rrt120_pct        = ask_float("Impurity RRT 1.20 (%)")
    isolated_yield_pct= ask_float("Isolated yield (%)")

    loss = compute_loss(hplc_purity_pct, rrt119_pct, rrt120_pct, isolated_yield_pct)

    opt.tell([T_C, H2_bar, time_h, PdC_wtpc, MSA_frac, solv_mL_per_g], loss)

    with open(STATE, "wb") as f:
        pickle.dump(opt, f)

    append_history({
        "T_C":T_C, "H2_bar":H2_bar, "time_h":time_h, "PdC_wtpc":PdC_wtpc,
        "MSA_frac":MSA_frac, "solv_mL_per_g":solv_mL_per_g,
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
