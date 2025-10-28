# optimize_lab.py
import pickle
import numpy as np
import pandas as pd
import ProcessOptimizer as po
from pathlib import Path

# ---- 3.1 Define your experimental space
# Example: Temperature [10..60 C], Time [10..180 min], Base equivalents [0.5..2.0] integer steps of 0.1 approximated as real,
# and Solvent as categorical
SPACE = po.space.Space([
    po.space.Real(10.0, 60.0, name="T_C"),
    po.space.Real(10.0, 180.0, name="time_min"),
    po.space.Real(0.5, 2.0, name="base_eq"),
    po.space.Categorical(["MeOH", "MeCN", "IPA"], name="solvent")
])

# ---- 3.2 Create the optimizer
opt = po.Optimizer(
    SPACE,
    base_estimator="GP",          # Gaussian Process (good default for noisy chemistry)
    n_initial_points=5,           # seed points before BO kicks in
    acq_func="EI"                 # Expected Improvement (robust default)
)

# ---- 3.3 (Optional) Warm-start from past data (CSV with columns matching names above + 'response')
def warm_start_from_csv(csv_path):
    if not Path(csv_path).exists():
        return
    df = pd.read_csv(csv_path)
    # build X (list of lists) in the same factor order; y is your response column
    X, y = [], []
    for _, r in df.iterrows():
        X.append([r["T_C"], r["time_min"], r["base_eq"], r["solvent"]])
        y.append(float(r["response"]))
    if X:
        opt.tell(X, y)

# ---- 3.4 Your experiment runner (stub)
def run_experiment(T_C, time_min, base_eq, solvent):
    """
    Replace this body with your lab workflow.
    - You can prompt the operator to run conditions and type the measured result.
    - Or trigger hardware/ELN/LIMS and poll a result.
    Returns: scalar to MINIMIZE (e.g., impurity %). For maximizing yield, pass negative yield or set up accordingly.
    """
    print(f"\n>>> Run: T={T_C:.1f} °C, time={time_min:.1f} min, base_eq={base_eq:.2f}, solvent={solvent}")
    y = float(input("Enter measured response to minimize (e.g., impurity %): "))
    return y

# ---- 3.5 (Optional) constraints filter for unsafe/impossible suggestions
def respects_constraints(x):
    T_C, time_min, base_eq, solvent = x
    # Example constraints:
    if solvent == "MeOH" and T_C > 55:  # column limit for this solvent
        return False
    if base_eq > 1.8 and T_C > 50:      # known runaway risk zone -> forbid
        return False
    return True

# ---- 3.6 Optimization loop (can run in batches)
def optimize(n_total=20, history_csv="history.csv", state_pkl="optimizer.pkl"):
    warm_start_from_csv(history_csv)
    tried = 0 if not Path(history_csv).exists() else len(pd.read_csv(history_csv))
    while tried < n_total:
        # get a candidate; reject until constraints satisfied
        x = opt.ask()
        while not respects_constraints(x):
            x = opt.ask()
        # run the experiment (human-in-the-loop or automated)
        y = run_experiment(*x)
        # record
        opt.tell(x, y)
        # persist state so you can stop/resume any time
        with open(state_pkl, "wb") as f:
            pickle.dump(opt, f)
        row = dict(zip([d.name for d in SPACE.dimensions], x)) | {"response": y}
        pd.DataFrame([row]).to_csv(history_csv, mode="a", header=not Path(history_csv).exists(), index=False)

        tried += 1
        # quick status
        res = opt.get_result()
        print(f"Best so far: x={res.x}, y={res.fun:.4f}")

if __name__ == "__main__":
    optimize(n_total=20)
