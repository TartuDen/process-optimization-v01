# optimize_lab_step1.py
import ProcessOptimizer as po
import numpy as np
import csv, os, pickle
from pathlib import Path

HISTORY = Path("history.csv")
STATE   = Path("optimizer.pkl")

# --- 1) define the factor space (1 variable for now)
SPACE = po.Space([[0.0, 1.0]])  # name-less is fine at first; we’ll add names later

# --- 2) create or resume optimizer
if STATE.exists():
    with open(STATE, "rb") as f:
        opt = pickle.load(f)
else:
    opt = po.Optimizer(SPACE, base_estimator="GP", n_initial_points=3)

# --- 3) warm start from CSV if present (idempotent)
if HISTORY.exists():
    X, y = [], []
    with open(HISTORY, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            X.append([float(row["x0"])])
            y.append(float(row["response"]))
    if X:
        opt.tell(X, y)

# --- 4) suggest next experiment
x = opt.ask()
x0 = float(x[0])
print(f"\nSuggested condition: x0 = {x0:.4f}")

# --- 5) collect measurement from user (MINIMIZE; e.g., impurity %)
resp = None
while resp is None:
    try:
        resp = float(input("Enter measured response to MINIMIZE (e.g., impurity %): "))
    except Exception:
        print("Please enter a number.")

# --- 6) record result and persist
opt.tell([x0], resp)

# save state
with open(STATE, "wb") as f:
    pickle.dump(opt, f)

# append to CSV (create header if new)
header_needed = not HISTORY.exists()
with open(HISTORY, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["x0", "response"])
    if header_needed:
        w.writeheader()
    w.writerow({"x0": x0, "response": resp})

# --- 7) quick status
res = opt.get_result()
print(f"Logged. Best so far: x*={res.x}  y*={res.fun:.6g}")
