"""
Step 1 - Generate a realistic SYNTHETIC production dataset.

Why synthetic? Real plant data is confidential. This dataset imitates a
5-line manufacturing plant over 90 days, 3 shifts per day, with hidden
patterns (bad machine on Line 3, material problems on Line 5, weaker
night shift, ...) that the analysis in step 2 should discover.

Outputs (in ./data):
  production.csv      one row per line / shift / day
  downtime_events.csv one row per downtime event, tagged with a 5M+1E category
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

LINES = ["Line 1", "Line 2", "Line 3", "Line 4", "Line 5"]
SHIFTS = ["Morning", "Evening", "Night"]
DATES = pd.date_range("2026-07-01", periods=90, freq="D")
PLANNED_MIN = 480  # 8-hour shift

# ideal cycle time (seconds per unit) for each line
IDEAL_CT = {"Line 1": 30, "Line 2": 36, "Line 3": 42, "Line 4": 30, "Line 5": 48}

# downtime reason -> (5M+1E category, mean duration in minutes)
REASONS = {
    "Equipment breakdown":   ("Machine", 38),
    "Tool wear":             ("Machine", 15),
    "Material shortage":     ("Material", 25),
    "Material quality issue": ("Material", 18),
    "Operator absence":      ("Man", 20),
    "Operator error":        ("Man", 12),
    "Changeover / setup":    ("Method", 22),
    "Calibration":           ("Measurement", 14),
    "Inspection delay":      ("Measurement", 10),
    "Power fluctuation":     ("Environment", 17),
    "High temperature":      ("Environment", 13),
}
REASON_NAMES = list(REASONS)

def reason_weights(line, shift, date):
    w = {r: 1.0 for r in REASON_NAMES}
    w["Equipment breakdown"] = 3.5 if line == "Line 3" else 1.2   # unreliable machine
    w["Material shortage"] = 3.0 if line == "Line 5" else 0.8     # supply problem
    if date.month == 8:
        w["Material shortage"] *= 1.8                             # supplier issue in August
    if shift == "Night":
        w["Operator error"] *= 2.5
        w["Operator absence"] *= 1.8
    if line == "Line 2":
        w["Changeover / setup"] *= 2.5                            # many product changes
    if date.month == 9 and date.day > 15:
        w["Calibration"] *= 2.0
    p = np.array([w[r] for r in REASON_NAMES])
    return p / p.sum()

prod_rows, event_rows = [], []
for date in DATES:
    for line in LINES:
        for shift in SHIFTS:
            n_events = rng.poisson(1.6 if line != "Line 3" else 2.2)
            downtime = 0.0
            for _ in range(n_events):
                reason = rng.choice(REASON_NAMES, p=reason_weights(line, shift, date))
                cat, mean_d = REASONS[reason]
                dur = max(3.0, rng.gamma(shape=2.0, scale=mean_d / 2.0))
                dur = min(dur, 150.0)
                if downtime + dur > 330:
                    break
                downtime += dur
                event_rows.append({"date": date.date(), "shift": shift, "line": line,
                                   "reason": reason, "category_5M1E": cat,
                                   "duration_min": round(dur, 1)})
            runtime = PLANNED_MIN - downtime

            perf = rng.normal(0.90, 0.03)
            perf -= 0.03 if shift == "Night" else 0
            perf -= 0.02 if line == "Line 5" else 0
            perf = float(np.clip(perf, 0.6, 1.0))
            max_units = runtime * 60 / IDEAL_CT[line]
            total = int(max_units * perf)

            q = rng.normal(0.975, 0.008)
            q -= 0.02 if shift == "Night" else 0
            q -= 0.025 if line == "Line 4" else 0
            q = float(np.clip(q, 0.85, 0.999))
            good = int(rng.binomial(total, q))

            prod_rows.append({"date": date.date(), "shift": shift, "line": line,
                              "planned_time_min": PLANNED_MIN,
                              "downtime_min": round(downtime, 1),
                              "ideal_cycle_time_sec": IDEAL_CT[line],
                              "total_units": total, "good_units": good})

pd.DataFrame(prod_rows).to_csv(OUT / "production.csv", index=False)
pd.DataFrame(event_rows).to_csv(OUT / "downtime_events.csv", index=False)
print(f"production.csv: {len(prod_rows)} rows | downtime_events.csv: {len(event_rows)} rows")
