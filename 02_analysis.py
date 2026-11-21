"""
Step 2 - OEE & downtime analysis (Pandas + Matplotlib).

OEE = Availability x Performance x Quality
  Availability = run time / planned time
  Performance  = (total units x ideal cycle time) / run time
  Quality      = good units / total units

Downtime is also grouped by the 5M+1E categories
(Man, Machine, Material, Method, Measurement, Environment) and ranked
with a Pareto chart.

Outputs: charts in ./charts and tables in ./results
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent
CH, RES = BASE / "charts", BASE / "results"
CH.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)

prod = pd.read_csv(BASE / "data" / "production.csv", parse_dates=["date"])
ev = pd.read_csv(BASE / "data" / "downtime_events.csv", parse_dates=["date"])

# ---------- 1. OEE calculation ----------
prod["run_time_min"] = prod["planned_time_min"] - prod["downtime_min"]

def oee_table(df, by):
    g = df.groupby(by).agg(planned=("planned_time_min", "sum"),
                           run=("run_time_min", "sum"),
                           units=("total_units", "sum"),
                           good=("good_units", "sum"),
                           ideal_sec=("ideal_cycle_time_sec", "mean"))
    # performance uses each row's own cycle time, so compute it row-wise first
    tmp = df.assign(ideal_time_min=df["total_units"] * df["ideal_cycle_time_sec"] / 60)
    ideal = tmp.groupby(by)["ideal_time_min"].sum()
    g["Availability"] = g["run"] / g["planned"]
    g["Performance"] = ideal / g["run"]
    g["Quality"] = g["good"] / g["units"]
    g["OEE"] = g["Availability"] * g["Performance"] * g["Quality"]
    return g[["Availability", "Performance", "Quality", "OEE"]]

by_line = oee_table(prod, "line")
by_shift = oee_table(prod, "shift")
prod["month"] = prod["date"].dt.to_period("M").astype(str)
by_month = oee_table(prod, "month")
by_line_shift = oee_table(prod, ["line", "shift"])["OEE"].unstack()
overall = oee_table(prod.assign(all="Plant"), "all")

for name, t in [("oee_by_line", by_line), ("oee_by_shift", by_shift),
                ("oee_by_month", by_month), ("oee_overall", overall),
                ("oee_line_x_shift", by_line_shift)]:
    (t * 100).round(1).to_csv(RES / f"{name}.csv")

# ---------- 2. Downtime: Pareto + 5M+1E ----------
by_reason = ev.groupby("reason")["duration_min"].sum().sort_values(ascending=False)
by_reason_df = by_reason.to_frame("downtime_min")
by_reason_df["share_%"] = (by_reason / by_reason.sum() * 100).round(1)
by_reason_df["cumulative_%"] = by_reason_df["share_%"].cumsum().round(1)
by_reason_df.to_csv(RES / "downtime_pareto.csv")

by_cat = ev.groupby("category_5M1E")["duration_min"].sum().sort_values(ascending=False)
(by_cat / by_cat.sum() * 100).round(1).to_csv(RES / "downtime_by_5M1E.csv", header=["share_%"])

cat_line = ev.pivot_table(index="line", columns="category_5M1E",
                          values="duration_min", aggfunc="sum").fillna(0)
cat_line.round(0).to_csv(RES / "downtime_line_x_5M1E.csv")

reason_month = ev.assign(month=ev["date"].dt.to_period("M").astype(str)) \
    .pivot_table(index="reason", columns="month", values="duration_min", aggfunc="sum").fillna(0)
reason_month.round(0).to_csv(RES / "downtime_reason_by_month.csv")

# ---------- 3. Charts ----------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREEN, RED, GREY = "#2E75B6", "#ED7D31", "#548235", "#C00000", "#7F7F7F"

# 3.1 OEE by line (with A/P/Q)
fig, ax = plt.subplots(figsize=(9, 5))
x = range(len(by_line)); w = 0.2
for i, (col, c) in enumerate([("Availability", BLUE), ("Performance", ORANGE),
                              ("Quality", GREEN), ("OEE", RED)]):
    ax.bar([p + i * w for p in x], by_line[col] * 100, w, label=col, color=c)
ax.axhline(85, color=GREY, ls="--", lw=1, label="World-class OEE (85%)")
ax.set_xticks([p + 1.5 * w for p in x]); ax.set_xticklabels(by_line.index)
ax.set_ylabel("%"); ax.set_ylim(0, 105)
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
ax.set_title("OEE and its components by production line")
fig.tight_layout(); fig.savefig(CH / "1_oee_by_line.png", dpi=150); plt.close(fig)

# 3.2 Monthly OEE trend
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(by_month.index, by_month["OEE"] * 100, marker="o", color=RED, label="OEE")
ax.plot(by_month.index, by_month["Availability"] * 100, marker="o", color=BLUE, label="Availability")
for xi, v in zip(by_month.index, by_month["OEE"] * 100):
    ax.annotate(f"{v:.1f}", (xi, v), textcoords="offset points", xytext=(0, 8), ha="center")
ax.set_ylabel("%"); ax.set_ylim(50, 100); ax.legend(); ax.set_title("Monthly OEE trend")
fig.tight_layout(); fig.savefig(CH / "2_monthly_trend.png", dpi=150); plt.close(fig)

# 3.3 Pareto of downtime reasons
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.bar(by_reason.index, by_reason.values / 60, color=BLUE)
ax.set_ylabel("Downtime (hours)"); ax.tick_params(axis="x", rotation=40)
for lbl in ax.get_xticklabels(): lbl.set_ha("right")
ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
ax2.plot(by_reason.index, by_reason_df["cumulative_%"], color=RED, marker="o")
ax2.axhline(80, color=GREY, ls="--", lw=1); ax2.set_ylim(0, 105); ax2.set_ylabel("Cumulative %")
ax.set_title("Pareto chart - downtime by reason")
fig.tight_layout(); fig.savefig(CH / "3_downtime_pareto.png", dpi=150); plt.close(fig)

# 3.4 5M+1E share
fig, ax = plt.subplots(figsize=(7.5, 4.5))
shares = by_cat / by_cat.sum() * 100
ax.barh(shares.index[::-1], shares.values[::-1], color=ORANGE)
for i, v in enumerate(shares.values[::-1]): ax.text(v + 0.5, i, f"{v:.1f}%", va="center")
ax.set_xlabel("% of total downtime"); ax.set_title("Downtime by 5M+1E category")
fig.tight_layout(); fig.savefig(CH / "4_downtime_5M1E.png", dpi=150); plt.close(fig)

# 3.5 Heatmap line x shift
fig, ax = plt.subplots(figsize=(7, 4.5))
im = ax.imshow(by_line_shift.values * 100, cmap="RdYlGn", vmin=55, vmax=85)
ax.set_xticks(range(len(by_line_shift.columns))); ax.set_xticklabels(by_line_shift.columns)
ax.set_yticks(range(len(by_line_shift.index))); ax.set_yticklabels(by_line_shift.index)
for i in range(by_line_shift.shape[0]):
    for j in range(by_line_shift.shape[1]):
        ax.text(j, i, f"{by_line_shift.values[i, j] * 100:.1f}", ha="center", va="center")
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("OEE % - line x shift"); fig.colorbar(im, label="OEE %")
fig.tight_layout(); fig.savefig(CH / "5_oee_heatmap.png", dpi=150); plt.close(fig)

# ---------- 4. Print the key numbers ----------
pd.set_option("display.width", 140)
print("=== OVERALL ==="); print((overall * 100).round(1))
print("\n=== BY LINE ==="); print((by_line * 100).round(1))
print("\n=== BY SHIFT ==="); print((by_shift * 100).round(1))
print("\n=== BY MONTH ==="); print((by_month * 100).round(1))
print("\n=== PARETO ==="); print(by_reason_df)
print("\n=== 5M+1E ==="); print((by_cat / by_cat.sum() * 100).round(1))
print("\n=== LINE x 5M+1E (min) ==="); print(cat_line.round(0))
print("\n=== REASON x MONTH (min) ==="); print(reason_month.round(0))
