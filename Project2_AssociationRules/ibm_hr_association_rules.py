import os
import itertools
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH  = os.getenv("IBM_HR_PATH")
OUTPUT     = "attrition_groupby_rules.csv"
MIN_COUNT  = 10   # minimum employees in a group to include it
MAX_COLS   = 3    # max number of columns to combine (1, 2, or 3)

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df.drop(columns=["EmployeeCount", "EmployeeNumber", "StandardHours", "Over18"], inplace=True)
print("Loaded:", df.shape)

# ── Bin numerical columns ────────────────────────────────────────────────────
df["Age"] = pd.cut(df["Age"],
    bins=[0, 30, 45, 99], labels=["Young", "Mid", "Senior"])

df["MonthlyIncome"] = pd.cut(df["MonthlyIncome"],
    bins=[0, 3000, 8000, 99999], labels=["Low", "Medium", "High"])

df["DistanceFromHome"] = pd.cut(df["DistanceFromHome"],
    bins=[0, 5, 15, 99], labels=["Near", "Medium", "Far"])

df["YearsAtCompany"] = pd.cut(df["YearsAtCompany"],
    bins=[-1, 2, 7, 99], labels=["New", "Mid", "Senior"])

df["TotalWorkingYears"] = pd.cut(df["TotalWorkingYears"],
    bins=[-1, 5, 15, 99], labels=["Junior", "Mid", "Senior"])

df["YearsInCurrentRole"] = pd.cut(df["YearsInCurrentRole"],
    bins=[-1, 2, 7, 99], labels=["New", "Established", "Long"])

df["YearsSinceLastPromotion"] = pd.cut(df["YearsSinceLastPromotion"],
    bins=[-1, 1, 5, 99], labels=["Recent", "Mid", "Long"])

df["YearsWithCurrManager"] = pd.cut(df["YearsWithCurrManager"],
    bins=[-1, 2, 7, 99], labels=["New", "Mid", "Long"])

df["NumCompaniesWorked"] = pd.cut(df["NumCompaniesWorked"],
    bins=[-1, 2, 5, 99], labels=["Few", "Some", "Many"])

df["TrainingTimesLastYear"] = pd.cut(df["TrainingTimesLastYear"],
    bins=[-1, 1, 3, 99], labels=["Low", "Medium", "High"])

df["PercentSalaryHike"] = pd.cut(df["PercentSalaryHike"],
    bins=[0, 12, 18, 99], labels=["Low", "Medium", "High"])

# ── Map ordinal scales ───────────────────────────────────────────────────────
satisfaction_map = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
for col in ["EnvironmentSatisfaction", "JobSatisfaction",
            "RelationshipSatisfaction", "JobInvolvement", "WorkLifeBalance"]:
    df[col] = df[col].map(satisfaction_map)

df["PerformanceRating"] = df["PerformanceRating"].map({3: "Excellent", 4: "Outstanding"})
df["Education"] = df["Education"].map({
    1: "BelowCollege", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"})
df["JobLevel"] = df["JobLevel"].map({
    1: "Entry", 2: "Junior", 3: "Mid", 4: "Senior", 5: "Executive"})
df["StockOptionLevel"] = df["StockOptionLevel"].map({
    0: "No Stock", 1: "Low", 2: "Medium", 3: "High"})

df.drop(columns=["DailyRate", "HourlyRate", "MonthlyRate"], inplace=True)

# ── Setup ────────────────────────────────────────────────────────────────────
feature_cols = [c for c in df.columns if c != "Attrition"]
overall_attrition_rate = (df["Attrition"] == "Yes").mean()
print(f"Feature columns: {len(feature_cols)}")
print(f"Total employees: {len(df)}")
print(f"Overall attrition rate: {overall_attrition_rate:.2%}")

total_combos = sum(
    len(list(itertools.combinations(feature_cols, n)))
    for n in range(1, MAX_COLS + 1)
)
print(f"Column combinations to evaluate: {total_combos:,}")

# ── Groupby all combinations ──────────────────────────────────────────────────
results = []

for n in range(1, MAX_COLS + 1):
    combos = list(itertools.combinations(feature_cols, n))
    print(f"\nProcessing {len(combos)} {n}-column combinations...")

    for combo in combos:
        combo = list(combo)
        grp = df.groupby(combo + ["Attrition"]).size().unstack(fill_value=0)

        yes = grp.get("Yes", pd.Series(0, index=grp.index))
        no  = grp.get("No",  pd.Series(0, index=grp.index))
        total = yes + no

        for idx in grp.index[total >= MIN_COUNT]:
            idx_tuple = idx if isinstance(idx, tuple) else (idx,)
            rate = yes.loc[idx] / total.loc[idx]
            row = {col: None for col in feature_cols}
            for k, v in zip(combo, idx_tuple):
                row[k] = v
            row.update({
                "n_cols":         n,
                "attrition_yes":  int(yes.loc[idx]),
                "attrition_no":   int(no.loc[idx]),
                "total":          int(total.loc[idx]),
                "attrition_rate": round(rate, 4),
                "attrition_lift": round(rate / overall_attrition_rate, 4),
            })
            results.append(row)

# ── Export ────────────────────────────────────────────────────────────────────
out = pd.DataFrame(results)
out.sort_values("attrition_rate", ascending=False, inplace=True)
out.to_csv(OUTPUT, index=False)

print(f"\nExported {len(out)} rows to {OUTPUT}")
metric_cols = ["n_cols", "attrition_yes", "total", "attrition_rate", "attrition_lift"]
print(f"\nTop 15 highest attrition-rate groups (min {MIN_COUNT} employees):")
print(out[feature_cols + metric_cols].head(15).to_string(index=False))
print(f"\nBottom 10 lowest attrition-rate groups:")
print(out[feature_cols + metric_cols].tail(10).to_string(index=False))
