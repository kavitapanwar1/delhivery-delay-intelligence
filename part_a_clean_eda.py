from pathlib import Path
import numpy as np
import pandas as pd
from src.shipment_prep import FEATURES, PROMISED_HOURS, clean_shipments, build_features

ROOT = Path(__file__).resolve().parent
for d in [ROOT / "outputs", ROOT / "models", ROOT / "data"]: d.mkdir(exist_ok=True)
raw = pd.read_csv(ROOT / "data/raw_shipments.csv")
hubs = pd.read_csv(ROOT / "data/hubs.csv")
raw_rows = len(raw)
dedup = raw.drop_duplicates().copy()
service_spellings_before = int(dedup["service_type"].nunique())
missing_distance = int(dedup["distance_km"].isna().sum())
missing_weight = int(dedup["parcel_weight_kg"].isna().sum())
sentinel_count = int((dedup["parcel_weight_kg"] > 100).sum())
missing_value = int(dedup["declared_value"].isna().sum())
missing_pay = int(dedup["payment_mode"].isna().sum())
clean, medians = clean_shipments(raw)
service_spellings_after = int(clean["service_type"].nunique())
features = build_features(clean, hubs)
features["is_delayed"] = (features["delivery_hours"] > features["promised_hours"]).astype(int)
assert len(features) == 1200, f"Expected 1200 rows after dedup and hub merge, got {len(features)}"
assert not features[FEATURES + ["delivery_hours", "is_delayed"]].isna().any().any()
features.to_csv(ROOT / "data/modelling_table.csv", index=False)

hours = features["delivery_hours"]
q1, q3 = hours.quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
log_values = np.log(features["declared_value_inr"])
metrics = [
    ("raw_rows", raw_rows), ("duplicate_rows_removed", raw_rows-len(dedup)), ("rows_after_dedup", len(dedup)),
    ("missing_distance_km_before", missing_distance), ("missing_parcel_weight_before", missing_weight),
    ("sentinel_weight_rows", sentinel_count), ("missing_declared_value_before", missing_value),
    ("missing_payment_mode_before", missing_pay), ("service_type_spellings_before", service_spellings_before),
    ("service_type_spellings_after", service_spellings_after),
    ("median_distance_km", medians["distance_km"]), ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]), ("rows_after_hub_merge", len(features)),
    ("nulls_in_features_and_target", int(features[FEATURES+["delivery_hours","is_delayed"]].isna().sum().sum())),
    ("delayed_count", int(features["is_delayed"].sum())), ("delay_rate", features["is_delayed"].mean()),
    ("hours_q1", q1), ("hours_q3", q3), ("hours_lower_bound", lower), ("hours_upper_bound", upper),
    ("hours_outlier_count", int(((hours < lower) | (hours > upper)).sum())),
    ("declared_value_mean_inr", features["declared_value_inr"].mean()),
    ("declared_value_median_inr", features["declared_value_inr"].median()),
    ("log_declared_value_mean", log_values.mean()), ("log_declared_value_median", log_values.median()),
]
pd.DataFrame(metrics, columns=["metric", "value"]).round(4).to_csv(ROOT / "outputs/part_a_checks.csv", index=False)

rows=[]
def add_group(dim, col, group):
    for name, g in features.groupby(group, observed=False, sort=True, dropna=False):
        rows.append({"dimension":dim, "group":str(name), "shipments":len(g), "delayed":int(g["is_delayed"].sum()),
                     "delay_rate":g["is_delayed"].mean(), "mean_hours":g["delivery_hours"].mean()})
add_group("service_type", "service_type", "service_type")
add_group("hub_tier", "hub_tier", "hub_tier")
features["distance_band"] = pd.cut(features["distance_km"], bins=[0,200,400,600,2000], labels=["0-200","201-400","401-600","601+"], right=True, include_lowest=True)
for name,g in features.groupby("distance_band", observed=False, sort=True):
    rows.append({"dimension":"distance_band","group":str(name),"shipments":len(g),"delayed":int(g["is_delayed"].sum()),"delay_rate":g["is_delayed"].mean(),"mean_hours":g["delivery_hours"].mean()})
pd.DataFrame(rows, columns=["dimension","group","shipments","delayed","delay_rate","mean_hours"]).round(4).to_csv(ROOT / "outputs/part_a_delay_by_group.csv", index=False)
features.select_dtypes(include=[np.number]).corr().round(4).to_csv(ROOT / "outputs/part_a_correlations.csv", index_label="column")
print("Part A complete:", len(features), "rows; delay rate", round(features.is_delayed.mean(),4))
