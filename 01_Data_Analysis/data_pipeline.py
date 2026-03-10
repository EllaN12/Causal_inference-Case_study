"""
===================================================
 Causal Inference Case Study — Data Pipeline
===================================================
Restaurant Rating Attribution Analysis
Author: Ella Ndala | ndallaella@gmail.com

This pipeline orchestrates all data preparation steps:
  Phase 1 → Data Ingestion      : Load 9 raw CSV files
  Phase 2 → Preprocessing       : Clean & merge data
  Phase 3 → Feature Engineering : Create engineered features
  Phase 4 → Encoding & Output   : Prepare final modelling dataset

Usage:
    python data_pipeline.py

Outputs:
    01_Data_Analysis/final_data.pkl       — cleaned merged dataframe
    Reports/EDA_Reports/eda_report.html   — automated EDA profile report
"""

# ============================================================
# 0. IMPORTS
# ============================================================
import os
import glob
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from math import radians, cos, sin, asin, sqrt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR   = Path(__file__).resolve().parent
RAW_DIR    = BASE_DIR / "00_Raw_Data"
OUT_DIR    = BASE_DIR / "01_Data_Analysis"
REPORT_DIR = BASE_DIR / "Reports" / "EDA_Reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PHASE 1 — DATA INGESTION
# ============================================================
def load_raw_data(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """
    Load all CSV files from 00_Raw_Data into a dictionary of DataFrames.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys = file stem (e.g. 'rating_final'), values = DataFrames.
    """
    print("\n" + "="*55)
    print("PHASE 1 — DATA INGESTION")
    print("="*55)

    csv_paths  = glob.glob(str(raw_dir / "*.csv"))
    dataframes = {}

    for path in csv_paths:
        name = Path(path).stem
        try:
            df = pd.read_csv(path)
            dataframes[name] = df
            print(f"  ✓ Loaded: {name:35s}  shape={df.shape}")
        except Exception as exc:
            print(f"  ✗ Error loading {name}: {exc}")

    print(f"\n  {len(dataframes)} files loaded successfully.")
    return dataframes


# ============================================================
# PHASE 2 — MERGING & INITIAL PREPROCESSING
# ============================================================
def merge_data(dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge the 9 source tables into a single flat DataFrame.

    Patron pipeline  : userprofile → userpayment → usercuisine → rating_final
    Restaurant pipeline: geoplaces2 → chefmozaccepts → chefmozparking
                         → chefmozcuisine → chefmozhours4
    Combined          : patrons LEFT JOIN restaurants ON placeID

    Returns
    -------
    pd.DataFrame  — merged but not yet cleaned
    """
    print("\n" + "="*55)
    print("PHASE 2 — MERGING DATA")
    print("="*55)

    # — Patron side —
    patrons_df = (
        dataframes["userprofile"]
        .merge(dataframes["userpayment"],   on="userID", how="left")
        .merge(dataframes["usercuisine"],   on="userID", how="left")
        .merge(dataframes["rating_final"],  on="userID", how="left")
    )
    patrons_df.rename(columns={"latitude": "p.latitude",
                                "longitude": "p.longitude"}, inplace=True)
    print(f"  Patrons merged:     {patrons_df.shape}")

    # — Restaurant side —
    restaurant_df = (
        dataframes["geoplaces2"]
        .merge(dataframes["chefmozaccepts"], on="placeID", how="left")
        .merge(dataframes["chefmozparking"], on="placeID", how="left")
        .merge(dataframes["chefmozcuisine"], on="placeID", how="left")
        .merge(dataframes["chefmozhours4"],  on="placeID", how="left")
    )
    print(f"  Restaurants merged: {restaurant_df.shape}")

    # — Combined —
    data = patrons_df.merge(restaurant_df, on="placeID", how="left")
    print(f"  Combined dataset:   {data.shape}")

    return data


def clean_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean categorical columns:
      - Replace '?' sentinel values with NaN
      - Rename columns for clarity
      - Parse business hours into time-of-day categories
    """
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cat_df   = df[cat_cols].copy()

    # Replace sentinel values
    cat_df.replace("?", np.nan, inplace=True)
    cat_df["days"] = cat_df["days"].replace("Mon;Tue;Wed;Thu;Fri;", "weekdays")

    # Rename for clarity
    cat_df.rename(columns={
        "Rcuisine_x":  "User_cuisine",
        "name":        "restaurant_name",
        "Rpayment":    "restaurant_payment",
        "Rcuisine_y":  "restaurant_specialty",
    }, inplace=True)

    # Drop unnecessary columns (ignore if missing)
    cat_df.drop(columns=[c for c in ["fax", "url"] if c in cat_df.columns], inplace=True)

    # — Business hours parsing —
    cat_df["hours"] = cat_df["hours"].astype(str)
    cat_df["start_time"] = cat_df["hours"].str[:-1].str.split("-").str[0]
    cat_df["end_time"]   = cat_df["hours"].str[:-1].str.split("-").str[1]
    cat_df.replace("na", np.nan, inplace=True)

    for col in ["start_time", "end_time"]:
        cat_df[col] = pd.to_datetime(cat_df[col], format="%H:%M", errors="coerce").dt.time

    cat_df["Business_hours"] = cat_df.apply(_categorize_time, axis=1)
    cat_df.drop(columns=["hours", "start_time", "end_time", "days"],
                errors="ignore", inplace=True)

    return cat_df


def _categorize_time(row) -> str:
    """Classify restaurant operating hours into a time-of-day bucket."""
    if pd.isna(row.get("start_time")) or pd.isna(row.get("end_time")):
        return np.nan
    s, e = row["start_time"], row["end_time"]
    t = pd.to_datetime
    if s >= t("06:00").time() and e <= t("12:00").time():
        return "Morning"
    elif s >= t("12:00").time() and s < t("18:00").time():
        return "Afternoon"
    elif s >= t("18:00").time() or (s < t("06:00").time() and e >= t("06:00").time()):
        return "Evening"
    elif e < s:
        return "24H"
    elif s >= t("06:00").time() and e >= t("18:00").time():
        return "Full Day"
    else:
        return "Night"


# ============================================================
# PHASE 3 — FEATURE ENGINEERING
# ============================================================
def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Geodesic distance between two (lat, lon) points in kilometres."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    return 2 * R * asin(sqrt(a))


def engineer_features(df: pd.DataFrame,
                      num_df: pd.DataFrame,
                      cat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features on top of the cleaned dataset.

    Features added
    --------------
    patron_restaurant_distance  : geodesic km between user and restaurant
    location_cluster            : KMeans cluster of restaurant location
    age_group                   : binned age from birth_year
    cuisine_match_score         : Jaccard similarity of cuisine preferences
    """
    print("\n" + "="*55)
    print("PHASE 3 — FEATURE ENGINEERING")
    print("="*55)

    # Start with combined numerical + categorical base
    feat_df = pd.concat([num_df.reset_index(drop=True),
                         cat_df.reset_index(drop=True)], axis=1)

    # ── Patron-restaurant distance ──
    coord_cols = ["p.latitude", "p.longitude", "latitude", "longitude"]
    if all(c in feat_df.columns for c in coord_cols):
        feat_df["patron_restaurant_distance"] = feat_df.apply(
            lambda r: _haversine_km(r["p.latitude"], r["p.longitude"],
                                    r["latitude"],   r["longitude"])
            if pd.notna(r["p.latitude"]) and pd.notna(r["latitude"]) else np.nan,
            axis=1
        )
        print("  ✓ patron_restaurant_distance  (Haversine km)")

    # ── Location clusters ──
    geo_cols = ["latitude", "longitude"]
    if all(c in feat_df.columns for c in geo_cols):
        geo_clean = feat_df[geo_cols].dropna()
        if len(geo_clean) > 5:
            km = KMeans(n_clusters=5, random_state=42, n_init=10)
            clusters = km.fit_predict(geo_clean)
            feat_df.loc[geo_clean.index, "location_cluster"] = clusters
            print("  ✓ location_cluster            (KMeans k=5)")

    # ── Age groups ──
    if "birth_year" in feat_df.columns:
        current_year = 2026
        feat_df["age"] = current_year - feat_df["birth_year"]
        feat_df["age_group"] = pd.cut(
            feat_df["age"],
            bins=[0, 25, 35, 50, 200],
            labels=["18-25", "26-35", "36-50", "50+"],
            right=True
        ).astype(str)
        print("  ✓ age_group                   (18-25 / 26-35 / 36-50 / 50+)")

    # ── Cuisine match score (Jaccard similarity) ──
    if "User_cuisine" in feat_df.columns and "restaurant_specialty" in feat_df.columns:
        def jaccard(u, r):
            if pd.isna(u) or pd.isna(r):
                return 0.0
            u_set = set(str(u).lower().split(";"))
            r_set = set(str(r).lower().split(";"))
            inter = len(u_set & r_set)
            union = len(u_set | r_set)
            return inter / union if union else 0.0

        feat_df["cuisine_match_score"] = feat_df.apply(
            lambda r: jaccard(r["User_cuisine"], r["restaurant_specialty"]), axis=1
        )
        print("  ✓ cuisine_match_score         (Jaccard similarity)")

    print(f"\n  Final feature matrix: {feat_df.shape}")
    return feat_df


# ============================================================
# PHASE 3b — IMPUTATION
# ============================================================
def impute_missing(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute remaining missing values.
    - Categorical columns: most-frequent strategy
    - Numerical columns: median strategy
    """
    print("\n" + "="*55)
    print("PHASE 3b — MISSING VALUE IMPUTATION")
    print("="*55)

    # Categorical
    cat_cols = feat_df.select_dtypes(include="object").columns.tolist()
    missing_cat = [c for c in cat_cols if feat_df[c].isnull().any()]
    if missing_cat:
        imp_cat = SimpleImputer(strategy="most_frequent")
        feat_df[missing_cat] = imp_cat.fit_transform(feat_df[missing_cat])
        print(f"  ✓ Mode-imputed {len(missing_cat)} categorical columns")

    # Numerical
    num_cols = feat_df.select_dtypes(include=[np.number]).columns.tolist()
    missing_num = [c for c in num_cols if feat_df[c].isnull().any()]
    if missing_num:
        imp_num = SimpleImputer(strategy="median")
        feat_df[missing_num] = imp_num.fit_transform(feat_df[missing_num])
        print(f"  ✓ Median-imputed {len(missing_num)} numerical columns")

    remaining = feat_df.isnull().sum().sum()
    print(f"  Remaining nulls after imputation: {remaining}")
    return feat_df


# ============================================================
# OPTIONAL — EDA PROFILING REPORT
# ============================================================
def generate_eda_report(df: pd.DataFrame, output_dir: Path = REPORT_DIR) -> None:
    """Generate a ydata-profiling HTML report (optional, requires ydata-profiling)."""
    try:
        from ydata_profiling import ProfileReport
        report = ProfileReport(df, title="Restaurant Ratings — EDA Report", minimal=True)
        out_path = output_dir / "eda_report.html"
        report.to_file(out_path)
        print(f"  ✓ EDA report saved: {out_path}")
    except ImportError:
        print("  ⚠  ydata-profiling not installed — skipping EDA report")
    except Exception as exc:
        print(f"  ⚠  EDA report generation failed: {exc}")


# ============================================================
# MAIN PIPELINE ORCHESTRATOR
# ============================================================
def run_pipeline(
    save_output: bool = True,
    run_eda_report: bool = False,
) -> pd.DataFrame:
    """
    Execute the full data pipeline end-to-end.

    Parameters
    ----------
    save_output     : save final_data.pkl to 01_Data_Analysis/
    run_eda_report  : generate ydata-profiling HTML report

    Returns
    -------
    pd.DataFrame — final preprocessed and feature-engineered dataset
    """
    print("\n" + "="*55)
    print("  CAUSAL INFERENCE — DATA PIPELINE")
    print("  Restaurant Rating Attribution Analysis")
    print("="*55)

    # Phase 1: Load
    dataframes = load_raw_data()

    # Phase 2: Merge
    merged = merge_data(dataframes)

    # Separate numerical and categorical
    num_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    num_df   = merged[num_cols].copy()

    # Phase 2b: Clean categorical
    print("\n" + "="*55)
    print("PHASE 2b — CATEGORICAL CLEANING")
    print("="*55)
    cat_df = clean_categorical(merged)
    print(f"  Cleaned categorical shape: {cat_df.shape}")

    # Phase 3: Feature engineering
    feat_df = engineer_features(merged, num_df, cat_df)

    # Phase 3b: Imputation
    final_df = impute_missing(feat_df)

    # Optional: EDA report
    if run_eda_report:
        generate_eda_report(final_df)

    # Save output
    if save_output:
        out_path = OUT_DIR / "final_data.pkl"
        final_df.to_pickle(out_path)
        print(f"\n  ✓ Final dataset saved: {out_path}")
        print(f"    Shape: {final_df.shape}")
        print(f"    Columns: {final_df.columns.tolist()}")

    print("\n" + "="*55)
    print("  PIPELINE COMPLETE")
    print("="*55 + "\n")

    return final_df


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    final_data = run_pipeline(
        save_output=True,
        run_eda_report=False,   # set True to generate HTML profiling report
    )
    print(final_data.head())
    print(f"\nFinal shape: {final_data.shape}")
