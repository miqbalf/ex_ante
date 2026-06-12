import os
import re

import pandas as pd

TREE_EVIDENCE_MEASUREMENT = "Nr Tree Evidence Expost"
LARGE_TREE_MEASUREMENT = "Nr Large Tree Expost"


def apply_date_to_csv_path(csv_path: str, current_date: str) -> str:
    """Insert or replace YYYY-MM-DD in the filename; keep the parent directory."""
    date_pattern = r"\d{4}-\d{2}-\d{2}"
    dirname = os.path.dirname(csv_path) or "."
    basename = os.path.basename(csv_path)
    if re.search(date_pattern, basename):
        basename = re.sub(date_pattern, current_date, basename)
    else:
        stem, ext = os.path.splitext(basename)
        basename = f"{current_date}_{stem}{ext}"
    return os.path.join(dirname, basename)


def change_year_int(year):
    return int(year.split("year")[-1].replace(" ", ""))


def adding_zero_meas(plot_id, plot_area, meas_year):
    obj_df = {
        "planting_year": [meas_year - 1],
        "measurement_year": [meas_year],
        "plot_area_ha": [plot_area],
        "plot_id": [plot_id],
        "co2_tree_captured_tonnes": [0],
        "tree_dbh_mm": [0],
        "tree_total_biomass_tonnes": [0],
        "measurement_id": [0],
    }

    return pd.DataFrame(obj_df)


def adding_input_gcs_zero_row(
    *,
    plot_id,
    plot_area_ha,
    planting_year,
    measurement_year,
    species,
    year_start,
    is_replanting=False,
    num_trees_init=0,
    num_trees_survived=0,
):
    return pd.DataFrame(
        {
            "planting_year": [planting_year],
            "measurement_year": [measurement_year],
            "plot_area_ha": [plot_area_ha],
            "plot_id": [plot_id],
            "species": [species],
            "year_start": [year_start],
            "is_replanting": [is_replanting],
            "co2_tree_captured_tonnes": [0],
            "tree_dbh_mm": [0],
            "tree_total_biomass_tonnes": [0],
            "measurement_id": [0],
            "num_trees_init": [num_trees_init],
            "num_trees_survived": [num_trees_survived],
        }
    )


def cleaning_csv_df(df):
    unnamed_columns = [col for col in df.columns if "Unnamed" in col]
    df = df.drop(columns=unnamed_columns, errors="ignore")
    return df


def technical_measurement_type(value):
    text = str(value or "").strip()
    if text in ("tree_evidence", TREE_EVIDENCE_MEASUREMENT):
        return TREE_EVIDENCE_MEASUREMENT
    if text in ("tree_measurement_auto", LARGE_TREE_MEASUREMENT):
        return LARGE_TREE_MEASUREMENT
    if text:
        return text
    return TREE_EVIDENCE_MEASUREMENT


def ensure_measurement_type_column(df, column="measurement_type"):
    if column in df.columns:
        df[column] = df[column].apply(technical_measurement_type)
        return df
    return df.assign(**{column: TREE_EVIDENCE_MEASUREMENT})
