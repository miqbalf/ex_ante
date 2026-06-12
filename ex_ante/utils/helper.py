import os
import re

import numpy as np
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


def compute_num_trees_adjusted(df):
    """
    Standing tree count per row, aligned with harvest / remnant / replant flow.

    Priority:
    1. Carbon stock present (tco2_per_tree > 0, total > 0): invert from tCO2e — used
       through growth, harvest weighing, and remnant rows with retained carbon.
    2. Delayed-growth / pre-carbon years (tco2_per_tree == 0): num_trees * proportion
       when proportion is set (mortality/thinning still applies).
    3. Replant rows (new cycle init, trees but no carbon yet): num_trees already
       reflects planting space after subtracting retained trees.
    4. Remnant rows without carbon this year: num_trees (set to trees_retained).
    5. Fallback: num_trees_retained from harvest pass.
    """
    mask = df["total_csu_tCO2e_species"].notna()
    num_trees = df["num_trees"].fillna(0).to_numpy(dtype=float)
    if "proportion_per_trees" in df.columns:
        proportion = df["proportion_per_trees"].fillna(0).to_numpy(dtype=float)
    else:
        proportion = np.zeros(len(df), dtype=float)
    tco2_per_tree = df["tco2_per_tree"].fillna(0).to_numpy(dtype=float)
    total = df["total_csu_tCO2e_species"].fillna(0).to_numpy(dtype=float)

    if "num_trees_retained" in df.columns:
        retained = df["num_trees_retained"].fillna(0).to_numpy(dtype=float)
    else:
        retained = np.zeros(len(df), dtype=float)

    if "remnant_trees" in df.columns:
        remnant = df["remnant_trees"].fillna(False).to_numpy(dtype=bool)
    else:
        remnant = np.zeros(len(df), dtype=bool)

    from_stock = np.round(total / np.where(tco2_per_tree != 0, tco2_per_tree, np.nan))
    from_stock = np.where(np.isfinite(from_stock), from_stock, 0)

    proportion_standing = np.round(num_trees * proportion)
    proportion_standing = np.where(np.isfinite(proportion_standing), proportion_standing, 0)

    has_carbon_stock = (tco2_per_tree > 0) & (total > 0)
    has_proportion_standing = (tco2_per_tree == 0) & (proportion > 0) & (num_trees > 0)
    has_replant_init = (tco2_per_tree == 0) & (proportion == 0) & (num_trees > 0)
    remnant_standing = remnant & (num_trees > 0) & ~has_carbon_stock
    has_retained = retained > 0

    adjusted = np.where(
        has_carbon_stock,
        from_stock,
        np.where(
            has_proportion_standing,
            proportion_standing,
            np.where(
                has_replant_init | remnant_standing,
                num_trees,
                np.where(has_retained, retained, 0),
            ),
        ),
    )
    adjusted = np.where(np.isfinite(adjusted), adjusted, 0).astype(int)

    return pd.Series(np.where(mask.to_numpy(), adjusted, np.nan), index=df.index)


def ensure_measurement_type_column(df, column="measurement_type"):
    if column in df.columns:
        df[column] = df[column].apply(technical_measurement_type)
        return df
    return df.assign(**{column: TREE_EVIDENCE_MEASUREMENT})


def _first_growth_year(group):
    """First rotation year with measurable growth (skip leading zeros in source)."""
    dbh = group["DBH"].astype(float).fillna(0)
    height = group["Height"].astype(float).fillna(0)
    active = group[(dbh > 0) | (height > 0)]
    if not active.empty:
        return int(active["year"].min())
    return int(group["year"].min())


def apply_delayed_growth_to_growth_df(growth_df, delay_years, species_col):
    """
    Rewrite growth input: exactly delay_years rows at the start are zero, then the
    original growth curve (from its first non-zero year) is placed without repeating
    the source's leading zero years.

    Example delay=2, original years 1-30 (years 1-3 are DBH=0, growth from year 4):
    output years 1-2 = 0; year 3 = original year 4; ...; tail extends by delay_years.
    """
    if delay_years <= 0 or growth_df is None or growth_df.empty:
        return growth_df

    if species_col not in growth_df.columns:
        raise ValueError(f"Species column '{species_col}' not found in growth dataframe.")
    if "year" not in growth_df.columns:
        raise ValueError("Growth dataframe must include a 'year' column.")

    parts = []
    for species, group in growth_df.groupby(species_col, sort=False):
        group = group.sort_values("year").copy()
        max_year = int(group["year"].max())
        first_growth_year = _first_growth_year(group)
        year_offset = first_growth_year - 1
        target_max_year = delay_years + (max_year - first_growth_year + 1)
        source_by_year = group.set_index("year")

        rows = []
        for year in range(1, target_max_year + 1):
            row = {species_col: species, "year": year}
            if year <= delay_years:
                row["DBH"] = 0.0
                row["Height"] = 0.0
            else:
                source_year = (year - delay_years) + year_offset
                if source_year in source_by_year.index:
                    source = source_by_year.loc[source_year]
                    row["DBH"] = float(source["DBH"])
                    row["Height"] = float(source["Height"])
                elif source_year > max_year:
                    source = source_by_year.loc[max_year]
                    row["DBH"] = float(source["DBH"])
                    row["Height"] = float(source["Height"])
                else:
                    row["DBH"] = 0.0
                    row["Height"] = 0.0
            for column in group.columns:
                if column not in row:
                    row[column] = group.iloc[0][column]
            rows.append(row)

        parts.append(pd.DataFrame(rows))

    delayed = pd.concat(parts, ignore_index=True)
    return delayed.sort_values([species_col, "year"]).reset_index(drop=True)


def export_growth_selected_csv(growth_df, csv_path, species_col):
    """Save selected growth in long form and CoreDB-style wide form."""
    growth_df = growth_df.copy()
    growth_df.to_csv(csv_path, index=False)

    if (
        growth_df is None
        or growth_df.empty
        or species_col not in growth_df.columns
        or "year" not in growth_df.columns
    ):
        return growth_df

    wide_rows = []
    for species, group in growth_df.groupby(species_col, sort=False):
        group = group.sort_values("year")
        for measure in ("DBH", "Height"):
            if measure not in group.columns:
                continue
            row = {species_col: species, "DBH/Height": measure}
            for _, measure_row in group.iterrows():
                row[f"year {int(measure_row['year'])}"] = measure_row[measure]
            wide_rows.append(row)

    if wide_rows:
        wide_path = csv_path.replace(".csv", "_wide.csv")
        pd.DataFrame(wide_rows).to_csv(wide_path, index=False)
    return growth_df
