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
