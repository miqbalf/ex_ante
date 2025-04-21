import pandas as pd


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


def cleaning_csv_df(df):
    unnamed_columns = [col for col in df.columns if "Unnamed" in col]
    df = df.drop(columns=unnamed_columns, errors="ignore")
    return df
