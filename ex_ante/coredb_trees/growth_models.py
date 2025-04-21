import os
import re
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from ..utils.gsheet_downloader import request_read_gsheet
from ..utils.helper import change_year_int

# Load the .env file
load_dotenv()


class GrowthModelSpecies:
    link_growth: str  # link to the CSV of growth model (coredb forester gsheet db)
    download_csv: bool
    growth_csv: str
    name_column_species: str  # gotta need to define this, since most of the time forester changes the column name of the csv and give you headache of error after

    # Access the environment variables in the default argument
    def __init__(
        self,
        link_growth: str = os.getenv("link_growth_model_data_csv"),
        download_csv: bool = False,  # check if you want to download first the csv or use existing one from the path below
        growth_csv: str = os.getenv(
            "path_local_growth_model_data_csv"
        ),  # we should pay attention on which file we use for the growth
        name_column_species: str = "Tree Species(+origin of allom. formula)",
    ):
        self.link_growth = link_growth
        self.download_csv = download_csv
        self.growth_csv = growth_csv
        self.name_column_species = name_column_species

    def df_30y_growth(self) -> pd.DataFrame:
        df_30y_growth = request_read_gsheet(self.link_growth)
        return df_30y_growth

    def restructure_growth_data(self) -> pd.DataFrame:
        if self.download_csv:
            df_30y_growth = self.df_30y_growth()
            df_30y_cleaned = df_30y_growth[
                [
                    column
                    for column in df_30y_growth.columns.to_list()
                    if "Unnamed" not in column
                ]
            ]
            df_30y_cleaned = df_30y_cleaned.reset_index()
            current_date = datetime.now().strftime(
                "%Y-%m-%d"
            )  # it require the default value or some input has yyyy-mm-dd

            date_pattern = r"\d{4}-\d{2}-\d{2}"

            match = re.search(date_pattern, self.allometry_csv)
            if match:
                # Replace existing YYYY-MM-DD with the current date
                self.growth_csv = re.sub(date_pattern, current_date, self.growth_csv)
            else:
                self.growth_csv = f"{current_date}_{self.growth_csv}"

            print(f"Saving to CSV: {self.growth_csv}")
            df_30y_cleaned.to_csv(self.growth_csv, index=False)

        else:
            print(f"reading from the CSV: {self.growth_csv}")
            df_30y_cleaned = pd.read_csv(self.growth_csv)

        year_columns = [
            column
            for column in df_30y_cleaned.columns.to_list()
            if "year " in column and "MAI" not in column
        ]
        meltdf_growth = pd.melt(
            df_30y_cleaned,
            id_vars=[self.name_column_species, "DBH/Height"],
            value_vars=year_columns,
        )

        clean_melt_growth_df = meltdf_growth.dropna(axis="rows")

        df_growth = clean_melt_growth_df.copy()

        df_growth["year"] = df_growth["variable"].apply(lambda x: change_year_int(x))

        # restructuring the data for merging later, need to make this compatible and joined with allometry function selected
        df_growth_fix = df_growth.pivot_table(
            values="value",
            index=[self.name_column_species, "year"],
            columns=["DBH/Height"],
            aggfunc="sum",
        )
        df_growth_fix = df_growth_fix.reset_index()

        return df_growth_fix
