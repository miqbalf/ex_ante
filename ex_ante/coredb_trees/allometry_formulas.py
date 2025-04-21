import math
import os
import re
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from ..utils.gsheet_downloader import request_read_gsheet

# Load the .env file
load_dotenv()


class AllometryFormulaDB:
    link_allometry: str
    name_column_species: str

    def __init__(
        self,
        link_allometry: str = os.getenv("link_allometry_csv_gsheet"),
        download_csv: bool = False,  # check if you want to download first the csv or use existing one from the path below
        allometry_csv: str = os.getenv(
            "path_local_allometry_data_csv"
        ),  # we should pay attention on which file we use for the allometry
        name_column_species: str = "Lat. Name",  # gotta need to define this, since most of the time forester changes the column name of the csv and give you headache of error after
        allo_formula_column_name="TTB formula, tdm",
    ):
        self.link_allometry = link_allometry
        self.download_csv = download_csv
        self.allometry_csv = allometry_csv
        self.name_column_species = name_column_species
        self.allo_formula_column_name = allo_formula_column_name

    def restructuring_allometric_data(self) -> pd.DataFrame:
        if self.download_csv:
            df_allometric = request_read_gsheet(self.link_allometry)
            df_allometric = df_allometric.reset_index()
            # Get current date in YYYY-MM-DD format
            current_date = datetime.now().strftime(
                "%Y-%m-%d"
            )  # it require the default value or some input has yyyy-mm-dd

            date_pattern = r"\d{4}-\d{2}-\d{2}"

            match = re.search(date_pattern, self.allometry_csv)
            if match:
                # Replace existing YYYY-MM-DD with the current date
                self.allometry_csv = re.sub(
                    date_pattern, current_date, self.allometry_csv
                )
            else:
                self.allometry_csv = f"{current_date}_{self.allometry_csv}"

            print(f"Saving to CSV: {self.allometry_csv}")

            # Save DataFrame to CSV
            df_allometric.to_csv(self.allometry_csv, index=False)

        else:
            print(f"reading from the CSV: {self.allometry_csv}")
            df_allometric = pd.read_csv(self.allometry_csv)

        # these columnns still hard-coded, if someone someday change the name of these column before you download the new csv, we need to change it.
        # since probably we will use database, let's ignore for the moment
        allometric_column_filter = df_allometric[
            [
                self.name_column_species,
                "Country of Use",
                "Allometric Formula, Type",
                self.allo_formula_column_name,
                "WD variable",
            ]
        ]
        allometric_column_filter = allometric_column_filter.dropna()

        return allometric_column_filter
