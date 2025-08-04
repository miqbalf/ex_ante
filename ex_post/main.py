import json
import os
import copy

import exante  # needed to get the module_path
import geopandas as gpd
import numpy as np
import pandas as pd
from typing import List, Optional
from dateutil.relativedelta import relativedelta
from ex_ante.population_tco2.main import num_tco_years
from ex_ante.utils.calc_formula_string import calc_biomass_formula

# from scipy.optimize import curve_fit, fsolve
# import matplotlib.pyplot as plt
from ex_ante.utils.growth_modeller import remodel_growth
from ex_ante.utils.helper import cleaning_csv_df
from ex_post.utils import species_reverse_coredb
from exante import ExAnteCalc

from .utils import (
    find_nearest_plot,
    species_match_coredb_treeocloud,
    species_reverse_coredb,
)

module_path = os.path.dirname(exante.__file__)

def process_scenarios(old_scenario_exante_toedit, concat_df, new_species_to_be_added_zone, adding_prev_mortality_rate=0, 
            override_mortality_replanting = 40, update_species_name={}):
    print(f"check argument here def process_scenarios(old_scenario_exante_toedit, concat_df, new_species_to_be_added_zone = {new_species_to_be_added_zone}, adding_prev_mortality_rate={adding_prev_mortality_rate}, \n "
            f"override_mortality_replanting = {override_mortality_replanting}, update_species_name={update_species_name})")

    """
    Processes and updates planting scenarios based on old data and new species.

    Key Logic:
    1.  Defaults to old scenario values for all parameters.
    2.  For new species, it first tries to find a template from an existing species
        with a key value species first (not to use other species, but the one setup in expost e.g change allometry (tree species code)).
    3.  If no keyword match is found, it falls back to using the first available in between replanting vs non-replanting, and after that existing
        species in that zone as a template.
    4.  Allows for specific values like mortality rate to be overridden.
    """
    all_scenario = {}
    
    # These are the only keys that will be processed as planting zones
    VALID_ZONES = ["production_zone", "protected_zone"]

    for is_replanting, zone_scenario in old_scenario_exante_toedit.items():
        updated_scenario = {}

        # 1. Process existing species from the old scenario
        for zone in VALID_ZONES:
            if zone in zone_scenario:
                species_scenario_map = zone_scenario[zone]
                updated_scenario[zone] = {}

                for species, scenario in species_scenario_map.items():
                    # Check if the species is still considered valid
                    if species in concat_df[concat_df["zone"] == zone]["Lat. Name"].to_list():
                        # Copy all parameters from the old scenario by default
                        updated_single_scenario = scenario.copy()
                        # Override any specific values
                        updated_single_scenario["mortality_percent"] = adding_prev_mortality_rate
                        updated_scenario[zone][species] = updated_single_scenario

        # 2. Add any new species with the special template-finding logic
        if new_species_to_be_added_zone:
            for zone, new_species_list in new_species_to_be_added_zone.items():
                if zone not in VALID_ZONES:
                    continue
                if zone not in updated_scenario:
                    updated_scenario[zone] = {}

                for new_species in new_species_list:
                    base_scenario = None
                    # # Extract the first word of the new species name as the keyword
                    # new_species_keyword = new_species.split(' ')[0]

                    # # A. Try to find a template in the current zone using the keyword
                    # if zone in zone_scenario:
                    #     for existing_species, existing_scenario in zone_scenario[zone].items():
                    #         if existing_species.startswith(new_species_keyword):
                    #             base_scenario = existing_scenario.copy()
                    #             break  # Found the best template, stop searching


                    # A. the same exact name
                    if zone in zone_scenario:
                        for existing_species, existing_scenario in zone_scenario[zone].items():
                            if existing_species == new_species:
                                base_scenario = existing_scenario.copy()
                                break  # Found the best template, stop searching

                    # A1 . Try template using the existing dictionary key, value (if there is changing code of species code) within the same species due to e.g allometry formula, or growth model changes
                    if zone in zone_scenario:
                        for existing_species, existing_scenario in zone_scenario[zone].items():
                            if update_species_name.get(existing_species) == new_species:
                                base_scenario = existing_scenario.copy()
                                break

                    # B . Before we do iter, that picking on the first item of base_scenario, let use all the possibility in the replanting and non_replanting
                    if base_scenario is None:
                        for other_zone_scenario in old_scenario_exante_toedit.values():
                            if zone in other_zone_scenario:
                                for existing_species, existing_scenario in other_zone_scenario[zone].items():
                                    if (existing_species == new_species or 
                                        update_species_name.get(existing_species) == new_species):
                                        base_scenario = existing_scenario.copy()
                                        break
                            if base_scenario:
                                break


                    # # B1. If no match at all, fall back to the first available species
                    if base_scenario is None and zone in zone_scenario and zone_scenario[zone]:
                        base_scenario = next(iter(zone_scenario[zone].values()), {}).copy()
                    
                    # C. If still no template, use an empty dictionary
                    if base_scenario is None:
                        base_scenario = {}
                    
                    #####################################
                    # Create the new scenario, which inherits all keys from the template <--- we can manouver this section later
                    new_scenario = base_scenario
                    if is_replanting == 'non_replanting':
                        # Override specific values
                        new_scenario["mortality_percent"] = adding_prev_mortality_rate

                    else: # create logic, separate the mort. existing previous trees, and newly replanting trees
                        new_scenario['mortality_percent'] = override_mortality_replanting
                    
                    # Set harvesting_year, using the template's value or a default
                    if zone == "protected_zone":
                        new_scenario["harvesting_year"] = base_scenario.get("harvesting_year", 30)
                    elif zone == "production_zone":
                        new_scenario["harvesting_year"] = base_scenario.get("harvesting_year", 10)

                    ######################################
                    
                    updated_scenario[zone][new_species] = new_scenario

        all_scenario[is_replanting] = updated_scenario
    
    return all_scenario

def process_plot_data(
    df: pd.DataFrame,
    year_max: Optional[int] = None,
    year_filter: Optional[List[int]] = None,
    expost_agg: bool = False,
    columns_to_remove: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Aggregates tree count data for plots based on specified parameters.

    This function processes a DataFrame by filtering, grouping, and aggregating
    tree counts. It can perform a standard grouping by 'year_start' or a
    cumulative aggregation up to a specified 'year_max' or the latest year
    in the dataset.

    Args:
        df (pd.DataFrame): Input DataFrame containing plot data. Must include
            'Plot_ID', 'year_start', and columns ending in '_num_trees'.
        year_max (Optional[int]): If provided, all data is aggregated into a
            single year specified by this value. Defaults to None.
        year_filter (Optional[List[int]]): A list of years to include from the
            'year_start' column. If provided, data is filtered before any
            aggregation. Defaults to None.
        expost_agg (bool): If True and year_max is not set, performs a
            cumulative aggregation up to the latest 'year_start' found in the
            data. Defaults to False.
        columns_to_remove (Optional[List[str]]): A list of column names to
            drop before processing. Defaults to None.

    Returns:
        pd.DataFrame: A processed DataFrame with aggregated tree counts and
            a 'num_trees_total' column.

    Raises:
        ValueError: If no columns ending in '_num_trees' are found.
    """
    # 1. Create a working copy and perform initial cleaning
    df_processed = df.copy()

    if columns_to_remove:
        # Gracefully handle columns that may not exist in the DataFrame
        cols_to_drop = [col for col in columns_to_remove if col in df_processed.columns]
        df_processed.drop(columns=cols_to_drop, inplace=True)

    if year_filter:
        df_processed = df_processed[df_processed['year_start'].isin(year_filter)]
        if df_processed.empty:
            return pd.DataFrame() # Return empty DataFrame if filter removes all data

    # 2. Identify columns for aggregation
    tree_cols = [col for col in df_processed.columns if col.endswith("_num_trees")]
    if not tree_cols:
        raise ValueError("No columns found ending with '_num_trees'.")

    # 3. Determine grouping strategy and prepare the DataFrame
    is_cumulative_agg = year_max is not None or expost_agg
    group_by_cols = ['Plot_ID']

    if is_cumulative_agg:
        year_index_col = 'year_max'
        # Set the target year for the cumulative aggregation
        agg_year = year_max if year_max is not None else df_processed['year_start'].max()
        df_processed[year_index_col] = agg_year
    else:
        year_index_col = 'year_start'
    
    group_by_cols.append(year_index_col)

    # 4. Aggregate the tree count data
    agg_spec = {col: 'sum' for col in tree_cols}
    aggregated_trees = df_processed.groupby(group_by_cols, as_index=False).agg(agg_spec)

    # 5. Prepare the metadata by getting unique rows based on group keys
    metadata_cols = df_processed.columns.drop(tree_cols).tolist()
    # When doing a cumulative aggregation, 'year_start' is no longer a relevant metadata key
    if is_cumulative_agg and 'year_start' in metadata_cols:
        metadata_cols.remove('year_start')
    
    metadata_df = df_processed[metadata_cols].drop_duplicates(subset=group_by_cols)

    # 6. Merge aggregated tree counts back with their metadata
    result_df = pd.merge(metadata_df, aggregated_trees, on=group_by_cols, how='left')

    # 7. Final calculations and cleanup
    result_df[tree_cols] = result_df[tree_cols].fillna(0)
    result_df['num_trees_total'] = result_df[tree_cols].sum(axis=1)

    return result_df


class ExPostAnalysis:
    def __init__(
        self,
        config: dict,
        manual_tco2_calc=False,
        plot_id_suffix=False,
        distance_0plot_threshold=10,
        override_num_trees_0=True,
    ):
        self.config = config
        self.manual_tco2_calc = manual_tco2_calc
        self.plot_id_suffix = plot_id_suffix
        self.csv_file_expost = config.get("csv_file_expost", None)
        self.csv_file_exante = config.get("csv_file_exante", None)
        self.plot_file_geojson = config.get("plot_file_geojson", None)
        self.species_json = config.get("species_json", None)
        self.TPP_name = config.get("TPP_name", None)
        self.project_name = config.get("project_name", None)
        self.is_zone_check = config.get("filter_by_zone", None)
        self.is_plot_check = config.get("filter_by_plot", None)
        self.filtered_by_datateam = config.get("filtered_by_datateam", None)

        self.cycle_monitoring = self.config.get("monitoring_year", None)
        self.distance_0plot_threshold = distance_0plot_threshold
        self.override_num_trees_0 = override_num_trees_0

        # create empty df instance to update later in below method
        self.exante_num_trees_yrs = pd.DataFrame()
        self.exante_tco2e_yrs = pd.DataFrame()
        self.validated_df = pd.DataFrame()
        self.validated_csu_species = pd.DataFrame()

        self.prev_exante_config = config.get('prev_exante_config',None)

        with open(self.prev_exante_config, 'r') as ex_ante_config:
            self.prev_ex_ante_main_json = json.load(ex_ante_config)

        self.prev_ex_ante_planting_year = self.prev_ex_ante_main_json.get('planting_year',None)

        self.update_species_name = {} # for replanting_plan later

        self.gdf_plot = gpd.read_file(self.config["plot_file_geojson"])
        self.gdf_plot_cleaned = self.gdf_plot[
            [
                "Plot_ID",
                "Plot_Area",
                "Plot_Name",
                "managementUnit",
                "plantingStartDate",
                "plantingEndDate",
            ]
        ]
        if self.gdf_plot_cleaned["Plot_ID"].duplicated().any():
            raise ValueError("Must not have duplicate Plot_ID")

        self.field_joins = {"Plot_ID": "Plot_ID_expost"}
        if plot_id_suffix:
            self.field_joins = {"Plot_ID": "plot_backup"}  # in gdf plot vs in meas
        else:
            self.field_joins = self.field_joins

        # put the df meas as df object in init
        df_meas_csv = pd.read_csv(self.csv_file_expost)
        # remove unnamed csv (record empty column)
        df_no_unname = df_meas_csv.loc[:, ~df_meas_csv.columns.str.contains("^Unnamed")]

        if self.filtered_by_datateam == True:
            df_expost_all = (
                df_no_unname.copy()
            )  # recent template from gowtham readjust to old template
            df_expost_all["check_result_data_species_check_manual"] = df_expost_all[
                "tree_species_code"
            ]
            df_expost_all["plot_id_plot_check"] = df_expost_all["plot_id"]
            df_expost_all["id"] = df_expost_all["measurement_uuid"]
        else:
            df_expost_all = df_no_unname.copy()  # old template from gowtham

        df_expost_all = self.get_nearbyplot_distance(
            self.gdf_plot, df_expost_all, self.distance_0plot_threshold
        )

        # separating the expost exante plot, with allowing NaN
        df_expost_all = df_expost_all.rename(
            columns={"plot_id_plot_check": "Plot_ID_expost"}
        )  # this is to ensure that the Plot_ID from gdf plot, if there is suffix, will be replaced by the meas data (with info suffix)

        list_columns_df_meas = df_expost_all.columns

        sorted_columns_df_meas = ["Plot_ID_expost"] + [
            f
            for f in list_columns_df_meas
            if f != "Plot_ID_expost" and f != "Plot_ID" and f != "plot_id"
        ]

        df_meas = df_expost_all[sorted_columns_df_meas]

        self.df_meas = df_meas

        meas_validated = self.get_df_validated(
            is_zone_check=self.is_zone_check, is_plot_check=self.is_plot_check
        )

        ## not needed anymore since we already override in all df_meas above
        # gdf_meas_validated = self.get_nearbyplot_distance(self.gdf_plot, meas_validated, self.distance_0plot_threshold)
        # self.meas_validated = gdf_meas_validated

        self.meas_validated = meas_validated

        df_meas_plot = pd.merge(
            self.gdf_plot_cleaned,
            self.meas_validated,
            left_on="Plot_ID",
            right_on=self.field_joins["Plot_ID"],
            how="right",
        )

        # this is how we will measure growth of month ######################################
        # now the column of these following mention is a must in geojson
        df_meas_plot["plantingStartDate"] = pd.to_datetime(
            df_meas_plot["plantingStartDate"]
        )
        df_meas_plot["plantingEndDate"] = pd.to_datetime(
            df_meas_plot["plantingEndDate"]
        )

        # display(df_meas_plot)
        # check if there is any NaN in plantingStartDate
        print(
            "check if there is any NaN in plantingStartDate ",
            df_meas_plot["plantingStartDate"].isna().any(),
        )  # Check for NaN values
        print(
            "check if there is any NaN in plantingEndDate ",
            df_meas_plot["plantingEndDate"].isna().any(),
        )  # Check for NaT values

        # print(df_meas_plot['plantingStartDate'].dtype)  # Ensure this is datetime64[ns, UTC]
        # print(df_meas_plot['plantingEndDate'].dtype)  # Ensure this is datetime64[ns, UTC]

        # Convert the 'monitoring_date' column to datetime object
        df_meas_plot["monitoring_date"] = pd.to_datetime(
            df_meas_plot["monitoring_date"], format="ISO8601"
        )

        # Format the datetime to include microseconds
        df_meas_plot["monitoring_date"] = df_meas_plot["monitoring_date"].dt.strftime(
            "%Y-%m-%d %H:%M:%S.%f%z"
        )

        # Assign a dummy timezone (UTC) to avoid TypeError: Cannot compare tz-naive and tz-aware timestamps
        try:
            df_meas_plot["plantingStartDate"] = df_meas_plot[
                "plantingStartDate"
            ].dt.tz_localize("UTC")
            df_meas_plot["plantingEndDate"] = df_meas_plot[
                "plantingEndDate"
            ].dt.tz_localize("UTC")
        except Exception as e:
            print(f"hotfix - ignore error because {e} for not using tz_localize")
            print("pass")
            pass

        # Ensure midpoint planting date is calculated correctly
        df_meas_plot["avg_planting_date"] = df_meas_plot["plantingStartDate"] + (
            (df_meas_plot["plantingEndDate"] - df_meas_plot["plantingStartDate"]) / 2
        )

        # Convert the 'avg_planting_date' column to datetime object
        df_meas_plot["avg_planting_date"] = pd.to_datetime(
            df_meas_plot["avg_planting_date"], format="ISO8601"
        )

        # Format the datetime to include microseconds
        df_meas_plot["avg_planting_date"] = df_meas_plot[
            "avg_planting_date"
        ].dt.strftime("%Y-%m-%d %H:%M:%S.%f%z")

        # Step 3: Check for NaT or missing values in the key date columns
        if df_meas_plot["monitoring_date"].isna().any():
            raise ValueError(
                "monitoring_date contains NaT (missing values). Please handle these before proceeding."
            )
        if df_meas_plot["avg_planting_date"].isna().any():
            raise ValueError(
                "avg_planting_date contains NaT (missing values). Please handle these before proceeding."
            )

        # Ensure the columns are datetimes
        df_meas_plot["monitoring_date"] = pd.to_datetime(
            df_meas_plot["monitoring_date"], errors="coerce"
        )
        df_meas_plot["avg_planting_date"] = pd.to_datetime(
            df_meas_plot["avg_planting_date"], errors="coerce"
        )

        # Check for missing values and handle them
        if (
            df_meas_plot["monitoring_date"].isnull().any()
            or df_meas_plot["avg_planting_date"].isnull().any()
        ):
            raise ValueError(
                "monitoring_date or avg_planting_date contains NaT (missing values). Please handle these before proceeding."
            )

        # Calculate the age in months using relateivedelta
        df_meas_plot["age_month"] = df_meas_plot.apply(
            lambda row: (
                relativedelta(row["monitoring_date"], row["avg_planting_date"]).years
                * 12
            )
            + relativedelta(row["monitoring_date"], row["avg_planting_date"]).months,
            axis=1,
        )

        # # Step 4: Calculate the age in months, ignoring time
        # df_meas_plot['age_month'] = df_meas_plot.apply(
        #     lambda row: (relativedelta(row['monitoring_date'], row['avg_planting_date']).years * 12) +
        #                 relativedelta(row['monitoring_date'], row['avg_planting_date']).months,
        #     axis=1
        # )

        list_columns_df_meas_plot = df_meas_plot.columns

        sorted_columns_df_meas_plot = ["Plot_ID_expost"] + [
            f
            for f in list_columns_df_meas_plot
            if f != "Plot_ID_expost" and f != "Plot_ID" and f != "plot_id"
        ]

        df_meas_plot = df_meas_plot[sorted_columns_df_meas_plot]

        self.df_meas_plot = df_meas_plot  # this the valid plot only for expost

        # for motality_analysis
        self.df_expost_filtered_mort = None
        self.mortality_csu_df = None  # need to be defined later in the notebook by assigning this, because some validation csv from data team has different column name

        ####################################################################################

    @staticmethod
    def get_nearbyplot_distance(gdf_plot, df_meas, distance_0plot_threshold):
        # example in the KPLPB if we want to approve the zone_check_failed, or plot_check_failed, we will need to fill the 0 plot_id,
        # not yet discussed with data team regarding the threshold, but we will include them if nearby based on distance

        # Create a GeoDataFrame for the validated meas, later with the other meas
        gdf_meas = gpd.GeoDataFrame(
            # meas_validated,
            df_meas,
            # geometry=gpd.points_from_xy(meas_validated['longitude'], meas_validated['latitude']),
            geometry=gpd.points_from_xy(df_meas["longitude"], df_meas["latitude"]),
            crs="EPSG:4326",  # WGS84 Latitude/Longitude
        )
        gdf_meas = gdf_meas.to_crs("EPSG:3857")  # Convert to a metric CRS (meters)

        # Ensure CRS is appropriate for distance calculation (use a projected CRS)
        gdf_plot = gdf_plot.to_crs("EPSG:3857")

        # we will only analyze if the plot_id from the result validation scoring is 0
        gdf_meas_0 = gdf_meas.copy()
        gdf_meas_0 = gdf_meas_0[(gdf_meas_0["plot_id_plot_check"] == 0)]

        # filter not-na, to be concat later
        gdf_meas_not_na = gdf_meas.copy()
        gdf_meas_not_na = gdf_meas_not_na[~(gdf_meas_not_na["plot_id_plot_check"] == 0)]

        # Apply the function to get the nearest_plot_id
        gdf_meas_0["nearest_plot_id"], gdf_meas_0["distance_to_nearest_plot"] = (
            find_nearest_plot("Plot_ID", gdf_plot, gdf_meas_0)
        )

        # override the plot_id_plot_check defined based on nearest_plot_id based on Plot Data (geojson)
        gdf_meas_0 = (
            gdf_meas_0.copy()
        )  # to avoid the annoying recommendation slicer in jup.notebook
        gdf_meas_0["plot_id_plot_check"] = gdf_meas_0["nearest_plot_id"]
        # gdf_meas_0['plot_id'] = gdf_meas_0['plot_id_plot_check']

        ################## APPLYING DISTANCE HERE!!!! for PLOT 0
        # apply the filter distance based on arguments
        ################## APPLYING DISTANCE HERE!!!!
        gdf_meas_0 = (
            gdf_meas_0.copy()
        )  # to avoid the annoying recommendation slicer in jup.notebook
        gdf_meas_0 = gdf_meas_0[
            gdf_meas_0["distance_to_nearest_plot"] <= distance_0plot_threshold
        ]

        gdf_meas_final = pd.concat([gdf_meas_0, gdf_meas_not_na]).reset_index(drop=True)
        return gdf_meas_final

    def get_df_validated(self, is_zone_check="", is_plot_check="") -> pd.DataFrame:
        if is_zone_check == "":
            is_zone_check = self.is_zone_check

        if is_plot_check == "":
            is_plot_check = self.is_plot_check

        if self.filtered_by_datateam == True:
            df_filter_no_tree = self.df_meas.copy()
            df_filter_no_tree = df_filter_no_tree[
                df_filter_no_tree["status"] == "approved"
            ]
            # display(df_filter_no_tree)
        else:
            df_no_unnamed = self.df_meas.copy()
            df_filter_no_tree = df_no_unnamed[
                ~(df_no_unnamed["check_result_data_species_check_manual"] == "no_tree")
            ]

            df_filter_no_tree = df_filter_no_tree[
                ~(df_filter_no_tree["check_result_data_species_check_manual"] == "Dead")
            ]
            df_filter_no_tree = df_filter_no_tree[
                ~(
                    df_filter_no_tree["check_result_data_species_check_manual"]
                    == "dead_tree"
                )
            ]
            df_filter_no_tree = df_filter_no_tree[
                ~(
                    df_filter_no_tree["check_result_data_species_check_manual"]
                    == "unvalidated"
                )
            ]
            df_filter_no_tree = df_filter_no_tree[
                ~(
                    df_filter_no_tree["check_result_data_species_check_manual"]
                    == "no_image_no_species"
                )
            ]
            df_filter_no_tree = df_filter_no_tree[
                ~(
                    df_filter_no_tree["check_result_data_species_check_manual"]
                    == "syzygium_aromaticum"
                )
            ]

            df_filter_no_tree = df_filter_no_tree[
                ~(df_filter_no_tree["check_result_data_species_check_manual"].isna())
            ]

            if is_zone_check and is_plot_check:
                df_filter_no_tree = df_filter_no_tree[
                    (df_filter_no_tree["check_result_data_zones_check"] == "in_go_zone")
                    & ~(
                        df_filter_no_tree["check_result_data_plot_check"].isin(
                            ["measurement_outside_plot", "measurement_not_on_plot"]
                        )
                    )
                ]  # since new script result now has different naming (measurement_outside_plot -> measurement_not_on_plot)

            elif is_plot_check:
                df_filter_no_tree = df_filter_no_tree[
                    ~(
                        df_filter_no_tree["check_result_data_plot_check"].isin(
                            ["measurement_outside_plot", "measurement_not_on_plot"]
                        )
                    )
                ]

            elif is_plot_check:
                df_filter_no_tree = df_filter_no_tree[
                    (df_filter_no_tree["check_result_data_zones_check"] == "in_go_zone")
                ]

        self.validated_df = df_filter_no_tree

        return df_filter_no_tree

    def read_plot(self) -> gpd.GeoDataFrame:
        df_plot = gpd.read_file(self.plot_file_geojson)
        df_plot["plotZone"] = df_plot.apply(
            lambda x: (
                "protected_zone"
                if x["plotZone"] == "conservation_zone"
                else (
                    "production_zone"
                    if x["plotZone"] == "production_zone"
                    else x["plotZone"]
                )
            ),
            axis=1,
        )
        return df_plot

    def comparing_exante_expost(
        self,
        is_zone_check="",
        is_plot_check="",
        cycle_monitoring="",
        formulas_allometry="",
        mortality_csu_df=None,
    ) -> pd.DataFrame:
        if is_zone_check == "":
            is_zone_check = self.is_zone_check

        if is_plot_check == "":
            is_plot_check = self.is_plot_check

        if cycle_monitoring == "":
            cycle_monitoring = self.config["monitoring_year"]

        if formulas_allometry == "":
            formulas_allometry = self.config["formulas_allometry"]

        selected_formulas_ex_ante = pd.read_csv(formulas_allometry)

        selected_formulas_ex_ante = selected_formulas_ex_ante[
            ["Lat. Name", "TTB formula, tdm", "WD variable"]
        ]  # hardcoded, please comeback again later if there is error here

        selected_formulas_ex_ante = selected_formulas_ex_ante.drop_duplicates()

        # expost_validated = self.get_df_validated(is_zone_check = is_zone_check, is_plot_check = is_plot_check)
        expost_validated = self.df_meas_plot

        # important!! to include no_plot if we still want to add them in the validated measurement
        expost_validated["Plot_ID_expost"] = expost_validated["Plot_ID_expost"].fillna(
            0
        )

        # count the number of measurements id, --> Nr Trees Expost
        expost_validated_count = (
            expost_validated.groupby(
                [
                    "Plot_ID_expost",
                    "check_result_data_species_check_manual",
                    "is_replanting",
                    'year_start'
                ]
            )["id"]
            .count()
            .reset_index()
        )

        expost_validated_count = expost_validated_count.rename(
            columns={"id": "Nr. Trees Ex-Post"}
        )

        if (
            cycle_monitoring == 0
        ):  # cycle monitoring here is year monitoring, it should be called as cycle 0 for year 0 or tree evidence
            expost_validated_count = expost_validated_count

        elif cycle_monitoring > 0:

            # Sum the tCO2e values for the same grouping
            if self.manual_tco2_calc != True:
                expost_validated = expost_validated

            else:
                expost_validated["species_coredb"] = expost_validated.apply(
                    lambda x: species_reverse_coredb(
                        x["check_result_data_species_check_manual"],
                        self.config["species_json"],
                    ),
                    axis=1,
                )

                expost_validated_merged = pd.merge(
                    expost_validated,
                    selected_formulas_ex_ante,
                    left_on="species_coredb",
                    right_on="Lat. Name",
                    how="left",
                )

                expost_validated_merged["dbh_cm"] = (
                    expost_validated_merged["treeDBHmm"].astype(float) / 10
                )

                # calc_biomass_formula(ttb_formula, wd, dbh, height):
                expost_validated_merged["ttb_manual"] = expost_validated_merged.apply(
                    lambda x: calc_biomass_formula(
                        x["TTB formula, tdm"],  # all this still in hard coded still
                        x["WD variable"],
                        x["dbh_cm"],
                        "",
                    ),
                    axis=1,
                )  # ignore height at the moment we put ''

                expost_validated_merged["tco2e_manual"] = (
                    expost_validated_merged["ttb_manual"] * 0.47 * (44 / 12)
                )

                # rename and changes / override with manual calc
                expost_validated_merged["co2_tree_captured_tonnes_archieve"] = (
                    expost_validated_merged["co2_tree_captured_tonnes"]
                )
                expost_validated_merged["co2_tree_captured_tonnes"] = (
                    expost_validated_merged["tco2e_manual"]
                )
                expost_validated = expost_validated_merged

            # get the information of tree evidence vs large tree
            if "measurement_type" in expost_validated.columns:
                expost_validated_count_meas = pd.pivot_table(
                    data=expost_validated,
                    index=[
                        "Plot_ID_expost",
                        "check_result_data_species_check_manual",
                        "is_replanting",
                        'year_start'
                    ],
                    columns=["measurement_type"],
                    values=["id"],
                    aggfunc="count",
                ).reset_index()

                expost_validated_count_meas.columns = [
                    (
                        "_".join([str(sub_col) for sub_col in col if sub_col != ""])
                        .strip()
                        .rstrip("_")
                        if isinstance(col, tuple)
                        else col
                    )
                    for col in expost_validated_count_meas.columns
                ]

                expost_validated_count = pd.merge(
                    expost_validated_count,
                    expost_validated_count_meas,
                    on=[
                        "Plot_ID_expost",
                        "check_result_data_species_check_manual",
                        "is_replanting",
                        'year_start'
                    ],
                    how="left",
                )

                expost_validated_count = expost_validated_count.rename(
                    columns={
                        "id_tree_evidence": "Nr Tree Evidence Expost",
                        "id_tree_measurement_auto": "Nr Large Tree Expost",
                    }
                )

            expost_validated_tco2e = (
                expost_validated.groupby(
                    [
                        "Plot_ID_expost",
                        "check_result_data_species_check_manual",
                        "is_replanting",
                        'year_start'
                    ]
                )["co2_tree_captured_tonnes"]
                .sum()
                .reset_index()
            )
            expost_validated_count = pd.merge(
                expost_validated_count,
                expost_validated_tco2e,
                on=[
                    "Plot_ID_expost",
                    "check_result_data_species_check_manual",
                    "is_replanting",
                    'year_start'
                ],
            )
            expost_validated_count = expost_validated_count.rename(
                columns={"co2_tree_captured_tonnes": "C02 Measured"}
            )

        self.validated_csu_species = expost_validated_count  # Plot_ID_expost

        df_plot = self.read_plot()
        df_ex_ante = pd.read_csv(self.csv_file_exante)
        distribution_seedling = self.config["formulas_allometry"].replace(
            "formulas_allometry.csv", "distribution_trees_seedling.csv"
        )

        # override_num_trees_0 = self.override_num_trees_0
        # if override_num_trees_0:
        #     mortality_csu_df = self.mortality_csu_df
        #     if mortality_csu_df is None:
        #         raise ValueError(f"you set override year 0 using mortality csu data override_num_trees_0 = {override_num_trees_0}, but no df given, please set to false or define mortality_csu_df in arg")
        #
        # else:
        #     mortality_csu_df = mortality_csu_df # equal to None if there is no override

        

        pop_tco2 = num_tco_years(
            df_ex_ante=df_ex_ante,
            distribution_seedling=distribution_seedling,
            override_num_trees_0=False,
            planting_year=self.prev_ex_ante_planting_year,
            current_gap_year=0,
            is_include_all_init_planting=True
        )  # default to False because this part using entirely ex-ante for initial comparison
        joined_pivot_tco2e_all = pop_tco2["joined_pivot_tco2e_all"]
        joined_pivot_num_trees_all = pop_tco2["joined_pivot_num_trees_all"]
        self.exante_num_trees_yrs = pop_tco2["exante_num_trees_yrs"]
        self.exante_tco2e_yrs = pop_tco2["exante_tco2e_yrs"]

        joined_pivot_num_trees_all = joined_pivot_num_trees_all.reset_index()
        # species_series = pd.Series(joined_pivot_num_trees_all.index.get_level_values('species'), index=joined_pivot_num_trees_all.index)
        joined_pivot_num_trees_all["species_treeocloud"] = joined_pivot_num_trees_all[
            "species"
        ].apply(
            lambda species_name: species_match_coredb_treeocloud(
                species_name, self.config["species_json"]
            )
        )
        joined_pivot_num_trees_all = joined_pivot_num_trees_all.set_index(
            [
                "is_replanting",
                'year_start',
                "plotZone",
                "managementUnit",
                "Plot_ID_exante",
                "species_treeocloud",
            ]
        )

        joined_pivot_tco2e_all = joined_pivot_tco2e_all.reset_index()
        # joined_pivot_tco2e_all = joined_pivot_tco2e_all.set_index(['year_start','is_replanting', 'plotZone', 'managementUnit', 'Plot_ID_exante','species_treeocloud'])
        # species_series = pd.Series(joined_pivot_tco2e_all.index.get_level_values('species'), index=joined_pivot_num_trees_all.index)
        joined_pivot_tco2e_all["species_treeocloud"] = joined_pivot_tco2e_all[
            "species"
        ].apply(
            lambda species_name: species_match_coredb_treeocloud(
                species_name, self.config["species_json"]
            )
        )
        joined_pivot_tco2e_all = joined_pivot_tco2e_all.set_index(
            [
                "is_replanting",
                'year_start',
                "plotZone",
                "managementUnit",
                "Plot_ID_exante",
                "species_treeocloud",
            ]
        )

        # let's start build the data frame
        df_tracking_performance = pd.DataFrame()
        df_tracking_performance.index = joined_pivot_tco2e_all.index

        # in the update of ex-ante by different year_start (if any large tree exist and other need to be delayed)
        def exec_plot_backup(suffix, plot):
            if suffix:
                plot_id_backup = int(plot.split("_")[0])
            else:
                plot_id_backup = plot

            return plot_id_backup

        df_tracking_performance["Nr. Trees Ex-Ante"] = joined_pivot_num_trees_all[
            cycle_monitoring + self.prev_ex_ante_planting_year
        ]
        df_tracking_performance["C02 Ex-Ante"] = joined_pivot_tco2e_all[
            cycle_monitoring + self.prev_ex_ante_planting_year
        ]
        df_tracking_performance = df_tracking_performance.reset_index()

        df_tracking_performance[self.field_joins["Plot_ID"]] = (
            df_tracking_performance.apply(
                lambda x: exec_plot_backup(self.plot_id_suffix, x["Plot_ID_exante"]),
                axis=1,
            )
        )

        # display(df_tracking_performance)

        df_plot = self.read_plot()
        # # only takes the important information
        list_plot_columns = [
            "Plot_ID",
            "Plot_Name",
            "Plot_Area",
            "Plot_Status",
            "managementUnit",
            "plotZone",
            "plantingStartDate",
            "plantingEndDate",
        ]
        df_plot_selected_columns = df_plot[list_plot_columns]

        # # implementation joining the information of plot geojson into ex-ante
        df_tracking_performance_exante = pd.merge(
            df_tracking_performance,
            df_plot_selected_columns,
            right_on=["Plot_ID"],
            left_on=[self.field_joins["Plot_ID"]],
            how="left",
            suffixes=("_exante", "_plot"),
        )

        # prioritize managementUnit info from the ex-ante data
        df_tracking_performance_exante = df_tracking_performance_exante.rename(
            columns={"managementUnit_exante": "managementUnit"}
        )
        df_tracking_performance_exante["managementUnit"] = (
            df_tracking_performance_exante["managementUnit"].fillna("").astype(str)
        )
        df_tracking_performance_exante = df_tracking_performance_exante.drop(
            columns=["managementUnit_plot"]
        )

        df_tracking_performance_exante = df_tracking_performance_exante.rename(
            columns={"plotZone_exante": "plotZone"}
        )
        df_tracking_performance_exante["plotZone"] = (
            df_tracking_performance_exante["plotZone"].fillna("").astype(str)
        )
        df_tracking_performance_exante = df_tracking_performance_exante.drop(
            columns=["plotZone_plot"]
        )

        # starting to build the information using the template from https://docs.google.com/spreadsheets/d/179IilSiwUpoBqZZrkxm1WEGPYT1cA8BroUtMZD5F6HU/edit?gid=389938548#gid=389938548
        df_tracking_performance_exante["TPP Name"] = self.TPP_name
        df_tracking_performance_exante["Project Name"] = self.project_name
        df_tracking_performance_exante["Year Measured"] = cycle_monitoring + self.prev_ex_ante_planting_year

        df_tracking_performance_exante = df_tracking_performance_exante.rename(
            columns={"species_treeocloud": "species_exante"}
        )

        df_tracking_performance_expost = expost_validated_count.copy()

        df_tracking_performance_expost[self.field_joins["Plot_ID"]] = (
            df_tracking_performance_expost.apply(
                lambda x: exec_plot_backup(self.plot_id_suffix, x["Plot_ID_expost"]),
                axis=1,
            )
        )

        # implement merge left join, expost data (data validation) and plot
        df_tracking_performance_expost = pd.merge(
            df_tracking_performance_expost,
            df_plot_selected_columns,
            right_on=["Plot_ID"],
            left_on=[self.field_joins["Plot_ID"]],
            how="left",
        )

        df_tracking_performance_expost["TPP Name"] = self.TPP_name
        df_tracking_performance_expost["Project Name"] = self.project_name
        df_tracking_performance_expost["Year Measured"] = cycle_monitoring + self.prev_ex_ante_planting_year

        df_tracking_performance_expost = df_tracking_performance_expost.rename(
            columns={"check_result_data_species_check_manual": "species_expost"}
        )

        # display(df_tracking_performance)

        # merging exante and expost
        df_tracking_performance = pd.merge(
            df_tracking_performance_exante,
            df_tracking_performance_expost,
            left_on=[
                "Plot_ID_exante",
                "species_exante",
                "TPP Name",
                "Project Name",
                "Year Measured",
                "is_replanting",
                'year_start',
                self.field_joins["Plot_ID"],
            ]
            + list_plot_columns,
            right_on=[
                "Plot_ID_expost",
                "species_expost",
                "TPP Name",
                "Project Name",
                "Year Measured",
                "is_replanting",
                'year_start',
                self.field_joins["Plot_ID"],
            ]
            + list_plot_columns,
            how="outer",
        )

        # Fill NaN in species as a whole grouping - assuming there will be no cross changing because it is impossible to track every trees indivually (replace ex-ante per tree level in expost)
        df_tracking_performance["species"] = df_tracking_performance[
            "species_exante"
        ].fillna(df_tracking_performance["species_expost"])

        # Fill NaN in each other columns - plot
        df_tracking_performance["Plot_ID"] = df_tracking_performance[
            "Plot_ID_exante"
        ].fillna(df_tracking_performance["Plot_ID_expost"])

        # as per template, we will check delta trees per row
        df_tracking_performance["Delta trees"] = (
            df_tracking_performance["Nr. Trees Ex-Post"]
            - df_tracking_performance["Nr. Trees Ex-Ante"]
        )

        # Calculate Delta trees % using apply with a lambda function
        df_tracking_performance["Delta trees %"] = df_tracking_performance.apply(
            lambda row: (
                (row["Delta trees"] / row["Nr. Trees Ex-Ante"] * 100)
                if row["Nr. Trees Ex-Ante"] > 0
                else 0
            ),
            axis=1,
        )

        list_all_columns = df_tracking_performance.columns

        first_columns = [
            "Plot_ID",
            "species",
            "Plot_Name",
            "Plot_Area",
            "Plot_Status",
            "managementUnit",
            "plotZone",
            "plantingStartDate",
            "plantingEndDate",
        ]

        sorted_columns = first_columns + [
            f for f in list_all_columns if f not in first_columns
        ]

        df_tracking_performance = df_tracking_performance[sorted_columns]

        return df_tracking_performance

    def summarize_species(self, cycle_monitoring="") -> pd.DataFrame:
        if cycle_monitoring == "":
            cycle_monitoring = self.config["monitoring_year"]

        list_fields = ["Nr. Trees Ex-Ante", "Nr. Trees Ex-Post"]
        if cycle_monitoring == 0:
            list_fields = list_fields

        elif cycle_monitoring > 0:
            list_fields = [
                "Nr. Trees Ex-Ante",
                "Nr. Trees Ex-Post",
                "C02 Ex-Ante",
                "C02 Measured",
            ]

        df_performance = self.comparing_exante_expost()

        # aggregation in summary by species dashboard performance
        df_species_summary = df_performance.groupby(["species"])[list_fields].sum()

        df_species_summary = df_species_summary.reset_index()

        df_species_summary.columns = ["species"] + list(df_species_summary.columns[1:])

        if cycle_monitoring > 0:
            # make sure if the data has tree_measurement_auto
            # Copy the DataFrame and apply the filter
            # df_filtered_large_tree = self.get_df_validated(is_zone_check=self.is_zone_check, is_plot_check=self.is_plot_check).copy()
            df_filtered_large_tree = self.df_meas_plot.copy()
            df_filtered_large_tree = df_filtered_large_tree[
                df_filtered_large_tree["measurement_type"] == "tree_measurement_auto"
            ]

            # Check if the filtered DataFrame is empty
            if df_filtered_large_tree.empty:
                print("there is no large tree in the measurement (ex-post)!")
            else:
                print("there are large tree in the measurement (ex-post)!")
                # print(df_filtered_large_tree)
                joined_df = pd.merge(
                    self.analyze_large_trees_only(),
                    df_species_summary,
                    right_on=["species"],
                    left_on=["species_expost"],
                    how="right",
                )

                df_species_summary = joined_df[
                    [
                        "species",
                        "Nr. Trees Ex-Ante",
                        "Nr. Trees Ex-Post",
                        "Large_tree num",
                        "C02 Ex-Ante",
                        "Large_tree_only CO2 Ex-ante",
                        "C02 Measured",
                    ]
                ]

        # Select only the numeric columns (integer or float types)
        numeric_columns = df_species_summary.select_dtypes(include=["number"])

        # Calculate the grand total by summing across the selected numeric columns
        grand_total = numeric_columns.sum()

        # Append the grand total as a new row with a custom label (e.g., 'Grand Total')
        grand_total_row = pd.DataFrame([grand_total], index=[("Grand Total", "")])

        # Append the grand total row to the summary DataFrame
        df_performance_summary_with_total = pd.concat(
            [df_species_summary, grand_total_row]
        ).fillna("")

        df_performance_summary_with_total["Delta trees"] = (
            df_performance_summary_with_total["Nr. Trees Ex-Post"]
            - df_performance_summary_with_total["Nr. Trees Ex-Ante"]
        )

        # Calculate Delta trees % using apply with a lambda function
        df_performance_summary_with_total["Delta trees %"] = (
            df_performance_summary_with_total.apply(
                lambda row: (
                    (row["Delta trees"] / row["Nr. Trees Ex-Ante"] * 100)
                    if row["Nr. Trees Ex-Ante"] > 0
                    else 0
                ),
                axis=1,
            )
        )
        # Calculate Delta trees % using apply with a lambda function
        df_performance_summary_with_total["Delta trees %"] = (
            df_performance_summary_with_total.apply(
                lambda row: (
                    (row["Delta trees"] / row["Nr. Trees Ex-Ante"] * 100)
                    if row["Nr. Trees Ex-Ante"] > 0
                    else 0
                ),
                axis=1,
            )
        )

        return df_performance_summary_with_total

    def summarize_plot(self, cycle_monitoring="") -> pd.DataFrame:
        df_performance = self.comparing_exante_expost()

        if cycle_monitoring == "":
            cycle_monitoring = self.config["monitoring_year"]

        list_fields = ["Nr. Trees Ex-Ante", "Nr. Trees Ex-Post"]
        if cycle_monitoring == 0:
            list_fields = list_fields

        elif cycle_monitoring > 0:
            list_fields = [
                "Nr. Trees Ex-Ante",
                "Nr. Trees Ex-Post",
                "C02 Ex-Ante",
                "C02 Measured",
            ]
            list_fields += [
                field
                for field in df_performance.columns
                if "Nr Tree Evidence Expost" == field or "Nr Large Tree Expost" == field
            ]

        # aggregation in summary by species dashboard performance
        df_plot_summary = df_performance.groupby(["Plot_ID"])[list_fields].sum()

        # Calculate the grand total (summing across all groups)
        grand_total = df_plot_summary.sum()

        # Append the grand total as a new row with a custom label (e.g., 'Grand Total')
        grand_total_row = pd.DataFrame([grand_total], index=[("Grand Total", "")])

        # Append the grand total row to the summary DataFrame
        df_performance_summary_with_total = pd.concat(
            [df_plot_summary, grand_total_row]
        )

        df_performance_summary_with_total["Delta trees"] = (
            df_performance_summary_with_total["Nr. Trees Ex-Post"]
            - df_performance_summary_with_total["Nr. Trees Ex-Ante"]
        )

        # Calculate Delta trees % using apply with a lambda function
        df_performance_summary_with_total["Delta trees %"] = (
            df_performance_summary_with_total.apply(
                lambda row: (
                    (row["Delta trees"] / row["Nr. Trees Ex-Ante"] * 100)
                    if row["Nr. Trees Ex-Ante"] > 0
                    else 0
                ),
                axis=1,
            )
        )

        df_performance_summary_with_total = (
            df_performance_summary_with_total.reset_index()
        )

        df_performance_summary_with_total.columns = ["Plot_ID"] + list(
            df_performance_summary_with_total.columns[1:]
        )

        return df_performance_summary_with_total

    def analyze_large_trees_only(self, formulas_allometry="", growth_model_exante=""):

        # filtering only the large trees to analyze in the ex-post
        # df_filtered_large_tree = self.get_df_validated(is_plot_check=self.is_plot_check, is_zone_check=self.is_zone_check).copy()
        df_filtered_large_tree = self.df_meas_plot.copy()
        df_filtered_large_tree = df_filtered_large_tree[
            df_filtered_large_tree["measurement_type"] == "tree_measurement_auto"
        ]  # makesure that this columns is available

        df_filtered_large_tree["species_coredb"] = df_filtered_large_tree.apply(
            lambda x: species_reverse_coredb(
                x["check_result_data_species_check_manual"], self.config["species_json"]
            ),
            axis=1,
        )

        if formulas_allometry == "":
            formulas_allometry = self.config["formulas_allometry"]

        if growth_model_exante == "":
            growth_model_exante = self.config["growth_model_exante"]

        selected_formulas_ex_ante = pd.read_csv(formulas_allometry)

        selected_formulas_ex_ante = selected_formulas_ex_ante[
            ["Lat. Name", "TTB formula, tdm", "WD variable"]
        ]  # hardcoded, please comeback again later if there is error here

        selected_formulas_ex_ante = selected_formulas_ex_ante.drop_duplicates()

        df_filtered_large_tree_merged = df_filtered_large_tree.copy()
        df_filtered_large_tree_merged = pd.merge(
            df_filtered_large_tree_merged,
            selected_formulas_ex_ante,
            left_on="species_coredb",
            right_on="Lat. Name",
            how="left",
        )

        selected_growth_model = pd.read_csv(growth_model_exante)
        cycle_monitoring = self.cycle_monitoring

        selected_growth_model_cycle = selected_growth_model.copy()
        selected_growth_model_cycle = selected_growth_model_cycle[
            selected_growth_model_cycle["year"] == cycle_monitoring
        ]

        selected_growth_model_cycle = selected_growth_model_cycle[
            ["Tree Species(+origin of allom. formula)", "year", "DBH"]
        ].rename(columns={"year": "year_Ex_ante", "DBH": "DBH_expected_cm"})

        df_filtered_large_tree_growth_merged = pd.merge(
            df_filtered_large_tree_merged,
            selected_growth_model_cycle,
            left_on="species_coredb",
            right_on="Tree Species(+origin of allom. formula)",  # THIS IS HARDCODED REMEMBER TO CHECK AGAIN
            how="left",
        )

        # calc_biomass_formula(ttb_formula, wd, dbh, height):
        df_filtered_large_tree_growth_merged["expected_tco2e"] = (
            df_filtered_large_tree_growth_merged.apply(
                lambda x: calc_biomass_formula(
                    x["TTB formula, tdm"], x["WD variable"], x["DBH_expected_cm"], ""
                ),
                axis=1,
            )
        )  # ignore height at the moment we put ''

        df_summary = (
            df_filtered_large_tree_growth_merged.groupby(
                ["check_result_data_species_check_manual"]
            )
            .agg(expected_tco2e_sum=("expected_tco2e", "sum"), id_count=("id", "count"))
            .reset_index()
            .rename(
                columns={
                    "check_result_data_species_check_manual": "species_expost",
                    "expected_tco2e_sum": "Large_tree_only CO2 Ex-ante",
                    "id_count": "Large_tree num",
                }
            )
        )

        return df_summary

    def stat_all(
        self, large_tree=True, scale="mu", filter_species=False, species_list="", rename_index_total = True
    ):
        # Validate the 'scale' argument
        # if scale not in ['mu', 'csu']:
        #     raise ValueError("Invalid value for scale. Expected 'mu' or 'csu'.")
        if large_tree:
            self.df_meas_plot["dbh_cm"] = (
                self.df_meas_plot["treeDBHmm"].astype(float) / 10.00
            )
        self.df_meas_plot_filtered = self.df_meas_plot.copy()
        list_values = ["age_month", "id"]
        agg_dict = {
            "age_month": ["mean", "std", "max", "min", "median"],
            "id": ["count"],
        }

        if filter_species:
            self.df_meas_plot_filtered = self.df_meas_plot_filtered[
                self.df_meas_plot_filtered["check_result_data_species_check_manual"].isin(species_list)
            ]

        if large_tree:
            self.df_meas_plot_filtered = self.df_meas_plot_filtered[
                self.df_meas_plot_filtered["measurement_type"]
                == "tree_measurement_auto"
            ]
            list_values = ["age_month", "dbh_cm", "co2_tree_captured_tonnes", "id"]
            agg_dict = {
                "age_month": ["mean", "std", "max", "min", "median"],
                "dbh_cm": ["mean", "std", "max", "min", "median"],
                "co2_tree_captured_tonnes": [
                    "mean",
                    "std",
                    "max",
                    "min",
                    "median",
                    "sum",
                ],
                "id": ["count"],
            }

        else:
            self.df_meas_plot_filtered = self.df_meas_plot_filtered
            list_values = list_values
            agg_dict = agg_dict

        index_column = ["managementUnit", "check_result_data_species_check_manual"]
        all_row = ("All", "")
        column_renamed = {"level_0": "managementUnit", "level_1": "species_treeocloud"}
        if scale == "mu":
            index_column = index_column
            column_renamed = column_renamed
            all_row = all_row
        elif scale == "csu":
            index_column = [
                "Plot_ID_expost",
                "managementUnit",
                "check_result_data_species_check_manual",
            ]
            column_renamed = {
                "level_0": "Plot_ID_expost",
                "level_1": "managementUnit",
                "level_2": "species_treeocloud",
            }
            all_row = ("Subtotal", "", "")

        # Create the pivot table without margins
        pivot_table = pd.pivot_table(
            data=self.df_meas_plot_filtered,
            values=list_values,
            index=index_column,
            aggfunc=agg_dict,
            #     margins=True # don't know why this error, then all the following process needed to manually add the aggregation (grand total)
        )

        # Rename columns to include function names
        pivot_table.columns = [f"{col[0]}_{col[1]}" for col in pivot_table.columns]

        # Round the mean of 'age_month' and convert to integer
        pivot_table["age_month_mean"] = (
            pivot_table["age_month_mean"].round(0).astype(int)
        )

        if large_tree:
            # Calculate growth metrics
            pivot_table["growth_cm_per_month"] = (
                pivot_table["dbh_cm_mean"] / pivot_table["age_month_mean"]
            )
            pivot_table["growth_cm_per_year"] = pivot_table["growth_cm_per_month"] * 12

        # Calculate overall statistics for 'All' row
        overall_metrics = (
            self.df_meas_plot_filtered.agg(agg_dict)
            .reset_index()
            .rename(columns={"index": "stat"})
        )

        melt_data_metrics = pd.melt(
            overall_metrics,
            id_vars=["stat"],
            value_vars=list_values,
            var_name="metric",
            value_name="value",
        )

        melt_data_metrics["id_seq"] = (
            melt_data_metrics["metric"] + "_" + melt_data_metrics["stat"]
        )

        melt_data_metrics = melt_data_metrics[["id_seq", "value"]]

        melt_data_metrics = pd.pivot_table(data=melt_data_metrics, columns=["id_seq"])

        melt_data_metrics = (
            melt_data_metrics.reset_index()
        )  # Reset the index to remove 'concat_name' from being the index

        # If there's an index name after resetting, clear it
        melt_data_metrics.index.name = None
        melt_data_metrics = melt_data_metrics[
            [f for f in melt_data_metrics.columns if f != "index"]
        ]

        overall_metrics = melt_data_metrics.copy()

        # Add growth metrics for the overall row
        overall_metrics["age_month_mean"] = (
            overall_metrics["age_month_mean"].round(0).astype(int)
        )

        if large_tree:
            overall_metrics["growth_cm_per_month"] = (
                overall_metrics["dbh_cm_mean"] / overall_metrics["age_month_mean"]
            )
            overall_metrics["growth_cm_per_year"] = (
                overall_metrics["growth_cm_per_month"] * 12
            )

        
        # --- MODIFIED LOGIC STARTS HERE ---
        if rename_index_total:
            # This 'True' path is unchanged. It uses the grand total 'overall_metrics'.
            overall_metrics.index = pd.MultiIndex.from_tuples([all_row])
            pivot_table = pd.concat([pivot_table, overall_metrics])
        else:
            # This 'False' path now generates the species-specific subtotals.
            
            # 1. Group by species and aggregate to get one subtotal row per species.
            species_subtotals = self.df_meas_plot_filtered.groupby(
                "check_result_data_species_check_manual"
            ).agg(agg_dict)

            # 2. Flatten the columns to match the main pivot_table's format.
            species_subtotals.columns = [f"{col[0]}_{col[1]}" for col in species_subtotals.columns]

            # 3. Apply the same post-calculations to the subtotal rows.
            species_subtotals["age_month_mean"] = (
                species_subtotals["age_month_mean"].round(0).astype(int)
            )
            if large_tree:
                species_subtotals["growth_cm_per_month"] = (
                    species_subtotals["dbh_cm_mean"] / species_subtotals["age_month_mean"]
                )
                species_subtotals["growth_cm_per_year"] = (
                    species_subtotals["growth_cm_per_month"] * 12
                )
            
            # 4. Create a special MultiIndex to match the pivot_table's structure.
            # The first level is '{species}_stat' and the second level is blank.
            # This is the key step to make your downstream filtering work.
            subtotal_index = pd.MultiIndex.from_tuples(
                [(f"{idx}_stat", "") for idx in species_subtotals.index if idx]
            )
            species_subtotals = species_subtotals[species_subtotals.index != '']
            species_subtotals.index = subtotal_index

            # 5. Combine the main pivot table with the new species subtotal rows.
            pivot_table = pd.concat([pivot_table, species_subtotals])
        # --- MODIFIED LOGIC ENDS HERE ---

        # Reset the index to make the current index a column
        pivot_table = pivot_table.reset_index()

        # Split the 'index' column into two separate columns: 'managementUnit' and 'species_treeocloud'
        pivot_table = pivot_table.rename(columns=column_renamed)
        return pivot_table

    def stat_per_species(self, large_tree=True, scale="mu", rename_index_total=True):
        df_meas_plot_species_list = self.df_meas_plot[
            "check_result_data_species_check_manual"
        ].unique()

        stat_pivot = self.stat_all(
                large_tree=large_tree,
                scale=scale,
                filter_species=True,
                species_list=df_meas_plot_species_list,
                rename_index_total=rename_index_total
        )
            
        return stat_pivot

    def analyze_growth(
        self, name_column_species_model="Tree Species(+origin of allom. formula)"
    ):
        # get the stat and has the sub-total of species
        stats_large_tree_species = self.stat_per_species(scale="mu", large_tree=True, rename_index_total=False)

        # get the unique list of the species
        list_species = [
            f"{i}_stat"
            for i in stats_large_tree_species.species_treeocloud.unique()
            if i != ""
        ]

        # only take the sub-total field to filter and get the info later to use
        filter_stat_df = stats_large_tree_species[
            stats_large_tree_species["managementUnit"].isin(list_species)
        ]
        filter_stat_df = filter_stat_df.copy()
        filter_stat_df["species_treeocloud"] = filter_stat_df.apply(
            lambda x: x["managementUnit"].replace("_stat", ""), axis=1
        )

        # get the original growth model in ex-ante
        selected_growth_model = self.config["growth_model_exante"]
        ex_ante_growth = cleaning_csv_df(pd.read_csv(selected_growth_model))
        ex_ante_growth["species_treeocloud"] = ex_ante_growth.apply(
            lambda x: species_match_coredb_treeocloud(
                x[name_column_species_model], self.config["species_json"]
            ),
            axis=1,
        )

        # merging/ joining the data from the stat result and original growth
        ex_ante_growth_add_stat = pd.merge(
            ex_ante_growth,
            filter_stat_df[["species_treeocloud", "growth_cm_per_year"]],
            left_on=["species_treeocloud"],
            right_on=["species_treeocloud"],
            how="left",
        )

        # calculate the linear dbh in cm
        ex_ante_growth_add_stat["adj_linear_cm"] = (
            ex_ante_growth_add_stat["growth_cm_per_year"]
            * ex_ante_growth_add_stat["year"]
        )

        # adding the max adjusted linear growth with the new row to target the max DBH
        df = ex_ante_growth_add_stat

        # Target DBH at the last recorded year
        target_dbh = df["DBH"].iloc[-1]
        current_adj_growth = df["adj_linear_cm"].iloc[-1]
        growth_cm_per_year = df["growth_cm_per_year"].iloc[-1]

        # Create a copy of the last row to use as a template for additional rows
        new_row = df.iloc[-1].copy()

        # Loop to add rows until adj_linear_growth meets or exceeds the target DBH
        while current_adj_growth < target_dbh:
            # Increment the year
            new_row["year"] += 1
            # Increase the adjusted linear growth
            current_adj_growth += growth_cm_per_year
            new_row["adj_linear_cm"] = current_adj_growth
            # Append the new row to the DataFrame
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # reassign
        ex_ante_growth_add_stat = df

        list_ex_ante_growth_final = []

        for species in ex_ante_growth_add_stat["species_treeocloud"].unique():
            species_df = ex_ante_growth_add_stat[
                ex_ante_growth_add_stat["species_treeocloud"] == species
            ].copy()

            # Extract initial DBH and parameters for growth model
            initial_dbh = species_df["adj_linear_cm"].iloc[0]  # DBH at year 1
            max_dbh = species_df[
                "adj_linear_cm"
            ].max()  # Maximum DBH to asymptotically approach
            inflection_point_guess = species_df[
                "year"
            ].median()  # Estimated year for inflection point, around mid year will get depleted growth rate, become stagnant

            species_df = remodel_growth(
                "DBH",
                "adj_linear_cm",
                initial_dbh=initial_dbh,
                max_dbh=max_dbh,
                inflection_point_guess=inflection_point_guess,
                projected_years=35,
                species=species,
                species_df=species_df,
            )

            list_ex_ante_growth_final.append(species_df)

        ex_ante_growth_final = pd.concat(list_ex_ante_growth_final)
        return ex_ante_growth_final

    def mortality_analysis(self, list_columns_analysis, list_index_level):
        validated_mortality_column = "mortality_validated"

        if self.df_expost_filtered_mort is None or self.df_expost_filtered_mort.empty:
            df_expost_all = self.df_meas

            # rejection for mortality rate dead
            list_probably_format_no_tree_data_team_version = [
                "no_tree",
                "no-tree",
                "no tree",
            ]
            list_probably_format_dead_tree_data_team_version = [
                "Dead",
                "dead",
                "dead_tree",
            ]
            # identify the other rejection beside no_tree and dead_tree
            list_other_rejection = [
                "zone_check_failed",
                "plot_check_failed",
                "no_image_no_species",
                "unvalidated",
            ]

            if (
                self.config.get("filtered_by_datateam", None) != None
            ):  # using the format (new one from data team))
                # aligning the same conventional name of Dead trees as dead_tree
                df_expost_all["reject_reason"] = df_expost_all.apply(
                    lambda x: (
                        "dead_tree"
                        if x["reject_reason"]
                        in list_probably_format_dead_tree_data_team_version
                        else x["reject_reason"]
                    ),
                    axis=1,
                )

                # formatting the null as alive based on data team format 'reject_reason' column
                # grouping the no_tree and dead_trees as dead_tree later in mortality calculation
                df_expost_all[validated_mortality_column] = df_expost_all.apply(
                    lambda x: (
                        x["reject_reason"]
                        if x["reject_reason"]
                        in list_probably_format_no_tree_data_team_version
                        + list_probably_format_dead_tree_data_team_version
                        else (
                            "alive"
                            if pd.isna(x["reject_reason"])
                            else x["reject_reason"]
                        )
                    ),
                    axis=1,
                )

                # since in the validation process there are no cross checking mostly, on the area outside go-zone, we will exclude them in the analysis
                df_expost_filtered = df_expost_all.copy()
                df_expost_filtered = df_expost_filtered[
                    (
                        ~df_expost_filtered[validated_mortality_column].isin(
                            list_other_rejection
                        )
                    )
                ]

                # using db if the data is not available in the validation result
                df_expost_filtered["species_treeocloud"] = df_expost_filtered.apply(
                    lambda x: (
                        x["species_db"]
                        if pd.isna(x["tree_species_code"])
                        else x["tree_species_code"]
                    ),
                    axis=1,
                )

            else:  # using the format (old one from data team (Gowtham))
                # aligning to the recent format from the old one - and standardize no_tree and dead_tree
                df_expost_all["measurement_uuid"] = df_expost_all["id"]
                df_expost_all["reject_reason"] = df_expost_all.apply(
                    lambda x: (
                        "dead_tree"
                        if x["check_result_data_species_check_manual"]
                        in list_probably_format_dead_tree_data_team_version
                        else (
                            "no_tree"
                            if x["check_result_data_species_check_manual"]
                            in list_probably_format_no_tree_data_team_version
                            else (
                                x["check_result_data_species_check_manual"]
                                if x["check_result_data_species_check_manual"]
                                in list_other_rejection
                                else None
                            )
                        )
                    ),  # as the current format, the data approved in reject_reason field assigned as ''
                    axis=1,
                )

                df_expost_all[validated_mortality_column] = df_expost_all.apply(
                    lambda x: (
                        x["reject_reason"]
                        if x["reject_reason"]
                        in [
                            "dead_tree",
                            "no_tree",
                        ]  # noted that these two will be marked as dead trees in mortality calculation
                        else (
                            "alive"
                            if pd.isna(x["reject_reason"])
                            else x["reject_reason"]
                        )
                    ),  # following the current format '' as approved (alive)
                    axis=1,
                )

                # since in the validation process there are no cross checking mostly, on the area outside go-zone, we will exclude them in the analysis
                df_expost_filtered = df_expost_all.copy()
                df_expost_filtered = df_expost_filtered[
                    (
                        ~df_expost_filtered[validated_mortality_column].isin(
                            list_other_rejection
                        )
                    )
                ]

                # using check_result_species_check_manual if the data is rejected
                df_expost_filtered["species_treeocloud"] = df_expost_filtered.apply(
                    lambda x: (
                        x["species_db"]
                        if x["check_result_data_species_check_manual"]
                        in list_probably_format_no_tree_data_team_version
                        + list_probably_format_dead_tree_data_team_version
                        else x["check_result_data_species_check_manual"]
                    ),
                    axis=1,
                )

            df_expost_filtered["species_coredb"] = df_expost_filtered.apply(
                lambda x: species_reverse_coredb(
                    x["species_treeocloud"], self.config["species_json"]
                ),
                axis=1,
            )

            self.df_expost_filtered_mort = df_expost_filtered
        else:
            df_expost_filtered = self.df_expost_filtered_mort.copy()

        pivot_plot_mortality = pd.pivot_table(
            aggfunc="count",
            data=df_expost_filtered[
                list_columns_analysis + [validated_mortality_column]
            ],
            columns=[validated_mortality_column],
            index=list_index_level,
            margins=True,
        )
        # Flatten the column names
        list_columns = pivot_plot_mortality.columns
        new_list_columns = [f"num_trees_{col[1]}" for col in list_columns]

        pivot_plot_mortality.columns = new_list_columns
        pivot_plot_mortality = pivot_plot_mortality.fillna(0)

        # mortality rate here define as (dead_Tree + no_tree)/ all trees
        pivot_plot_mortality["mortality_rate"] = (
            pivot_plot_mortality.get("num_trees_dead_tree", 0)
            + pivot_plot_mortality.get("num_trees_no_tree", 0)
        ) / pivot_plot_mortality.get("num_trees_All", np.nan)

        return pivot_plot_mortality

    def replanting_plan(
        self,
        use_mortality_data=False,
        df_mortality: pd.DataFrame = None,
        mortality_threshold="",
        list_override_species: list = [],  # list species here as the treeocloud db manually not using threshold above - ensure that exist in species_json
        list_override_proportion: list = [],  # list proportion in 100 (%) of i above, with every item is a dictionary of protected_zone and production_zone
        ex_ante_dist_csu: pd.DataFrame = None,
        ex_post_dist_csu: pd.DataFrame = None,
        manual_replanting_plan="",
        conditional_gap_replanting_year=False, proportion_delay_list=[50,50],
        num_year_replanting_add=1,
        is_include_large_tree=False,  # by default we will use no large tree, if there are large tree in expost, please adjust to True
        include_prevnatural_thinning=0,  # this one added and set as default (5%) that we will replant after the previous existing number trees has apply the natural thinning so that we replant trees more
        use_remaining_year_exante = False,
        earlier_year = 0
    ):
        '''
        ########### example argument, and description
        manual_replanting_plan = ''  # this is if we have pre-define csv of replanting plan only (not entire seedling dist)

        is_include_large_tree = True # relate to the column grouping, set this to True, if your calc define the large tree already with the column of 'measurement_type'

        # set for the condition if we want to split the replanting action e.g into two years
        conditional_gap_replanting_year = True
        # if the conditional_gap_replanting_year = True --> set this
        proportion_delay_list = [50,50]  # this should get the 100

        num_year_replanting_add = 1 #if there is any delay

        ##### this one is important related to the tpp agreement (stick with addendum, or new replanting)
        use_remaining_year_exante = False # if we want to use the prev. scenario (e.g replanting addendum) without additional replanting adjustment

        ####
        # if use remaining_year_exante = False --> set these
        use_mortality_data = True # for this to be True, we need to setup the dataframe mortality from prev. scenario
        # if use_mortality_data True --> set this
        df_mortality = species_mortality_tcloud
        # if use_mortality_data False --> set these
        list_override_species = ['falcataria_moluccana']
        # list_override_proportion = [{'protected_zone': 100,
        #                               'production_zone':100}] # percentage above as per index (i) of list_override_species
        list_override_proportion = []
        include_prevnatural_thinning = 0 # adding more tree to mitigate natural thinning
        mortality_threshold = 20 #selection of tree species based on threshold on mortality rate data (df)
        '''

        if manual_replanting_plan == "":
            # previous_scenario = self.updated_exante.csu_seedling
            previous_scenario = ex_ante_dist_csu
            update_species_name = self.update_species_name
            if update_species_name != {}:
                for k,v in update_species_name.items():
                    previous_scenario = previous_scenario.rename(columns={k+'_num_trees':v+'_num_trees'})# change this into the updated column of the data update
            max_year_exante = int(previous_scenario['year_start'].max())
            raw_expost_update = ex_post_dist_csu.copy()
            raw_expost_update['mu'] = raw_expost_update['managementUnit']
            max_year_expost = int(raw_expost_update['year_start'].max())

            # if there is large tree, we need to ensure the distribution clearly for plot distribution large trees and added into replanting plan with mixing later the tree evidence
            if is_include_large_tree == True:  # this is after the process in recalc_exante
                agg_expost = process_plot_data(ex_post_dist_csu, expost_agg=True, columns_to_remove=['measurement_type', "unique_label",'is_replanting'])

            else:
                # plot_distribution_update = expost.seedling_distribution_updated
                agg_expost = process_plot_data(ex_post_dist_csu, expost_agg=True, columns_to_remove=["unique_label",'is_replanting'])


            if use_remaining_year_exante:
                if  max_year_expost <  max_year_exante:
                    list_year_filter = [i+max_year_expost+1 for i in range(max_year_exante-max_year_expost)] # list year filter, get the remaining list year after expost data
                    agg_exante = process_plot_data(ex_ante_dist_csu, year_filter=list_year_filter,
                                                columns_to_remove = ['is_replanting', 'plantingStartDate', 'plantingEndDate', 'area_ha'])
                    
                    if update_species_name != {}:
                        for k,v in update_species_name.items():
                            agg_exante = agg_exante.rename(columns={k+'_num_trees':v+'_num_trees'})# change this into the updated column of the data update
                
                    df_replanting_only = agg_exante.copy()
                    df_replanting_only['All_trees_need_replanted'] = df_replanting_only['num_trees_total']
                    
                
                else:
                    raise ValueError(f"you can't choose the use_remaining_year_exante argument, because max_year_expost ({max_year_expost}) >  max_year_exante ({max_year_exante}))")
            
            else:
                agg_exante = process_plot_data(ex_ante_dist_csu,
                                                columns_to_remove = ['is_replanting', 'plantingStartDate', 'plantingEndDate', 'area_ha'], 
                                                year_max=max_year_expost) # use expost max, to join the data later

                plot_distribution_update = agg_expost

                previous_scenario = agg_exante.copy()
                if update_species_name != {}:
                    for k,v in update_species_name.items():
                        previous_scenario = previous_scenario.rename(columns={k+'_num_trees':v+'_num_trees'})# change this into the updated column of the data update


                # previous_scenario['All_trees_planned'] = previous_scenario[
                #     [col for col in previous_scenario.columns if col.endswith("_num_trees")]
                # ].sum(axis=1)

                previous_scenario['All_trees_planned'] = previous_scenario['num_trees_total']

                # check ex-post based only update
                plot_distribution_update = plot_distribution_update.rename(
                    columns={"year_max": "year_all_planted"}
                )  # we will replace as planted
                # plot_distribution_update['year_start_planted'] = plot_distribution_update['year_start']
                new_replanting_plot_distribution = pd.merge(
                    plot_distribution_update,
                    previous_scenario[["Plot_ID", "All_trees_planned"]],
                    on=["Plot_ID"],
                    how="left",
                )

                new_replanting_plot_distribution["All_trees_num_expost"] = (
                    new_replanting_plot_distribution[
                        [
                            col
                            for col in new_replanting_plot_distribution.columns
                            if col.endswith("_num_trees")
                        ]
                    ].sum(axis=1)
                )

                new_replanting_plot_distribution["All_trees_num_expost"] = (
                    new_replanting_plot_distribution.apply(
                        lambda x: x["All_trees_num_expost"]
                        * float(100 - include_prevnatural_thinning)
                        / 100.00,
                        axis=1,
                    )
                )

                # Step 1: Calculate 'All_trees_need_replanted' --- readjust the new trees_planned with forecasting the natural thinning if any
                new_replanting_plot_distribution["All_trees_need_replanted"] = (
                    new_replanting_plot_distribution["All_trees_planned"]
                    - new_replanting_plot_distribution["All_trees_num_expost"]
                )

                # Step 2: Adjust 'All_trees_need_replanted' to set negative values to 0 and carry over the deficit
                carry_over = 0
                adjusted_values = []

                # enable the carry over algorithm --> if previous column has negative number (due to over planted) it will put the negative to next column, next column has less number,
                # in total all the trees will be the same
                for index, row in new_replanting_plot_distribution.iterrows():
                    # Add carry-over from the previous row
                    current_value = row["All_trees_need_replanted"] + carry_over

                    if current_value < 0:
                        carry_over = (
                            current_value  # Set carry-over if current value is negative
                        )
                        adjusted_values.append(0)  # Set current row value to 0
                    else:
                        carry_over = 0  # Reset carry-over if no deficit
                        adjusted_values.append(current_value)

                # Update the column with adjusted values
                new_replanting_plot_distribution["All_trees_need_replanted"] = (
                    adjusted_values
                )
                # new_replanting_plot_distribution['is_replanting'] = False

                if use_mortality_data == True:
                    # for the option-1 replanting species selection, we will use the species mortality rate that below the rate of all project mortality
                    # in kplpb all the fruit trees are higher than overall mortality rate (>19%), therefore, all the trees < 19% will be included
                    # let's do it programmatically if the default is using this all number
                    if mortality_threshold == "":
                        mortality_threshold = df_mortality["mortality_rate"].iloc[-1]
                    else:
                        mortality_threshold = mortality_threshold
                    list_good_species = df_mortality[
                        df_mortality["mortality_rate"] < mortality_threshold
                    ].index.to_list()
                    list_good_species = [
                        x for x in list_good_species if x != "All"
                    ]  # just to ensure the line of All (from prev. total row) is not included
                else:  # meaning we dont use the data frame generated
                    if (
                        list_override_species != [] and list_override_proportion == []
                    ):  # manually to just put the species we want to add
                        list_good_species = list_override_species  # this species should be a treeocloud list
                        print(
                            "please make sure this list below is exist in ex-ante plan previously for every zone/ management plot"
                        )
                        print(
                            "other wise if not, please consider to put manual <list_override_proportion> as well"
                        )
                    elif list_override_proportion != [] and list_override_species != []:
                        list_good_species = list_override_species
                        list_good_prop = list_override_proportion
                    else:
                        print("please add argument list_override_species there")
                        raise ValueError("please add argument list_override_species there")
                # return back to the original proportion based on first ex-ante in plot
                # this list from treeo cloud will be converted to coredb naming, and will assess the proportion based on ex-ante proportion
                list_good_species_coredb = [
                    f"{species_reverse_coredb(i, self.config['species_json'])}"
                    for i in list_good_species
                ]
                print("list species for replanting: ", list_good_species_coredb)

                list_proportion_suffix = [i + "_prop" for i in list_good_species_coredb]
                list_num_trees_suffix = [i + "_num_trees" for i in list_good_species_coredb]

                previous_scenario_check_proportion = previous_scenario.copy()
                previous_scenario_check_proportion["filtered_total_planned"] = (
                    previous_scenario_check_proportion.apply(
                        lambda x: x[list_num_trees_suffix].sum(), axis=1
                    )
                )

                # debugging
                # if len(list_override_proportion) == 1:

                #     # Your predefined line, which creates a list of references
                #     list_override_proportion_mult = list_override_proportion * len(list_good_species)

                #     # **THE FIX**: Immediately overwrite the list with a new one containing true copies.
                #     # This iterates through your list of references and makes an independent copy of each item.
                #     list_override_proportion_mult = [item.copy() for item in list_override_proportion_mult]

                #     # The rest of your code now works as expected
                #     list_len = len(list_override_proportion_mult) 

                #     # print(f"DEBUG: The first item's value is {list_override_proportion_mult[0]['protected_zone']} and the divisor is {list_len}")

                #     for item in list_override_proportion_mult:
                #       item['protected_zone'] /= list_len
                #       item['production_zone'] /= list_len

                #     # print(f"Final result: {list_override_proportion_mult[0]['protected_zone']}")
                #     list_override_proportion = list_override_proportion_mult
                
                def cond_prop_zone(row, override_prop_dict):
                    if row["zone"] in override_prop_dict:
                        return float(override_prop_dict[row["zone"]]) / 100.00
                    else:
                        return 0

                # Create new columns using a lambda function and list comprehension - for enforce also manual list shares above
                # data validation is required, that all the zone need to be 100?, so that there is no len here
                for i in range(len(list_good_species)):
                    species = list_good_species[i]
                    if list_override_proportion == [] and use_mortality_data:
                        # we will put 0 if there is over planting
                        previous_scenario_check_proportion[list_proportion_suffix[i]] = (
                            previous_scenario_check_proportion.apply(
                                lambda x: (
                                    x[list_num_trees_suffix[i]]
                                    / x["filtered_total_planned"]
                                    if x["filtered_total_planned"] != 0
                                    else 0
                                ),
                                axis=1,
                            )
                        )
                    elif list_override_proportion != [] and use_mortality_data:
                        if len(list_override_proportion) != len(list_good_species):
                            print('list_override_proportion: ', list_override_proportion)
                            print('list_good_species: ',list_good_species)
                            raise ValueError(f'inconsistent number of species will be replant vs scenario proportion zone \n len(list_override_proportion) ({len(list_override_proportion)})) should equal to len(list_good_species) ({len(list_good_species)})')
                    else:
                        # we will put 0 if there is over planting and we will put the specific protected_zone and production proportion override based on index in the list
                        

                        # print('you manually define the proportion zone, if you define use_mortality_rate True, suggest to use argument prop = [] to use prev. exante proportion scenario')

                        # debug in above, with len. meaning that it will treat as the same
                        zone_prop = list_override_proportion[i]

                        # Apply the function to the DataFrame
                        previous_scenario_check_proportion[list_proportion_suffix[i]] = (
                            previous_scenario_check_proportion.apply(
                                lambda x: cond_prop_zone(x, zone_prop), axis=1
                            )
                        )

                # replanting df creation
                new_replanting_plot_distribution_joined = pd.merge(
                    new_replanting_plot_distribution,
                    previous_scenario_check_proportion[
                        ["Plot_ID", "year_max"]
                        + [
                            col
                            for col in previous_scenario_check_proportion.columns
                            if col.endswith("prop")
                        ]
                    ],
                    left_on=["Plot_ID", "year_all_planted"],
                    right_on=["Plot_ID", "year_max"],
                    how="left",
                )

                df_replanting_only = new_replanting_plot_distribution_joined[
                    [
                        col
                        for col in new_replanting_plot_distribution_joined
                        if not col.endswith("_num_trees")
                    ]
                ]

                df_replanting_only = df_replanting_only.copy()
                for i in list_good_species_coredb:
                    df_replanting_only[i + "_num_trees"] = df_replanting_only.apply(
                        lambda x: round(x[i + "_prop"] * x["All_trees_need_replanted"], 0),
                        axis=1,
                    )

                num_year_replanting_add = num_year_replanting_add
                df_replanting_only["year_start"] = (
                    df_replanting_only["year_all_planted"] + num_year_replanting_add
                )


            if conditional_gap_replanting_year == True:
                # let's automate later if there is some conditional statement related to replanting gap
                # example in inprosula, nursery will give additional gap_year 1 compare to buy seedling
                # change above line code later

                list_to_concat = []
                process_calc = df_replanting_only.copy()
                for i in range(len(proportion_delay_list)):
                    process_calc = df_replanting_only.copy()
                    process_calc['year_start'] = process_calc['year_start'] + [i]
                    process_calc['All_trees_need_replanted'] = process_calc['All_trees_need_replanted'] * proportion_delay_list[i] / 100
                    # process_calc['All_trees_planned'] = process_calc['All_trees_planned'] * proportion_delay_list[i] / 100
                    process_calc['num_trees_total'] = process_calc['num_trees_total'] * proportion_delay_list[i] / 100

                    # loop to all the column with endswith _num_trees
                    for col in process_calc.columns:
                        if col.endswith("_num_trees"):
                            process_calc[col] = process_calc[col] * proportion_delay_list[i] / 100
                    list_to_concat.append(process_calc)

                delayed_df_replanting_only = pd.concat(list_to_concat, ignore_index=True)
                df_replanting_only = delayed_df_replanting_only
            else:
                pass

        else:
            df_replanting_only = cleaning_csv_df(pd.read_csv(manual_replanting_plan))

        df_replanting_only["is_replanting"] = True
        new_replanting_plot_distribution_finalize = pd.concat(
            [ex_post_dist_csu, df_replanting_only], ignore_index=True
        )
        new_replanting_plot_distribution_finalize["is_replanting"] = (
            new_replanting_plot_distribution_finalize.apply(
                lambda x: True if x["is_replanting"] == True else False, axis=1
            )
        )

        if earlier_year != 0:
            df_replanting_only['year_start'] = df_replanting_only['year_start'] - earlier_year
            df_replanting_only['measurement_type'] ='Nr Tree Evidence Expost'  # generalize that the earlier replant, will get tree evidence expected, because we will grouping this
            new_replanting_plot_distribution_finalize = pd.concat(
                    [ex_post_dist_csu, df_replanting_only], ignore_index=True
                )
            
            #merge again and group, to avoid duplication headache (unique combination) - make sure that this following name has the consistent (between plot id and plot name)

            tree_cols = [col for col in new_replanting_plot_distribution_finalize.columns if ('num_trees') in col]

            agg_function = {}
            for i in tree_cols:
                agg_function[i] = 'sum'

            new_replanting_plot_distribution_finalize['managementUnit'] = new_replanting_plot_distribution_finalize['managementUnit'].fillna(new_replanting_plot_distribution_finalize['mu'])
            new_replanting_plot_distribution_finalize['mu'] = new_replanting_plot_distribution_finalize['mu'].fillna(new_replanting_plot_distribution_finalize['managementUnit'])

            if is_include_large_tree:
                column_group = ['Plot_ID','year_start', 'mu','managementUnit', 'zone','is_replanting','Plot_Name','measurement_type']
            else:
                column_group = ['Plot_ID','year_start', 'mu','managementUnit', 'zone','is_replanting','Plot_Name']

            new_replanting_plot_distribution_finalize = new_replanting_plot_distribution_finalize.groupby(column_group).agg(agg_function)

            new_replanting_plot_distribution_finalize = new_replanting_plot_distribution_finalize.reset_index()

        return new_replanting_plot_distribution_finalize

    def scenario_replanting(
        self, df_replanting: pd.DataFrame = None, prev_scenario={}, override_scenario={}
    ):
        if override_scenario == {}:
            # new scenario
            update_all_scenario = {}
            # make sure if version two is not exist yet
            if "replanting" in prev_scenario.keys():
                prev_scenario = prev_scenario["non_replanting"]

            else:
                prev_scenario = prev_scenario

            update_all_scenario["non_replanting"] = prev_scenario

            # replanting only
            # replanting_only = replanting_no_fruit_trees[replanting_no_fruit_trees['is_replanting']==True]
            replanting_only = df_replanting
            replanting_only = replanting_only.dropna(axis=1, how="all")

            # Identify columns ending with '_num_trees'
            num_tree_cols = [
                col for col in replanting_only.columns if col.endswith("_num_trees")
            ]

            # Group by 'zone' and extract species lists
            replanting_dict = {
                "replanting": {
                    zone: [col[:-10] for col in num_tree_cols]
                    for zone, group in replanting_only.groupby("zone")
                }
            }

            # print(replanting_dict)

            adjusted_dict = {}
            for zone, species_list in replanting_dict["replanting"].items():
                adjusted_dict[zone] = {
                    species: self.input_scenario_prev[zone][species]
                    for species in species_list
                    if species in self.input_scenario_prev[zone]
                }

            update_all_scenario["replanting"] = adjusted_dict

        else:
            update_all_scenario = override_scenario

        return update_all_scenario

    def recalc_ex_ante(
        self,
        project_name_update="",
        json_config_prev_relpath="",
        # this will setting and store the info to use as coredb and 30y
        download_csv=False,  # if we want to re-download the coredb google sheet to a csv as follows
        using_rel_path = True,# accomodate the old model
        growth_csv_rel_path="ex_ante/00_input/growth_model_2024-08-13.csv",
        allometry_csv_rel_path="ex_ante/00_input/allometry_model_2024-08-13.csv",
        json_config_prev_abspath = '',
        growth_csv_abs_path='',
        allometry_csv_abs_path='',
        name_column_species_allo="Lat. Name",  # change if necessary the csv is changed
        name_column_species_growth="Tree Species(+origin of allom. formula)",  # use the csv growth, change if needed
        # reupdating growth and allometry for the expost forecast
        re_select_growth_data=False,  # this one if we want to update the growth based on csv above or download first and change lcoation csv and update
        re_update_allometry_model=False,  # this one if we want to update the allometry formula based on csv above or download first and change lcoation csv and update
        # expost arrangement how to forecast
        override_data=False,
        name_mu="",
        zone="",
        year_start=1,
        all_tree_evidence=False,
        override_new_growth_model="",
        gap_harvest=False,
        harvesting_max_percent=59.9,
        thinning_stop=False, # set to False, to accomodate the old model
        override_avg_tree_perha = '', #for thinning stop in expost if it set to True above, should apply this, because in reality, the space for planting is not entire plot
        force_load_seedling_csv="",
        override_new_formula="",
        override_new_scenario="",
        adding_prev_mortality_rate=0,  # if we want to add more mortality rate to expost data in the year 1
        override_mortality_replanting = 40,
        override_natural_thinning='',
        update_species_name={},
        sigmoid_remodel_growth=False,
        override_planting_year=''
    ):

        module_path = os.path.dirname(exante.__file__)
        self.update_species_name = update_species_name

        import json

        # updating and creating the folder for the update_ex-ante in expost folder
        # Directory 01_output to create
        self.output_dir_exante = (
            os.path.dirname(self.config["plot_file_geojson"])
            + "/02_updated_exante/"
            + project_name_update
        )

        # Check if directory exists, if not, create it
        if not os.path.exists(self.output_dir_exante):
            os.makedirs(self.output_dir_exante)
        print(self.output_dir_exante)

        #### for the num_trees_year update with the new monitoring expost as year 0 num_tree
        list_columns_analysis = [
            "measurement_uuid",
            "Plot_ID_expost",
            "species_coredb",
            "is_replanting",
            "year_start",
        ]
        list_index_level = [
            "is_replanting",
            "year_start",
            "Plot_ID_expost",
            "species_coredb",
        ]
        try:
            csu_species_mortality_check = self.mortality_analysis(
                list_columns_analysis, list_index_level
            )
        except Exception as e:
            print(
                "MAKE SURE IN EXPOST DATA THERE are COLUMNS : \n ['measurement_uuid','Plot_ID_expost','species_coredb','is_replanting','year_start']"
            )
            raise ValueError(
                "Please add the following columns in expost data: \n "
                "['is_replanting','year_start' ]",
                f"error {e}",
            )

        if update_species_name != {}:
            with open(self.config["species_json"], "r") as f:
                species_json = json.load(f)

            # Replace the key in the dictionary
            for old_key, new_key in update_species_name.items():
                if old_key in species_json:
                    species_json[new_key] = species_json.pop(old_key)

            # Output the updated dictionary
            # print(species_json)

            with open(
                os.path.join(self.output_dir_exante, "updated_speciescoredb_info.json"),
                "w",
            ) as f:
                json.dump(species_json, f, indent=4)
            self.config["species_json"] = os.path.join(
                self.output_dir_exante, "updated_speciescoredb_info.json"
            )

            # Update only the "species_coredb" level
            csu_species_mortality_check.index = csu_species_mortality_check.index.map(
                lambda index: (
                    index[0],index[1],index[2],
                    (
                        update_species_name[index[3]] # as per update new index with is_replanting # THIS IS HARDCODED INDEX POSITION, PRONE TO BUG!
                        if index[3] in update_species_name.keys()
                        else index[3]
                    ),
                )
            )

        else:
            csu_species_mortality_check = csu_species_mortality_check

        ## NOW ITS TIME TO CALC EX ANTE BASED ON EXPOST and without another planting next year (only update based on to-date planted trees (expost))
        df_performance = self.comparing_exante_expost()
        self.pivot_plot_species = df_performance.pivot(
            index=[
                "Plot_ID",
                "Plot_Name",
                "Plot_Area",
                "managementUnit",
                "Plot_Status",
                "plantingStartDate",
                "plantingEndDate",
                "plotZone",
                'is_replanting',
                'year_start'
            ],
            columns="species",
            values="Nr. Trees Ex-Post",
        )
        print(
            []
            if self.pivot_plot_species.isna().any().any()
            else pd.unique(self.pivot_plot_species.values.ravel())
        )
        # print(self.pivot_plot_species[np.nan].unique())

        # Drop columns with NaN as column names
        # expost.pivot_plot_species = expost.pivot_plot_species.drop(columns=np.nan)
        self.pivot_plot_species = self.pivot_plot_species.loc[
            :, self.pivot_plot_species.columns.notna()
        ]
        print("NaN column is removed")
        print(self.pivot_plot_species.columns)

        # filling the nan value
        self.pivot_plot_species.fillna(0, inplace=True)

        # list_name_columns_species = self.pivot_plot_species.columns
        # for i in list_name_columns_species:
        #    print(i)
        # T4T expost after validation --> there is syzigium aromaticum, we need to remove them
        # print(pivot_plot_species.columns)

        # re-engineer to convert treeo naming to the coredb
        self.pivot_plot_species = self.pivot_plot_species.rename(
            columns=lambda x: species_reverse_coredb(x, self.config["species_json"])
            + "_num_trees"
        )
        print("after edited: ", self.pivot_plot_species.columns)
        self.pivot_plot_species = self.pivot_plot_species.reset_index()

        # now connecting to the plot data
        # plot_df = gpd.read_file(self.config['plot_file_geojson'])

        # getting the part of the plot data since it's already joined
        # plot_df_selected_column = df_performance[['Plot_ID','Plot_Name','Plot_Area', 'managementUnit','Plot_Status','plantingStartDate','plantingEndDate','plotZone']]

        # merging / join to get the same structure of distribution_seedling updated by ex_post
        # merge_plot_species = pd.merge(self.pivot_plot_species, plot_df_selected_column, left_on=['Plot_ID'],right_on = ['Plot_ID'], how='left')
        # we will try to use the existing pivot without rejoin the plot, since already done before in the above instead
        merge_plot_species = self.pivot_plot_species.copy()
        merge_plot_species = merge_plot_species.rename(
            columns={"plotZone": "zone", "managementUnit": "mu"}
        )

        if override_data:
            # override zone and mu since all the same T4T
            # merge_plot_species['mu'] = 'MU_55_1'
            if name_mu != "":
                merge_plot_species["mu"] = name_mu
                # merge_plot_species['zone'] = 'protected_zone'
            if zone != "":
                merge_plot_species["zone"] = zone
                # merge_plot_species['year_start'] = 1
                merge_plot_species["year_start"] = year_start
            # merge_plot_species['year_start'] = 1

        else:
            print(" we will use the plot from treeo cloud data")
            has_nan_mu = merge_plot_species["mu"].isna().any()
            has_nan_zone = merge_plot_species["zone"].isna().any()
            if has_nan_mu or has_nan_zone:
                print(
                    "there is nan (ManagementUnit, and Zone) in plot_data, please override manually or update in the treeo cloud"
                )
                return None

            print(
                "we will also define the year_start based on the data_datetime later if large tree exist"
            )
            if all_tree_evidence == True:
                merge_plot_species["year_start"] = (
                    1  # meaning that its considered as year 0, trees does not growing one year properly and can't get dbh info
                )
            else:
                # if there are large trees in the ex-post!
                # now we need to understand, which plot that actually has been delayed (updated the year_start)
                # if the trees is small tree evidence we will consider the plot in year 0, here we will delayed 1 year after the last large tree
                # large trees is year 1, since the monitoring just only one year
                print(
                    "age of planting every plot will be generated if large tree monitoring vs tree evidence. \n \
                      Year 0 is tree evidence, and large tree monitoring is Year 1. This is fine in the first annual monitoring only"
                )

                # identify if the field name has either tree evidence or large trees, to be included later
                field_names = [
                    field
                    for field in df_performance.columns
                    if "Nr Tree Evidence Expost" == field
                    or "Nr Large Tree Expost" == field
                ]

                # get the melted data structure to get the trees num based on category measurement_type
                plot_species_meas_type = pd.melt(
                    df_performance,
                    id_vars=[
                        "Plot_ID",
                        "Plot_Name",
                        "Plot_Area",
                        "managementUnit",
                        "Plot_Status",
                        "plantingStartDate",
                        "plantingEndDate",
                        "plotZone",
                        'is_replanting',
                'year_start',
                        "species",
                    ],
                    value_vars=field_names,
                    var_name="measurement_type",
                    value_name="trees_num",
                ).dropna()

                # enable to get the stat based on large trees only (this is to get the information of age)
                stat_csu_species = self.stat_per_species(scale="csu", large_tree=True)
                # Drop the row where 'Plot_ID' is 'Subtotal'
                stat_csu_species = stat_csu_species.drop(
                    stat_csu_species[
                        stat_csu_species["Plot_ID_expost"] == "Subtotal"
                    ].index
                )

                # get the max year, to adjust the baseline (year_start)
                max_age_yr = round(stat_csu_species["age_month_max"].max() / 12)

                joined_plot_stat = pd.merge(
                    plot_species_meas_type,
                    stat_csu_species[
                        ["Plot_ID_expost", "species_treeocloud", "age_month_mean"]
                    ],
                    left_on=["Plot_ID", "species"],
                    right_on=["Plot_ID_expost", "species_treeocloud"],
                    how="left",
                )

                joined_plot_stat = joined_plot_stat[
                    ~(joined_plot_stat["trees_num"].isna())
                ]

                # fix if age_month in tree evidence, it will give as 0
                joined_plot_stat["age_month_mean_fixed"] = joined_plot_stat.apply(
                    lambda x: (
                        0
                        if x["measurement_type"] == "Nr Tree Evidence Expost"
                        else x["age_month_mean"]
                    ),
                    axis=1,
                )

                # year_start is the column execution for ex-ante
                # just use the age_month_mean_fixed (because tree evidence is 0,
                # and we will use max age year, to traceaback the year_start,
                # the oldest tree is year_start =1, therefore here is the formula)
                joined_plot_stat["year_start"] = joined_plot_stat.apply(
                    lambda x: (
                        max_age_yr - max(1, round(x["age_month_mean_fixed"] / 12)) + 1
                        if x["measurement_type"] == "Nr Large Tree Expost"
                        else max_age_yr + self.config["monitoring_year"]
                    ),
                    axis=1,
                )

                # re-assign the pivot table since this is different method from above - we will breakdown the plot and year start (plot id not unique)
                self.pivot_plot_species = pd.pivot_table(
                    data=joined_plot_stat,
                    index=[
                        "Plot_ID",
                        "measurement_type",
                        'is_replanting',
                            'year_start',
                        "Plot_Name",
                        "Plot_Area",
                        "managementUnit",
                        "Plot_Status",
                        "plantingStartDate",
                        "plantingEndDate",
                        "plotZone",
                    ],
                    columns=["species"],
                    values="trees_num",
                )
                self.pivot_plot_species = self.pivot_plot_species.rename(
                    columns=lambda x: species_reverse_coredb(
                        x, self.config["species_json"]
                    )
                    + "_num_trees"
                ).reset_index()

                # merging with plot data (geopandas)
                # merge_plot_species = pd.merge(self.pivot_plot_species, plot_df_selected_column, left_on=['Plot_ID'],
                #   right_on = ['Plot_ID'], how='left')

                # no need to join above with Plot data, since we already extracted (joined) in the compare_ func
                merge_plot_species = self.pivot_plot_species.copy()

                merge_plot_species = merge_plot_species.rename(
                    columns={"plotZone": "zone", "management_unit": "mu"}
                )

                # re-adjust the unique_label
                merge_plot_species["unique_label"] = merge_plot_species.apply(
                    lambda x: f"{x['Plot_ID']}_{x['measurement_type']}_{x['year_start']}_{x['is_replanting']}", # fix the issue with consistency unique value of plot_id based on this combination update. found issue in inprosula, now its fixed
                    axis=1,
                )

                # # this is a quickfix/ hotfix we will now focus on unique_label, but we make them as Plot_ID, and has backup column
                # merge_plot_species = merge_plot_species.rename(
                #                 columns={'Plot_ID': 'Plot_ID_backup' })
                
                # merge_plot_species = merge_plot_species.rename(
                #                 columns={'unique_label': 'Plot_ID' })

        # recalculate the area_ha based on treeo cloud Plot_Area
        merge_plot_species["area_ha"] = (
            merge_plot_species["Plot_Area"].astype(float) / 10000.0000
        )

        self.merge_plot_species = merge_plot_species
        # lets recap the number
        # Select columns that contain '_num_trees' in their names
        num_trees_columns = merge_plot_species.filter(like="_num_trees")

        # Sum the values across those columns for the entire DataFrame (column-wise sum)
        summary_row = num_trees_columns.sum()
        self.summary_expost = summary_row

        # cleaning if all 0 because ex-ante expost previous comparison
        num_trees_columns = merge_plot_species.filter(like="num_trees")

        # Select rows where all 'num_trees' columns are 0
        # Filter the columns that contain 'num_trees' in their names
        num_trees_columns = merge_plot_species.filter(like="num_trees")

        # Drop rows where all 'num_trees' columns are 0
        filtered_df = merge_plot_species[~(num_trees_columns == 0).all(axis=1)]
        merge_plot_species = filtered_df.copy()
        self.merge_plot_species_cleaned = merge_plot_species

        
        json_config_relpath = json_config_prev_relpath
        # json_config_relpath = '00_ex_ante_result/00_bll_pool_1_sold/03_t4t/t4t_main.json' # should not a hard-coded
        if using_rel_path:
            # accomodate the old model
            json_config_abspath = os.path.join(module_path, json_config_relpath)
            # growth_csv_abs_path = os.path.join(module_path, 'ex_ante/00_input/growth_model_2024-08-13.csv')
            growth_csv_abs_path = os.path.join(module_path, growth_csv_rel_path)
            # allometry_csv_abs_path = os.path.join(module_path,'ex_ante/00_input/allometry_model_2024-08-13.csv')
            allometry_csv_abs_path = os.path.join(module_path, allometry_csv_rel_path)
        else:
            if json_config_prev_abspath != '' and growth_csv_abs_path != '' and allometry_csv_abs_path != '':
                json_config_abspath = json_config_prev_abspath
                print('json sucessfully conf added')
            else:
                raise ValueError('error with json setup')

        # class based implementation update
        ex_ante = ExAnteCalc(
            json_path_config=json_config_abspath,
            first_run=False,
            re_select_growth_data=re_select_growth_data,  # reselect the growth data from the csv exist or first download and change csv location
            download_csv=download_csv,
            growth_csv=growth_csv_abs_path,
            allometry_csv=allometry_csv_abs_path,
            name_column_species_allo=name_column_species_allo,  #'Lat. Name',  # hard-coded, revisit later, or you can put in above recalc_ex_ante method as an arguments
            name_column_species_growth=name_column_species_growth,  #'Tree Species(+origin of allom. formula)', # hardcoded, revisit later
            override_num_trees_0=self.override_num_trees_0,
            mortality_csu_df=csu_species_mortality_check,
            override_avg_tree_perha=override_avg_tree_perha
        )

        ### variable input for the scenario
        # accessing the general input variable for the previous version of the ex-ante (before update)
        location_general_var = os.path.join(
            module_path, self.config["prev_exante_config"]
        )
        import json

        with open(location_general_var, "r") as json_general:
            conf_general = json.load(json_general)
            # conf_general is dictionary of prev. ex ante conf

        print("prev ex ante conf: \n", conf_general)

        self.prev_config_exante = conf_general

        # project_name_update = 't4t_updated_expost_20241010'

        # because there is no tco2e in this expost, therefore, planting year we will delayed one year (ex-ante always expect tco2e at year 1)
        if all_tree_evidence == True:
            # we will adjust and treat that the tree evidence in annual monitoring as they are just planted, therefore planting year started at max. monitoring year
            # conf_general["planting_year"] = int(
            #     self.validated_df["monitoring_date"].max()[:4]
            # )  # make sure the data has monitoring date
            conf_general["planting_year"] = conf_general["planting_year"]  # decided not to use this approach because it will confuse the data. planting year should be the same

        else:
            conf_general["planting_year"] = conf_general["planting_year"]

        # change the name of project
        project_name_prev = conf_general["project_name"]
        conf_general["project_name"] = project_name_update
        project_name = project_name_update

        # change the gap_harvest to be false as default for the updated expost
        # gap_harvest_prev = conf_general["gap_harvest"]
        conf_general["gap_harvest"] = gap_harvest

        # harvesting_max_percent_prev = conf_general["harvesting_max_percent"]
        conf_general["harvesting_max_percent"] = harvesting_max_percent

        # change the thinning_stop to be True as default for the updated expost
        # thinning_stop_prev = conf_general["thinning_stop"]
        conf_general["thinning_stop"] = thinning_stop

        # trace the old folder ex-ante
        root_folder_prev = os.path.join(module_path, conf_general["root_folder"])

        # change the location root_folder
        conf_general["root_folder"] = self.output_dir_exante
        root_folder = self.output_dir_exante

        if override_planting_year !='':
            conf_general['planting_year'] = int(override_planting_year) # need to ensure that this is int

        # now check the update
        print("updated ---> ", conf_general)

        # write in the output_dir_exante for the t4t_main.json
        json_path_new_config = os.path.join(
            self.output_dir_exante, f"{project_name}_main.json"
        )
        with open(json_path_new_config, "w") as json_file:
            json.dump(conf_general, json_file, indent=4)

        self.conf_general_updated = conf_general

        # now we will write the plot_summary_csv
        gdrive_location_seedling = os.path.join(
            root_folder, f"{project_name}_distribution_trees_seedling.csv"
        )
        self.merge_plot_species_cleaned.to_csv(gdrive_location_seedling)
        self.seedling_distribution_updated = self.merge_plot_species_cleaned

        if force_load_seedling_csv != "":
            # gdrive_location_seedling = force_load_seedling_csv
            self.seedling_distribution_updated = cleaning_csv_df(
                pd.read_csv(force_load_seedling_csv)
            )
            self.seedling_distribution_updated.to_csv(gdrive_location_seedling)

        # it needs to clean first in order to avoid error in ex-ante (some data has all 0 expost)

        print("copying the scenario from", root_folder_prev)
        # checking the formula used in previous ex ante
        old_df_tree_selected = os.path.join(
            root_folder_prev, f"{project_name_prev}_formulas_allometry.csv"
        )
        gdrive_location_df_tree_selected = os.path.join(
            root_folder, f"{project_name}_formulas_allometry.csv"
        )

        old_df_tree_selected = pd.read_csv(old_df_tree_selected)
        # Drop columns that start with 'Unnamed_'
        old_df_tree_selected = old_df_tree_selected.loc[
            :, ~old_df_tree_selected.columns.str.startswith("Unnamed")
        ]
        # display(old_df_tree_selected)
        self.df_tree_selected_prev = old_df_tree_selected

        # need to match the number of species selected of old prev ex-ante and new species in terms of species formula selection (allometry)
        # list the species name to match with df
        list_column_species_coredb = list(
            merge_plot_species.filter(like="_num_trees").columns
        )
        list_column_species_coredb = [
            l.replace("_num_trees", "") for l in list_column_species_coredb
        ]
        print(list_column_species_coredb)
        ############################################

        new_df_selected = old_df_tree_selected[
            old_df_tree_selected[ex_ante.name_column_species_allo].isin(
                list_column_species_coredb
            )
        ]
        list_existing_species = [
            l
            for l in list_column_species_coredb
            if l in new_df_selected[ex_ante.name_column_species_allo].unique()
        ]

        # now we need to identify which species is not in the new expost
        new_species_to_be_added = [
            l
            for l in list_column_species_coredb
            if l not in new_df_selected[ex_ante.name_column_species_allo].unique()
        ]
        print("new species need to be added: ", new_species_to_be_added)
        # now we will avoid the dbh and height based : durian case has two type allometry
        allometry_df_to_add = ex_ante.allometry_df[
            (
                ex_ante.allometry_df[ex_ante.name_column_species_allo].isin(
                    new_species_to_be_added
                )
            )
            & (
                ~ex_ante.allometry_df["Allometric Formula, Type"].eq(
                    "DBH & Height based"
                )
            )  # remember this column name hardcoded, may change later
        ]

        # list_columns_csv_coredb = ex_ante.allometry_df.columns
        # list_columns_csv_df_selected = old_df_tree_selected.columns
        # common_columns = set(list_columns_csv_coredb.columns).intersection(set(list_columns_csv_df_selected.columns))
        list_columns_updated = [
            name_column_species_allo,
            "Country of Use",
            "Allometric Formula, Type",
            "TTB formula, tdm",
            "WD variable",
        ]  # hardcoded, revisit later

        # reupdate the formula if necessary from the last formula we used in previous ex-ante setting
        if re_update_allometry_model == True:
            new_df_selected = pd.merge(
                new_df_selected,
                ex_ante.allometry_df[list_columns_updated],
                on=[name_column_species_allo],
                how="left",
                suffixes=("_prev", "_updated"),
            )

            list_columns_prev = [
                col for col in new_df_selected.columns if col.endswith(("_prev"))
            ]
            list_columns_updated = [
                col for col in new_df_selected.columns if col.endswith(("_updated"))
            ]
            list_columns_restored = [
                col.replace("_prev", "")
                for col in new_df_selected.columns
                if col.endswith(("_prev"))
            ]
            new_df_selected = new_df_selected.drop(list_columns_prev, axis=1)
            new_df_selected = new_df_selected.rename(
                columns=dict(zip(list_columns_updated, list_columns_restored))
            )

        # add to the df selected
        concat_df = pd.concat([new_df_selected, allometry_df_to_add])
        # to avoid override the zone (prev. code)
        # step -1 melt and only add relevant column names
        df_long = merge_plot_species[
            ["Plot_ID", "zone"]
            + [f for f in merge_plot_species.columns if "num_trees" in f]
        ].melt(
            id_vars=["Plot_ID", "zone"], var_name="species_full", value_name="num_trees"
        )

        # Step 2: Extract species name by removing '_num_trees' suffix
        df_long[ex_ante.name_column_species_allo] = df_long["species_full"].str.replace(
            "_num_trees", "", regex=False
        )

        # Step 3: Filter out rows where num_trees is zero
        df_long = df_long[df_long["num_trees"] > 0]

        summary_df_species_zone = (
            df_long[[ex_ante.name_column_species_allo, "zone"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        # display(concat_df)

        # merge joined the unique formula and each species used formula on zones,
        concat_df = pd.merge(
            concat_df,
            summary_df_species_zone,
            left_on=ex_ante.name_column_species_allo,
            right_on=ex_ante.name_column_species_allo,
            how="right",
            suffixes=("_concated", "_from_plot"),
        )

        concat_df = concat_df.drop(columns=["zone_concated"]).rename(
            columns={"zone_from_plot": "zone"}
        )

        # self.df_tree_selected_prev = ex_ante.df_tree_selected

        # Assuming your data is in a DataFrame called `df`
        concat_df = concat_df.drop_duplicates()

        # If you want to reset the index afterward
        concat_df.reset_index(drop=True, inplace=True)

        self.df_tree_selected_updated = concat_df

        concat_df.to_csv(gdrive_location_df_tree_selected)

        # put the force reading the manual df (e.g if we want to force to specific formulas)
        if override_new_formula != "":
            gdrive_location_df_tree_selected = override_new_formula
            self.df_tree_selected_updated = cleaning_csv_df(
                pd.read_csv(gdrive_location_df_tree_selected)
            )
            # no need to do .to_csv because the path is already there

        ################################

        #######################
        # do the same with growth model selected
        prev_growth_selected = os.path.join(
            root_folder_prev, f"{project_name_prev}_selected_growth_model.csv"
        )
        gdrive_growth_selected = os.path.join(
            root_folder, f"{project_name}_selected_growth_model.csv"
        )
        prev_growth_selected = pd.read_csv(prev_growth_selected)
        prev_growth_selected = prev_growth_selected.loc[
            :, ~prev_growth_selected.columns.str.startswith("Unnamed")
        ]  # for cleaning the csv there is Unnamed*
        self.growth_selected_prev = prev_growth_selected

        # acquire the growth db from coredb
        growth_df = ex_ante.growth_df
        prev_growth_fix_re_select = ex_ante.growth_selected
        # first remove if there are trees that no longer in expost new_growth_df is clean version
        new_growth_df = prev_growth_fix_re_select[
            prev_growth_fix_re_select[ex_ante.name_column_species_growth].isin(
                list_column_species_coredb
            )
        ]
        growth_df_to_add = growth_df[
            growth_df[ex_ante.name_column_species_growth].isin(new_species_to_be_added)
        ]

        # add to the df selected
        concat_df_growth = pd.concat([new_growth_df, growth_df_to_add])
        self.growth_selected_updated = concat_df_growth
        if all_tree_evidence != True:
            if sigmoid_remodel_growth == True and re_select_growth_data == False:
                new_growth_calc = self.analyze_growth()
                new_growth_calc = new_growth_calc[
                    [name_column_species_growth, "year", "Height", "sigmoid_dbh_cm"]
                ]
                new_growth_calc = new_growth_calc.rename(
                    columns={"sigmoid_dbh_cm": "DBH"}
                )
                merge_growth_updated_func = pd.merge(
                    new_growth_calc,
                    new_growth_df,
                    on=[ex_ante.name_column_species_growth, "year"],
                    how="left",
                    suffixes=("", "_old"),
                )
                concat_df_growth = pd.concat(
                    [merge_growth_updated_func, growth_df_to_add]
                )
                self.growth_selected_updated = concat_df_growth

        # concat_df_growth.to_csv(gdrive_growth_selected)

        if override_new_growth_model != "":
            self.growth_selected_updated = cleaning_csv_df(
                pd.read_csv(override_new_growth_model)
            )
            # gdrive_growth_selected = override_new_growth_model

        self.growth_selected_updated.to_csv(gdrive_growth_selected)

        # assume the scenario with the following condition --> other scenario still the same
        old_scenario_exante_path = os.path.join(
            root_folder_prev, f"{project_name_prev}_forestry_scenario.json"
        )
        with open(old_scenario_exante_path, "r") as scenario_json:
            old_scenario_exante = json.load(scenario_json)

        self.input_scenario_prev_path = old_scenario_exante_path
        old_scenario_exante_toedit = old_scenario_exante

        if 'non_replanting' in old_scenario_exante_toedit.keys():
            old_scenario_exante_toedit = old_scenario_exante_toedit

        else:
            old_scenario_exante_toedit['non_replanting'] = old_scenario_exante_toedit

        # look at the number of species in old scenario vs current expost species
        # ensure all the species in the species selected updated
        # old_scenario_exante = expost.input_scenario_prev

        # list_column_species_coredb = list(merge_plot_species.filter(like='_num_trees').columns)
        # list_column_species_coredb = [l.replace('_num_trees','') for l in list_column_species_coredb]
        # # print(list_column_species_coredb)

        old_df_tree_selected = self.df_tree_selected_prev

        new_df_selected = old_df_tree_selected[
            old_df_tree_selected[ex_ante.name_column_species_allo].isin(
                list_column_species_coredb
            )
        ]

        # new_species_to_be_added = [l for l in list_column_species_coredb if l not in new_df_selected['Lat. Name'].unique()]

        new_dict_species_zone = (
            concat_df.groupby("zone")[ex_ante.name_column_species_allo]
            .apply(list)
            .to_dict()
        )
        # new_dict_species_zone

        dict_zone_species = {
            k: [species_list for species_list in v.keys()]
            for k, v in old_scenario_exante_toedit.items()
        }

        # Find differences
        new_data = {}
        for zone, species_list in new_dict_species_zone.items():
            crosscheck_list = dict_zone_species.get(
                zone, []
            )  # Get corresponding list from dict2
            new_data[zone] = [
                species for species in species_list if species not in crosscheck_list
            ]

        # Filter out empty zones (optional)
        new_data = {zone: species for zone, species in new_data.items() if species}

        new_species_to_be_added_zone = new_data

        # Find the path to the original, correct scenario file
        old_scenario_exante_path = os.path.join(
            root_folder_prev, f"{project_name_prev}_forestry_scenario.json"
        )
        
        # ... more code ...

        if override_new_scenario == "":
            gdrive_location_scenario_rate = os.path.join(
                root_folder, f"{project_name}_forestry_scenario.json"
            )

            # all_scenario_exante_toedit
            concat_df = self.df_tree_selected_updated

            # Find differences
            new_dict_species_zone = (
                        concat_df.groupby("zone")[name_column_species_allo]
                        .apply(list)
                        .to_dict()
                    )
            # new_dict_species_zone

            dict_zone_species = {
                k: [species_list for species_list in v.keys()]
                for k, v in old_scenario_exante_toedit.items()
            }

            new_data = {}
            for zone, species_list in new_dict_species_zone.items():
                crosscheck_list = dict_zone_species.get(
                    zone, []
                )  # Get corresponding list from dict2
                new_data[zone] = [
                    species for species in species_list if species not in crosscheck_list
                ]

            new_data = {zone: species for zone, species in new_data.items() if species}

            new_species_to_be_added_zone = new_data


            all_scenario = {}
    
            # These are the only keys that will be processed as planting zones
            VALID_ZONES = ["production_zone", "protected_zone"]
            print('old_scenario_exante_toedit :',old_scenario_exante_toedit)
            print('update_species_name :', update_species_name)

            for is_replanting, zone_scenario in old_scenario_exante_toedit.items():
                
                updated_scenario = {}

                # 1. Process existing species from the old scenario
                for zone in VALID_ZONES:
                    if zone in zone_scenario:
                        species_scenario_map = zone_scenario[zone]
                        updated_scenario[zone] = {}

                        for species, scenario in species_scenario_map.items():
                            # Check if the species is still considered valid
                            if species in concat_df[concat_df["zone"] == zone]["Lat. Name"].to_list():
                                # Copy all parameters from the old scenario by default
                                updated_single_scenario = scenario.copy()
                                # Override any specific values
                                updated_single_scenario["mortality_percent"] = adding_prev_mortality_rate
                                updated_scenario[zone][species] = updated_single_scenario

                # 2. Add any new species with the special template-finding logic
                if new_species_to_be_added_zone:
                    for zone, new_species_list in new_species_to_be_added_zone.items():
                        if zone not in VALID_ZONES:
                            continue
                        if zone not in updated_scenario:
                            updated_scenario[zone] = {}

                        for new_species in new_species_list:
                            print('new_species :',new_species)
                            base_scenario = None
                            # # Extract the first word of the new species name as the keyword
                            # new_species_keyword = new_species.split(' ')[0]

                            # # A. Try to find a template in the current zone using the keyword
                            # if zone in zone_scenario:
                            #     for existing_species, existing_scenario in zone_scenario[zone].items():
                            #         if existing_species.startswith(new_species_keyword):
                            #             base_scenario = existing_scenario.copy()
                            #             break  # Found the best template, stop searching


                            # A. the same exact name
                            if zone in zone_scenario:
                                for existing_species, existing_scenario in zone_scenario[zone].items():
                                    if existing_species == new_species:
                                        base_scenario = existing_scenario.copy()
                                        break  # Found the best template, stop searching

                            # A1 . Try template using the existing dictionary key, value (if there is changing code of species code) within the same species due to e.g allometry formula, or growth model changes
                            if zone in zone_scenario:
                                for existing_species, existing_scenario in zone_scenario[zone].items():
                                    if update_species_name.get(existing_species) == new_species:
                                        base_scenario = existing_scenario.copy()
                                        break

                            # B . Before we do iter, that picking on the first item of base_scenario, let use all the possibility in the replanting and non_replanting
                            if base_scenario is None:
                                for other_zone_scenario in old_scenario_exante_toedit.values():
                                    if zone in other_zone_scenario:
                                        for existing_species, existing_scenario in other_zone_scenario[zone].items():
                                            if (existing_species == new_species or 
                                                update_species_name.get(existing_species) == new_species):
                                                base_scenario = existing_scenario.copy()
                                                break
                                    if base_scenario:
                                        break


                            # # B1. If no match at all, fall back to the first available species
                            if base_scenario is None and zone in zone_scenario and zone_scenario[zone]:
                                base_scenario = next(iter(zone_scenario[zone].values()), {}).copy()
                            
                            # C. If still no template, use an empty dictionary
                            if base_scenario is None:
                                base_scenario = {}
                            
                            #####################################
                            # Create the new scenario, which inherits all keys from the template <--- we can manouver this section later
                            print('base_scenario: ',base_scenario)
                            new_scenario = base_scenario
                            if is_replanting == 'non_replanting':
                                # Override specific values
                                new_scenario["mortality_percent"] = adding_prev_mortality_rate

                            else: # create logic, separate the mort. existing previous trees, and newly replanting trees
                                new_scenario['mortality_percent'] = override_mortality_replanting

                            if override_natural_thinning!='':
                                new_scenario['natural_thinning'] = override_natural_thinning
                            
                            # Set harvesting_year, using the template's value or a default
                            if zone == "protected_zone":
                                new_scenario["harvesting_year"] = base_scenario.get("harvesting_year", 30)
                            elif zone == "production_zone":
                                new_scenario["harvesting_year"] = base_scenario.get("harvesting_year", 10)

                            ######################################
                            
                            updated_scenario[zone][new_species] = new_scenario

                all_scenario[is_replanting] = updated_scenario

            # all_scenario = process_scenarios(old_scenario_exante_toedit, concat_df, new_species_to_be_added_zone, 
            #                 adding_prev_mortality_rate=adding_prev_mortality_rate, update_species_name = update_species_name, override_mortality_replanting=override_mortality_replanting)
            updated_scenario = all_scenario # all_scnario is fixing the bug earlier

            with open(gdrive_location_scenario_rate, "w") as scenario_json:
                json.dump(all_scenario, scenario_json, indent=4)

        else:  # if we want to override the scenario with the manual input path in override_new_scenario
            gdrive_location_scenario_rate = override_new_scenario
            with open(override_new_scenario, "r") as json_scenario:
                updated_scenario = json.load(json_scenario)

            saved_manual_scenario = os.path.join(
                root_folder, os.path.basename(override_new_scenario)
            )
            with open(saved_manual_scenario, "w") as scenario_json:
                json.dump(updated_scenario, scenario_json, indent=4)

        print(updated_scenario)
        self.input_scenario_updated = updated_scenario

        gdrive_raw_output = os.path.join(
            root_folder, f"{project_name}_OUTPUT_RAW_CALCULATION.csv"
        )
        gdrive_input_cs = os.path.join(
            root_folder, f"{project_name}_input_gcs_generated.csv"
        )
        # not used yet
        gdrive_plot_csu = os.path.join(root_folder, f"{project_name}_csu_updated.shp")

        # UPDATING THE CLASS OBJECT INSTANCE ---> re-read the csv and configuration to run the exante and UPDATE
        ex_ante.gdrive_location_df_tree_selected = gdrive_location_df_tree_selected
        ex_ante.gdrive_growth_selected = gdrive_growth_selected
        ex_ante.gdrive_location_scenario_rate = gdrive_location_scenario_rate
        ex_ante.gdrive_location_seedling = gdrive_location_seedling
        ex_ante.gdrive_raw_output = gdrive_raw_output
        ex_ante.root_folder = root_folder
        ex_ante.gdrive_input_cs = gdrive_input_cs
        ex_ante.gdrive_plot_csu = gdrive_plot_csu

        ex_ante.json_path_config = json_path_new_config
        ex_ante.df_tree_selected = cleaning_csv_df(
            pd.read_csv(gdrive_location_df_tree_selected)
        )
        ex_ante.growth_selected = cleaning_csv_df(pd.read_csv(gdrive_growth_selected))
        ex_ante.input_scenario_species = self.input_scenario_updated
        ex_ante.unique_species_selected = ex_ante.df_tree_selected[
            ex_ante.name_column_species_allo
        ].unique()
        ex_ante.df_unique_species = ex_ante.df_tree_selected.drop(
            columns=["zone"]
        ).drop_duplicates()
        ex_ante.co2_data_dict = ex_ante._co2_data(
            ex_ante.override_tco2_series(
                ex_ante.df_unique_species, ex_ante.growth_selected
            ),
            ex_ante.name_column_species_allo,
        )

        ex_ante.df_tco2_selected = ex_ante.get_selected_co2(
            ex_ante.co2_data_dict, ex_ante.unique_species_selected
        )

        ex_ante.conf_general = self.conf_general_updated
        ex_ante.config = ex_ante.conf_general

        # self.input_scenario_prev_path
        with open(self.input_scenario_prev_path, "r") as scenario_json:
            old_scenario_exante = json.load(scenario_json)

        self.input_scenario_prev = old_scenario_exante
        # self.input_scenario_prev

        # ex_ante.csi_plot_model()
        # self.input_scenario_prev = old_scenario_exante #remove this part, it has a glitch, updated the mortality percent above (unexpected)
        self.updated_exante = ex_ante
