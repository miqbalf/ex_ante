import io
import json
import os
import pprint
from pickle import FALSE
from typing import List, Union

import numpy as np
import pandas as pd
# from cooling.src.class_cooling import Cooling, CSink
from dotenv import load_dotenv
from ex_ante.coredb_trees.allometry_formulas import AllometryFormulaDB
from ex_ante.coredb_trees.growth_models import GrowthModelSpecies
from ex_ante.csi_tree.input_cooling_creation import input_cooling
from ex_ante.csi_tree.main import CSIExante
from ex_ante.plot.utils import calc_plot_csu
from ex_ante.population_tco2.main import num_tco_years
from ex_ante.ui.main import CSUEntryForm, Project_Setting_Species, SelectingScenario
from ex_ante.ui.utils import is_running_in_colab
from ex_ante.utils.calc_formula_string import calc_biomass_formula
from ex_ante.utils.helper import adding_zero_meas, cleaning_csv_df
from ex_ante.utils.max_density import get_max_density
from IPython.display import display, HTML
from ipywidgets import interact, interact_manual, interactive, widgets

# Load the .env file
load_dotenv()

# think like this is django setting, we will map the relative path of default location of input_db of dbh and allometry formula
# path obtained relative to this file
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

growth_csv_abs_path = os.path.join(
    current_directory, "ex_ante/00_input/growth_model_2024-08-13.csv"
)  # change this accordingly, and it can be as part of the argument
allometry_csv_abs_path = os.path.join(
    current_directory, "ex_ante/00_input/allometry_model_2024-08-13.csv"
)

import sys

def is_colab():
    """Detects if running in Google Colab"""
    return 'google.colab' in sys.modules

# Initialize Colab if needed
if is_colab():
    from google.colab import output
    output.enable_custom_widget_manager()

class AllometryLibrary(AllometryFormulaDB, GrowthModelSpecies):

    # _list_allometry_formulas = {
    #     "sengon": ,
    #     "pine": []
    # }

    allometry_df: pd.DataFrame
    growth_df: pd.DataFrame

    def __init__(
        self,
        link_growth: str = os.getenv("link_growth_model_data_csv"),
        download_csv: bool = False,  # check if you want to download first the csv or use existing one from the path below
        growth_csv: str = os.getenv(
            "path_local_growth_model_data_csv"
        ),  # we should pay attention on which file we use for the growth
        name_column_species_allo: str = "Lat. Name",
        name_column_species_growth: str = "Tree Species(+origin of allom. formula)",
        allometry_csv: str = allometry_csv_abs_path,  # we should pay attention on which file we use for the allometry
        link_allometry: str = os.getenv("link_allometry_csv_gsheet"),
        allo_formula_column_name="TTB formula, tdm",
        conversion_tco2=44 / 12,
    ):
        # general action selection
        # self.download_csv = download_csv

        # # growth model instance
        # self.link_growth = link_growth
        # self.growth_csv = growth_csv
        self.name_column_species_growth = name_column_species_growth

        # # allometry formulas instance
        # self.link_allometry = link_allometry
        self.name_column_species_allo = name_column_species_allo
        self.conversion_tco2 = conversion_tco2

        # self.allometry_csv = allometry_csv

        # self.allo_formula_column_name = allo_formula_column_name

        # Call the constructors of both parent classes
        AllometryFormulaDB.__init__(
            self,
            link_allometry=link_allometry,
            download_csv=download_csv,
            allometry_csv=allometry_csv,
            name_column_species=self.name_column_species_allo,
            allo_formula_column_name=allo_formula_column_name,
        )
        # put in the first instance
        self.allometry_df = self.restructuring_allometric_data()

        GrowthModelSpecies.__init__(
            self,
            link_growth=link_growth,
            download_csv=download_csv,
            growth_csv=growth_csv,
            name_column_species=self.name_column_species_growth,
        )

        self.growth_df = self.restructure_growth_data()

        # check the unique list of species based on allometry list species, and growth model, and consider all the species name
        # first we clean the list only if allometry has the formula text, not na
        cleaned_na_allometry = self.allometry_df.dropna(
            subset=[self.allo_formula_column_name]
        )
        # # now we will get this unique list of the species name
        self.species_unique_allo_db = list(
            cleaned_na_allometry[name_column_species_allo].unique()
        )
        # # second, we clean the df first if the dbh data is only give you 0, here DBH column name is still hard-coded
        cleaned_0_dbh = self.growth_df[self.growth_df["DBH"] != 0]
        self.species_unique_growth_db = list(
            cleaned_0_dbh[name_column_species_growth].unique()
        )

    # this method will clean the available data, ie. if data does not have growth, and
    def list_all_species(self) -> list:
        print(f"num species based on allometry data {len(self.species_unique_allo_db)}")
        print(f"num species based on growth data {len(self.species_unique_growth_db)}")
        print("merging allometry + growth db species into unique")

        species_unique_list = list(
            set(self.species_unique_allo_db + self.species_unique_growth_db)
        )
        print(f"num species unique based on all data {len(species_unique_list)}")

        return species_unique_list

    # this methods will pair the information of the same name species that exist in allometry and growth model db, if they are available
    def default_pair_allometry_growth(self) -> pd.DataFrame:
        conversion_tco2 = self.conversion_tco2
        list_all_species = self.list_all_species()
        not_in_allodb_species = [
            species
            for species in list_all_species
            if species not in self.species_unique_allo_db
        ]
        print(
            f"here are the list of species does not have the info of allometry formula, \
              \n but has the info of growth data: \n  {not_in_allodb_species}"
        )
        print(
            f"total number of species for that are {len(not_in_allodb_species)}\n----------------------------------"
        )

        not_in_growthdb_species = [
            species
            for species in list_all_species
            if species not in self.species_unique_growth_db
        ]
        print(
            f"here are the list of species does not have the info of growth model, \
              \n but has the info of allometry formula: \n  {not_in_growthdb_species}"
        )
        print(
            f"total number of species for that are {len(not_in_growthdb_species)}\n----------------------------------"
        )

        list_species_available_pair = [
            species
            for species in list_all_species
            if (species not in not_in_allodb_species)
            and (species not in not_in_growthdb_species)
        ]

        print(
            'now we will remove the NA species if they don"t have either allometry or growth'
        )
        print(f"total number species clean to pair: {len(list_species_available_pair)}")
        print(list_species_available_pair)

        biomass_per_tree_year = pd.merge(
            self.allometry_df[
                self.allometry_df[self.name_column_species_allo].isin(
                    list_species_available_pair
                )
            ],
            self.growth_df[
                self.growth_df[self.name_column_species_growth].isin(
                    list_species_available_pair
                )
            ],
            left_on=self.name_column_species_allo,
            right_on=self.name_column_species_growth,
            how="inner",
        )

        biomass_per_tree_year["DBH"] = (
            biomass_per_tree_year["DBH"].astype(float).fillna(0)
        )

        # these columnns still hard-coded, if someone someday change the name of these column before you download the new csv, we need to change it.
        # since probably we will use database, let's ignore for the moment
        biomass_per_tree_year["TTB_value_pertree_ton"] = biomass_per_tree_year.apply(
            lambda x: calc_biomass_formula(
                x[self.allo_formula_column_name],
                x["WD variable"],
                x["DBH"],
                x["Height"],
            ),
            axis=1,
        )

        # we still need to clean up the pair biomass = allometry(growth xi) because, not only DBH data that is consider as growth,
        # other data such as Height also may not available yet. e.g Durio zibethinus | Durian - IDN 	Indonesia 	DBH & Height based
        # now we will just drop NA from the result, to make it clean
        biomass_per_tree_year = biomass_per_tree_year.dropna(
            subset=["TTB_value_pertree_ton"]
        )
        # not only na, 0 value is also put into the gsheet such as wd, height, dbh that is actually null,
        # :( we also need to remove them
        biomass_per_tree_year = biomass_per_tree_year[
            biomass_per_tree_year["TTB_value_pertree_ton"] != 0
        ]

        tco2_per_tree_year = biomass_per_tree_year.copy()
        tco2_per_tree_year["tco2_value_pertree"] = (
            tco2_per_tree_year["TTB_value_pertree_ton"] * 0.47 * (conversion_tco2)
        )

        return tco2_per_tree_year

    # instead of getting biomas per year (prev. jupyter), now we will focus on the tc20 per year series in this method
    def override_tco2_series(self, df_allometry, df_growth) -> pd.DataFrame:
        conversion_tco2 = self.conversion_tco2
        biomass_per_tree_year = pd.merge(
            df_allometry,
            df_growth,
            left_on=self.name_column_species_allo,
            right_on=self.name_column_species_growth,
            how="inner",
        )

        biomass_per_tree_year["DBH"] = (
            biomass_per_tree_year["DBH"].astype(float).fillna(0)
        )
        biomass_per_tree_year["TTB_value_pertree_ton"] = biomass_per_tree_year.apply(
            lambda x: calc_biomass_formula(
                x[self.allo_formula_column_name],
                x["WD variable"],
                x["DBH"],
                x["Height"],
            ),
            axis=1,
        )

        biomass_per_tree_year = biomass_per_tree_year.dropna(
            subset=["TTB_value_pertree_ton"]
        )
        biomass_per_tree_year = biomass_per_tree_year[
            biomass_per_tree_year["TTB_value_pertree_ton"] != 0
        ]
        tco2_per_tree_year = biomass_per_tree_year.copy()
        tco2_per_tree_year["tco2_value_pertree"] = (
            tco2_per_tree_year["TTB_value_pertree_ton"] * 0.47 * (conversion_tco2)
        )

        return tco2_per_tree_year

    @staticmethod
    def _co2_data(tco2_df: pd.DataFrame, name_column_species_allo: str):
        # convert the df and pair in the dictionary, such as {species: [list_carbon_start_from_year_1]}
        _co2_data = (
            tco2_df.groupby(name_column_species_allo)["tco2_value_pertree"]
            .apply(list)
            .to_dict()
        )
        _co2_data = {key: [0] + values for key, values in _co2_data.items()}
        return _co2_data

    @staticmethod
    def add_species(co2_data_dict: dict, tree_species: str, co2_data: list):
        if tree_species in co2_data_dict:
            raise ValueError(f"Species '{tree_species}' already exists.")
        co2_data_dict[tree_species] = co2_data

    @staticmethod
    def get_co2(co2_data_dict: dict, tree_species: str, year: int) -> float:
        try:
            return co2_data_dict[tree_species][year]
        except KeyError:
            raise ValueError(f"Tree species '{tree_species}' not found.")
        except IndexError:
            raise ValueError(
                f"Year '{year}' is out of range for species '{tree_species}'."
            )

    @staticmethod
    def get_selected_co2(
        co2_data_dict: dict,
        list_species_selected: Union[List, np.ndarray],
    ) -> pd.DataFrame:
        dict_species_tco2 = {
            "species": [k for k in list_species_selected],
            "tco2_per_tree": [co2_data_dict[k] for k in list_species_selected],
        }
        # Creating a DataFrame
        df_list = []
        for species, tco2_values in zip(
            dict_species_tco2["species"], dict_species_tco2["tco2_per_tree"]
        ):
            # Create a temporary DataFrame for each species with an index from 0 to length of tco2 values
            temp_df = pd.DataFrame(
                {
                    "species": species,
                    "year": list(range(len(tco2_values))),
                    "tco2_per_tree": tco2_values,
                }
            )
            df_list.append(temp_df)

        # Concatenate all the temporary DataFrames into one
        df_tco2_selected = pd.concat(df_list, ignore_index=True)
        df_tco2_selected = df_tco2_selected.drop(
            df_tco2_selected[df_tco2_selected["year"] == 0].index
        )  # drop year 0
        return df_tco2_selected


class Tree:
    planted: int
    tree_species: str
    library: AllometryLibrary

    def __init__(self, library, co2_data_dict: dict, planted, tree_species: str):
        self.planted = planted  # planting year started
        self.tree_species = tree_species
        self.library = library
        self.co2_data_dict = co2_data_dict

    def get_co2(self, current_year: int):
        year = current_year - self.planted
        if year < 0:
            raise ValueError(
                f"Year {current_year} is less than the planted year {self.planted}"
            )
        return self.library.get_co2(self.co2_data_dict, self.tree_species, year)

    def __str__(self):
        return f"{self.tree_species}, planted: {self.planted}"


class Plot:
    # Ideas for improvement:
    # - keep the list of trees for every year (make a copy of year -1 and apply changes)
    # - list of trees can be also a dataframe to make operations easier (e.g. apply changes to specific species)
    area_ha: float
    current_year: int
    trees: {}

    def __init__(self, area_ha=0, csv_plot=""):
        self.area_ha = area_ha
        self.current_year = 1
        self.trees = set()
        self.csv_plot = csv_plot
        if csv_plot == "":
            print("no csv file for the seedling location")
        else:
            print("now we will choose the csv file")
            None

    def csu_distribution(self) -> dict:
        return calc_plot_csu(self.csv_plot)

    # average scenario on getting number trees based on trees per ha
    def plant_trees(self, tree_species: str, trees_per_ha: int):
        total_trees = trees_per_ha * self.area_ha
        for i in range(total_trees):
            tree = Tree(self.current_year, tree_species)
            self.trees.add(tree)

    def add_years(self, years: int):
        self.current_year += years

    def get_year(self):
        return self.current_year

    def get_co2(self):
        total_co2 = 0.0
        for tree in self.trees:
            total_co2 += tree.get_co2(self.current_year)
        return total_co2

    def remove_trees(self, remove_share: float):
        trees_to_remove = round(remove_share * len(self.trees))
        index = 0
        for tree in set(self.trees):
            if index < trees_to_remove:
                self.trees.remove(tree)
            index += 1


# wrapping all the notebook into single class
class ExAnteCalc(AllometryLibrary):
    # first_run is required, if you started the first model and want to create folder of configuration, set True, otherwise set config_location
    # option 1 provide all json_path_config,
    # option 2 give parent_folder, config_folder_name, and project_name later for if running first_run as True
    def __init__(
        self,
        first_run: bool,
        json_path_config="",
        parent_folder="",
        config_folder_name="",
        project_name="",
        download_csv=False,
        link_allometry: str = os.getenv("link_allometry_csv_gsheet"),
        link_growth: str = os.getenv("link_growth_model_data_csv"),
        growth_csv=growth_csv_abs_path,
        allometry_csv=allometry_csv_abs_path,
        name_column_species_allo="Lat. Name",
        name_column_species_growth="Tree Species(+origin of allom. formula)",
        re_select_growth_data=False,
        conversion_tco2=44 / 12,
        override_num_trees_0=False,  # by default there is no override for num_trees_0 because this is for initial ex-ante,
        # but for updated ex-ante we should put to True if you want to visualize in num_trees_over_years, also with this following argument its mortality df
        mortality_csu_df=pd.DataFrame(),  # default as empty data Frame
    ):
        self.growth_csv = growth_csv
        self.allometry_csv = allometry_csv
        self.download_csv = download_csv
        self.link_growth = link_growth
        self.link_allometry = link_allometry
        self.name_column_species_allo = name_column_species_allo
        self.name_column_species_growth = name_column_species_growth

        self.first_run = first_run
        self.df_selected = None
        self.df_tree_selected = None
        self.project_name = project_name
        self.json_path_config = json_path_config
        self.config_folder_name = config_folder_name
        self.parent_folder = parent_folder
        self.re_select_growth_data = re_select_growth_data

        # self.selection_complete_event = asyncio.Event()  # Event to signal selection completion
        self.selection_complete = False
        self.conversion_tco2 = conversion_tco2
        self.override_num_trees_0 = override_num_trees_0
        self.mortality_csu_df = mortality_csu_df

        if self.parent_folder == "":
            self.parent_folder = "00_ex_ante_result"
        else:
            self.parent_folder = parent_folder

        # this is directory root
        self.config_location = os.path.join(
            current_directory, self.parent_folder, self.config_folder_name
        )
        self.root_folder = self.config_location
        root_folder = self.config_location

        # NOW TO GET THE MAIN_JSON
        if self.project_name != "" and self.json_path_config == "":
            self.json_path_config = os.path.join(
                self.config_location, f"{project_name}_main.json"
            )

        # let's create the default location if we don't specify json_path_config as config_folder_name = project_name
        elif self.config_folder_name != "" and json_path_config == "":
            self.json_path_config = os.path.join(
                root_folder, f"{self.config_folder_name}_main.json"
            )

        else:
            self.json_path_config = self.json_path_config
            self.config_location = os.path.dirname(self.json_path_config)
            self.root_folder = self.config_location
            root_folder = self.config_location

        if self.first_run == True:
            print("create a new project!")
            project_name = input("What is the project name? ")
            # to be implemented later
            csv_plot_summary_zone = True  # as default for now
            duration_project = int(
                input(
                    "How long the project will last? now its 30 year using coredb and updated to 35, enter number only"
                )
            )
            # gap_harvest
            gap_harvest = input("gap_harvest (yes/no)")
            if gap_harvest.lower() in ["true", "yes", "1"]:
                gap_harvest = True
            elif gap_harvest.lower() in ["false", "no", "0"]:
                gap_harvest = False
            else:
                print("Invalid input")

            harvesting_max_percent = float(
                input("enter number that maximum harvest percentage! only number!")
            )
            planting_year = int(
                input(
                    "what year is the planting started -- please enter the number only (integer)"
                )
            )

            self.json_path_config = os.path.join(
                self.config_location, f"{project_name}_main.json"
            )

            # Create the folder if it doesn't exist
            os.makedirs(self.config_location, exist_ok=True)

            self.conf_general = {
                "project_name": project_name,
                "csv_plot_summary_zone": csv_plot_summary_zone,
                "duration_project": duration_project,
                "gap_harvest": gap_harvest,
                "harvesting_max_percent": harvesting_max_percent,
                "planting_year": planting_year,
                "root_folder": root_folder,
            }

            with open(self.json_path_config, "w") as json_conf:
                json.dump(self.conf_general, json_conf, indent=4)

            print(f"Created, {project_name} project configuration!")

        else:
            # directly goes to json_path_config
            with open(self.json_path_config, "r") as json_general:
                conf_general = json.load(json_general)

            self.conf_general = conf_general
            project_name = conf_general["project_name"]
            csv_plot_summary_zone = conf_general[
                "csv_plot_summary_zone"
            ]  # used if we have the example production and conservation example like this file AEE based on exact seedling distribution number and Arief template https://docs.google.com/spreadsheets/d/1I53EmoYjHW89uxlrqadgQnYp6gfV4dCQPVXiQkNqzM4/edit#gid=1063470375
            # # # Duration project, extension of one cycle
            duration_project = conf_general["duration_project"]  # year
            # gap_harvest = False --> no gap harvesting means there is no break in one year after harvest year, it will promply re-plant in the same year for the next cycle
            # True # gap harvesting, means the replant will be delayed, due to some factors - operation of tpp, soil recovery, etc.
            ## all gap or no gap, will follow the standard tree c-sink, retained 40% tc
            gap_harvest = conf_general["gap_harvest"]
            planting_year = conf_general[
                "planting_year"
            ]  # meaning, if 2023, it will be planted at 2023, measured in 2024
            harvesting_max_percent = conf_general[
                "harvesting_max_percent"
            ]  # csi standard at least retaining 40%

        self.config = self.conf_general

        # THIS IS THE ALL DATA WE WILL GENERATED FOR INPUT AND OUTPUT in the SAME FOLDER in ROOT_FOLDER above
        # the name gdrive is copy paste from previous code, because it was using google shared drive link before
        self.gdrive_location_seedling = os.path.join(
            root_folder, f"{project_name}_distribution_trees_seedling.csv"
        )
        # csv for allometry formula used
        self.gdrive_location_df_tree_selected = os.path.join(
            root_folder, f"{project_name}_formulas_allometry.csv"
        )
        # json for mortality rate, thinning etc
        self.gdrive_location_scenario_rate = os.path.join(
            root_folder, f"{project_name}_forestry_scenario.json"
        )
        # shp for CSU, if any
        self.gdrive_plot_csu = os.path.join(
            root_folder, f"{project_name}_csu_updated.shp"
        )
        # output raw model, simulation - can be used in gsheet, pivot excel etc
        self.gdrive_raw_output = os.path.join(
            root_folder, f"{project_name}_OUTPUT_RAW_CALCULATION.csv"
        )
        # input cooling service
        self.gdrive_input_cs = os.path.join(
            root_folder, f"{project_name}_input_gcs_generated.csv"
        )

        self.gdrive_growth_selected = os.path.join(
            root_folder, f"{project_name}_selected_growth_model.csv"
        )

        # creation of the class library to acquire the coredb information
        AllometryLibrary.__init__(
            self,
            growth_csv=self.growth_csv,
            allometry_csv=self.allometry_csv,
            name_column_species_allo=self.name_column_species_allo,
            name_column_species_growth=self.name_column_species_growth,
            download_csv=self.download_csv,
            conversion_tco2=self.conversion_tco2,
            link_growth=self.link_growth,
            link_allometry=self.link_allometry
        )

        self.growth_melt = self.growth_df
        self.allometry_df = self.allometry_df

        self.co2_data_dict = self._co2_data(
            self.default_pair_allometry_growth(), self.name_column_species_allo
        )

        if self.first_run:
            # Display and interact with the allometry type selection
            ## case for t4t balancer ex-ante --> we should avoid the default pair because we alter manually the growth and allometry csv

            # self.co2_data_dict  = self._co2_data(self.default_pair_allometry_growth(), self.name_column_species_allo)
            self.interact_allometry_type()
            # by default we use the default pair from the db
            # (it might be different if you alter some data in the csv growth and allometry formula if the coredb data is no longer exist! - case t4t balancer ex-ante)

        else:
            # Handle the non-first-run logic as required
            self.load_existing_data()

    def load_existing_data(self):
        # Load existing data for non-first-run initialization
        self.df_tree_selected = cleaning_csv_df(
            pd.read_csv(self.gdrive_location_df_tree_selected)
        )
        if self.re_select_growth_data:
            self.acquire_growth_data()
        else:
            self.growth_selected = cleaning_csv_df(
                pd.read_csv(self.gdrive_growth_selected)
            )
        print("Loaded existing allometry and growth data.")
        display(self.df_tree_selected)
        display(self.growth_selected)

        self.csu_seedling = cleaning_csv_df(pd.read_csv(self.gdrive_location_seedling))
        display(self.csu_seedling)

        self.df_unique_species = self.df_tree_selected.drop(
            columns=["zone"]
        ).drop_duplicates()

        # this is important if someday we want to alter manually the csv! (growth model selected and allometry) - case t4t balancer ex-ante
        self.co2_data_dict = self._co2_data(
            self.override_tco2_series(self.df_unique_species, self.growth_selected),
            self.name_column_species_allo,
        )

        # input_scenario_species json
        with open(self.gdrive_location_scenario_rate, "r") as json_write_scenario:
            self.input_scenario_species = json.load(json_write_scenario)
        pprint.pprint(self.input_scenario_species)

        # Get unique species selected and filter growth data based on selections
        self.unique_species_selected = self.df_tree_selected[
            self.name_column_species_allo
        ].unique()
        # highlight on the df_tco2_selected only
        self.df_tco2_selected = self.get_selected_co2(
            self.co2_data_dict, self.unique_species_selected
        )

    def interact_allometry_type(self):
        """Step 1: Initialize and display SelectingScenario widget for interactive selection"""
        # Initialize SelectingScenario widget
        self.wm = SelectingScenario(
            allometric_column_filter=self.allometry_df,
            name_column_species_allo=self.name_column_species_allo,
        )

        # Output widget for displaying logs and updates
        self.output = widgets.Output()

        # Create the submit button
        submit_button = widgets.Button(description="Submit species!")
        submit_button.on_click(self.on_submit_click)  # Attach the submit action

        # Display the widget, button, and output
        display(self.wm)
        display(submit_button)
        display(self.output)

    def on_submit_click(self, button):
        with self.output:
            print("Step 1: Submit button clicked.")
            try: # Add try block for better error catching
                # Capture selected data
                self.df_selected = self.wm.selected_data
                if not self.df_selected or not any(not df.empty for df in self.df_selected.values()):
                    print("ERROR: No valid data selected from previous step.")
                    return # Stop if no data
                print("Step 2: Data captured from wm.")
                self.df_tree_selected = pd.concat(list(self.df_selected.values())).reset_index(drop=True) # Ensure it's a list for concat
                print("Step 3: Data concatenated.")
                display(self.df_tree_selected)

                # Save the selections
                self.df_tree_selected.to_csv(self.gdrive_location_df_tree_selected, index=False)
                print("Step 4: Data saved to CSV.")

                # Acquire growth data
                self.df_unique_species = self.df_tree_selected.drop(columns=["zone"]).drop_duplicates()
                # Check if unique_species_selected is populated correctly before acquire_growth_data if it depends on it
                print(f"Step 5: Unique species identified: {getattr(self, 'unique_species_selected', 'Attribute missing')}")
                self.acquire_growth_data() # Ensure this defines self.unique_species_selected if needed below
                print("Step 6: Growth data acquired.")

                # Create a template DataFrame
                list_column_name = [
                    "Plot_ID",
                    "Plot_Name",
                    "zone",
                    "area_ha",
                    "is_replanting",
                    "year_start",
                    "mu",
                ] + [f"{species}_num_trees" for species in self.unique_species_selected]
                plot_csu = pd.DataFrame(columns=list_column_name)
                plot_csu = pd.DataFrame(columns=list_column_name)
                print("Step 7: Template DataFrame created.")
                display(plot_csu)

                # Display the data entry form
                print("Step 8: Initializing CSUEntryForm...")
                self.csu_form = CSUEntryForm(plot_csu)
                print("Step 9: Displaying CSUEntryForm...")
                self.csu_form.display_form()
                # print("Step 10: CSUEntryForm displayed.")

                # COLAB-SPECIFIC FIX START =================================
                if is_running_in_colab():
                    from google.colab import output
                    output.clear()  # Clear previous widget states
                    
                    # Display form elements sequentially
                    display(self.csu_form.form)
                    display(self.csu_form.output)
                    
                    # Force widget registration
                    output.register_widget(self.csu_form.add_row_button)
                    output.register_widget(self.csu_form.reset_button)
                else:
                    self.csu_form.display_form()
                # COLAB-SPECIFIC FIX END ===================================

                # Add a submit button for the CSU form
                print("Step 11: Creating Submit CSU Seedling button ---> please submit after all data is added...")
                self.submit_csu_form_button.on_click(self.on_submit_form_csu)
            
                if is_running_in_colab():
                    from google.colab import output
                    output.register_widget(self.submit_csu_form_button)
                    display(HTML("<hr>"))  # Visual separator
                    display(self.submit_csu_form_button)
                    display(HTML("<p>Click submit after completing data entry</p>"))
                else:
                    display(self.submit_csu_form_button)
                    
                print("----------------------------------------------")
                display(self.submit_csu_form_button)
                # print("Click submit above if you done setup the csu \n------------------------------------")

            
            except Exception as e:
                print(f"ERROR in on_submit_click: {e}")
                import traceback
                traceback.print_exc() # Print full traceback

    def on_submit_form_csu(self, button):
        """Handles submission of CSU seedling data"""
        print('handle the submission for scenario is started')
        with self.output:
            # Save the updated DataFrame
            self.csu_form.csu_seedling.to_csv(
                self.gdrive_location_seedling, index=False
            )
            self.csu_seedling = self.csu_form.csu_seedling

            print(f"CSU Seedling data saved to {self.gdrive_location_seedling}")

            # Generate widgets for the next step
            # self.generate_species_widgets()
            # print("Proceeding to widget generation for scenario data.")

    def generate_species_widgets(self):
        """Step 3: Generate and display species-specific widgets for zones"""
        with self.output:
            # Group max years for each species
            grouping_max_year = {
                species: len(self.co2_data_dict[species]) - 1
                for species in self.unique_species_selected
            }

            # Generate widgets for production and protected zones
            if "production_zone" in self.df_selected:
                self.widget_production_zone = self.create_species_widgets(
                    self.df_selected["production_zone"], grouping_max_year
                )
                print("Production/Plantation zone widgets:")
                display(self.widget_production_zone)

            if "protected_zone" in self.df_selected:
                self.widget_protected_zone = self.create_species_widgets(
                    self.df_selected["protected_zone"], grouping_max_year
                )
                print("\n------------------------------------------------")
                print("Protected zone widgets:")
                display(self.widget_protected_zone)

            if "is_replanting" in self.df_selected:
                self.widget_replanting = self.create_species_widgets(
                    self.df_selected["is_replanting"], grouping_max_year
                )
                print("\n------------------------------------------------")
                print("replanting widgets:")
                display(self.widget_replanting)

            # Add a final submit button
            self.final_submit_button = widgets.Button(
                description="Submit Scenario Data"
            )
            self.final_submit_button.on_click(self.on_final_submit_click)
            display(self.final_submit_button)

    def on_final_submit_click(self, button):
        """Triggered when the final submit button is clicked to capture widget data"""
        self.generate_input_scenario_species()
        print("Scenario data submitted successfully!")

    def acquire_growth_data(self):
        # Get unique species selected and filter growth data based on selections
        self.unique_species_selected = self.df_tree_selected[
            self.name_column_species_allo
        ].unique()
        # highlight on the df_tco2_selected only
        self.df_tco2_selected = self.get_selected_co2(
            self.co2_data_dict, self.unique_species_selected
        )

        self.growth_selected = self.growth_melt[
            self.growth_melt[self.name_column_species_growth].isin(
                self.unique_species_selected
            )
        ]
        # self.growth_selected.to_csv(self.gdrive_growth_selected) # to avoid re-update the growth selected on initial model (expost update scheme) we will move this to the outside in method not in init
        print("Growth data selected:")
        display(self.growth_selected)

    def create_species_widgets(self, df_zone, grouping_max_year):
        widgets_list = []
        for species in df_zone[self.name_column_species_allo].to_list():
            widget = Project_Setting_Species(species, grouping_max_year)
            widget.add_class("box_style")
            widgets_list.append(widget)
        return widgets.VBox(widgets_list)

    def generate_input_scenario_species(self):
        """Step 4: Create input_scenario_species based on widget output and save to JSON file"""
        # Initialize a dictionary to store scenario data for each zone
        self.input_scenario_species = {}

        # Define widget zones and corresponding names
        zones = {
            "production_zone": getattr(self, "widget_production_zone", None),
            "protected_zone": getattr(self, "widget_protected_zone", None),
        }

        # Loop through each zone, only processing zones with existing widgets
        for zone_name, widget_zone in zones.items():
            if widget_zone is not None:
                # Collect data for each species in the zone
                input_scenario_per_zone = {}
                for widget in widget_zone.children:
                    # Extract species name and details from the widget
                    species_name = (
                        widget.wid_species_name.value.split(">")[2]
                        .replace("</b", "")
                        .strip()
                    )
                    input_scenario_per_zone[species_name] = {
                        "harvesting_year": widget.year_harvesting.value,
                        "mortality_percent": widget.mortality_percent.value,
                        "natural_thinning": widget.natural_thinning.value,
                        "frequency_manual_thinning": widget.frequency_thinning.value,
                    }

                    # Add any additional parameters from the widget's `check` property
                    input_scenario_per_zone[species_name].update(widget.check)

                # Add this zone's data to the main dictionary
                self.input_scenario_species[zone_name] = input_scenario_per_zone

        # Save the dictionary to a JSON file only if data was collected
        if self.input_scenario_species:
            with open(self.gdrive_location_scenario_rate, "w") as json_file:
                json.dump(self.input_scenario_species, json_file, indent=4)

            print("Input scenario species saved to JSON:")
            display(self.input_scenario_species)
        else:
            print("No data available to save for input scenario species.")

    # now after all the configuration input done, we proceed to the plot calculation, and graph chart, and later with input cooling
    # plot calc
    def csi_plot_model(self) -> pd.DataFrame:
        # run the export csv after model is started running
        self.growth_selected.to_csv(self.gdrive_growth_selected)

        p = Plot(csv_plot=self.gdrive_location_seedling)
        self.plot_seedling = p.csu_distribution()

        self.plot_sum = self.plot_seedling["plot_sum"]
        self.melt_plot_species = self.plot_seedling["melt_plot_species"]

        if self.config.get("thinning_stop", False):
            max_density_df = pd.read_csv(
                os.path.join(
                    os.path.dirname(__file__),
                    "ex_ante/00_input/max_density_trees_ha.csv",
                )
            )  # hardcoded

            # growth max_dbh
            species_col = self.name_column_species_growth
            dbh_col = "DBH"

            # Group by the species column, select the DBH column, and find the maximum
            max_dbh_per_species = self.growth_selected.groupby(species_col)[
                dbh_col
            ].max()

            melt_plot_species_with_max = pd.merge(
                self.melt_plot_species,  # Left DataFrame
                max_dbh_per_species,  # Right object (our renamed Series)
                left_on="species",  # Column from left DataFrame to join on
                right_index=True,  # Use the index of the Series as the join key
                how="left",  # Keep all rows from left DataFrame ('melt_plot_species')
            ).rename(columns={"DBH": "dbh_max"})

            # Rename columns for easier access (optional but recommended)
            dbh_threshold_col = "Average Pop DBH [cm]"
            density_col = "Max Stand Density [trees / ha]"

            dbh_input_col = "dbh_max"

            # Create a new column in df2 by applying the function to each value in the dbh_input_col
            melt_plot_species_with_max["calc_max_density"] = melt_plot_species_with_max[
                dbh_input_col
            ].apply(
                get_max_density,  # The function to apply
                args=(max_density_df, dbh_threshold_col, density_col),
            )

            self.melt_plot_species = melt_plot_species_with_max

        # STORING SCENARIO TO DICTIONARY
        csi = CSIExante(
            # plot_seedling=self.plot_seedling,
            plot_sum=self.plot_sum,
            melt_plot_species=self.melt_plot_species,
            input_scenario_species=self.input_scenario_species,
            df_tco2_selected=self.df_tco2_selected,
            config=self.conf_general,
        )

        self.dict_plot_scenario = csi.dict_plot_scenario
        self.melt_plot_species = csi.melt_plot_species
        self.dict_plot_harvest_year_list = csi.dict_plot_harvest_year_list
        self.dict_min_harvest_year_plot = csi.dict_min_harvest_year_plot
        self.dict_plot_start_year = csi.dict_plot_start_year
        self.updated_input_scenario = csi.updated_input_scenario
        self.zone_plot_dict_scenario = csi.zone_plot_dict_scenario
        self.plot_grouped_dict_scenario = csi.plot_grouped_dict_scenario
        self.new_dict_plot_harvest_year_list = csi.new_dict_plot_harvest_year_list

        # Pretty print the resulting dictionary
        # pprint.pprint(self.dict_plot_scenario)
        pprint.pprint(self.plot_grouped_dict_scenario)
        pprint.pprint(self.dict_plot_harvest_year_list)
        pprint.pprint(self.dict_min_harvest_year_plot)
        pprint.pprint(self.dict_plot_start_year)

        self.plot_carbon_melt = csi.plot_carbon_melt()

        self.config = csi.config
        all_df_merged = csi.ex_ante()

        # adding tco2_per_tree column in the main df
        all_df_merged = pd.merge(
            all_df_merged,
            self.df_tco2_selected[["tco2_per_tree", "species", "year"]],
            left_on=["species", "rotation_year"],
            right_on=["species", "year"],
            how="left",
            suffixes=("_merged", "_coredb"),
        )

        # all_df_merged['num_trees_adjusted'] = (all_df_merged['total_csu_tCO2e_species'] / (all_df_merged['TTB_value_pertree_ton']*0.47*3.67)).astype(int)  #num trees = total tco2e/ per tree tco2e in rotation year
        # because the gap year I think
        mask = all_df_merged["total_csu_tCO2e_species"].notna()

        all_df_merged["num_trees_adjusted"] = np.where(
            mask,
            np.round(
                (
                    all_df_merged["total_csu_tCO2e_species"]
                    / all_df_merged["tco2_per_tree"]
                ).fillna(0)
            ).astype(
                int
            ),  # Ensure rounding is applied before converting to int
            np.nan,
        )

        all_df_merged = all_df_merged.drop(columns="year_coredb")
        all_df_merged = all_df_merged.rename(columns={"year_merged": "year"})
        all_df_merged["year"] = all_df_merged["year"].astype(int)

        # let's say if the data is from an initial model, meaning there is no replanting, we will setup the is_replanting as False, if there is exist is_replanting column we will use it
        all_df_merged["is_replanting"] = all_df_merged.apply(
            lambda x: (
                x["is_replanting"]
                if "is_replanting" in all_df_merged.columns
                else False
            ),
            axis=1,
        )

        all_df_merged.to_csv(self.gdrive_raw_output)

        # for num_trees_tco2
        output_dir = os.path.join(
            os.path.dirname(self.gdrive_location_seedling),
            "num_trees_tco2_over_the_year",
        )
        os.makedirs(output_dir, exist_ok=True)
        path_file = os.path.join(output_dir, self.config["project_name"])

        # display(all_df_merged)  # troubleshoot only

        num_tco_years_run = num_tco_years(
            df_ex_ante=all_df_merged,
            is_save_to_csv=path_file,
            distribution_seedling=self.gdrive_location_seedling,
            override_num_trees_0=self.override_num_trees_0,
            mortality_csu_df=self.mortality_csu_df,
        )
        display(num_tco_years_run["exante_num_trees_yrs"])
        display(num_tco_years_run["exante_tco2e_yrs"])

        return all_df_merged

    # def cooling_service(self, all_df_merged=None, use_input_cooling=""):
    #     if use_input_cooling == "":
    #         # start to create input cooling and generate the result
    #         self.input_gcs = input_cooling(
    #             all_df_merged,
    #             self.growth_melt,
    #             self.name_column_species_growth,
    #             self.conf_general["planting_year"],
    #         )
    #         file_input = self.gdrive_input_cs
    #         file_input = os.path.normpath(file_input)
    #         # file_input = os.path.join(input_base_dir,file_input_name)

    #         self.input_gcs.to_csv(file_input, index=False)
    #         print(
    #             f"file input to estimating the cooling and reflux effect: {file_input}"
    #         )

    #         file_input_name = os.path.basename(file_input)
    #         # starting to call the class needed - Stefan Script
    #         cs = CSink(csv_filename_path=file_input, is_file_path=True)
    #         c = Cooling(cs, is_file_path=True)

    #         output_folder = self.root_folder
    #         output_file_name = os.path.join(output_folder, "cooling-output_")

    #         # c.reporting()
    #         # c.plot(is_save_img_graph= False)
    #         c.to_file(
    #             output_file_name + file_input_name
    #         )  # save into the complete name the output_gcs
    #         print(f"file output csv saved into {output_file_name + file_input_name}")

    #     else:
    #         # starting to call the class needed - Stefan Script
    #         cs = CSink(csv_filename_path=use_input_cooling, is_file_path=True)
    #         c = Cooling(cs, is_file_path=True)

    #         # let's not save the output because we already have it usually

    #     return c

    # def graph_cooling_service(self, cooling, location_save=""):
    #     cooling.reporting()
    #     if location_save != "":
    #         cooling.plot(is_save_img_graph=True, location_save=location_save)
    #     else:
    #         cooling.plot(is_save_img_graph=False)

    def graph_species_project(self, input_gcs, location_save=""):
        from ex_ante.csi_tree.input_cooling_creation import plot_co2_species

        plot_co2_species(input_gcs, location_save=location_save)

    def graph_species(self):
        # creating the chart biomass, proportion and tC - growth for every species selected, every year, after thinning, in one cycle only
        # adapted from https://i.stack.imgur.com/Dkj75.png

        selected_tco2e = {
            k: v
            for k, v in self.co2_data_dict.items()
            if k in self.unique_species_selected
        }

        @interact
        def line_plot(column=["tCO2e"], species=sorted(self.unique_species_selected)):
            # Convert the selected species data into a DataFrame
            prop_filtered = pd.DataFrame({column[0]: selected_tco2e[species]})

            # Plot the filtered data for the selected column and species
            if not prop_filtered.empty:
                ax = prop_filtered.plot(title=f"{species}: {column}")
                ax.set_ylabel("tCO2e")
                ax.set_xlabel("year")
                return ax
            else:
                print("No data available for the selected species.")

    def baseline_calc(
        self,
        input_str: str = "",
        detailed_plot_baseline: str = "",
    ):
        base_year = self.conf_general["planting_year"]
        yearly_summary = (
            self.input_gcs.groupby("measurement_year")["co2_tree_captured_tonnes"]
            .sum()
            .reset_index()
        )
        df_baseline = pd.DataFrame()
        if input_str != "":
            # example input_str --> # copy paste from Kirill template
            """-2329.87531,	-4934.43412,	-4934.43412,	-4934.43412,	-4934.43412,	-4934.43412,	-4934.43412,	-4934.43412,	-4934.43412,	-4934.43412,	-4934.43412,	-4934.43412,	-4934.43412,	-4934.43412,	-4934.43412,	-4934.43412,	-4934.43412,	-4934.43412,	-4934.43412,	-4934.43412,	-4934.43412,	-4934.43412,	-4934.43412,	-4934.43412,	-4934.43412,	-4934.43412,	-4934.43412,	-4934.43412,	-4934.43412,	-4934.43412"""
            df_baseline = pd.read_csv(io.StringIO(input_str), header=None)
            df_baseline = df_baseline.T
            df_baseline.columns = ["baseline_tCO2e"]
            # df_baseline['baseline_tCO2e']  = df_baseline['baseline_tCO2e'].apply(lambda x: x.replace(' ','').replace('-',''))
            df_baseline["baseline_tCO2e"] = (
                df_baseline["baseline_tCO2e"].astype(float) * -1
            )

            # add info of year
            df_baseline["measurement_year"] = range(
                base_year + 1, base_year + 1 + len(df_baseline)
            )

            # df_baseline

        elif detailed_plot_baseline != "":
            df_baseline = pd.read_csv(detailed_plot_baseline)
            # df that is column based - TODO

        # Summarize data based on year
        joined_yearly_tCO2ecapture_baseline = pd.merge(
            yearly_summary, df_baseline, on="measurement_year"
        )
        joined_yearly_tCO2ecapture_baseline["net_tCO2e_sequestration"] = (
            joined_yearly_tCO2ecapture_baseline["co2_tree_captured_tonnes"]
            - joined_yearly_tCO2ecapture_baseline["baseline_tCO2e"]
        )

        path_saved_baseline = os.path.join(
            self.root_folder,
            f"table_nettCO2withbaseline_{self.config['project_name']}.csv",
        )
        joined_yearly_tCO2ecapture_baseline.to_csv(path_saved_baseline)

        return joined_yearly_tCO2ecapture_baseline

    def graph_seq_baseline(self, joined_yearly_tCO2ecapture_baseline, location_save=""):
        import matplotlib.pyplot as plt

        # Calculate the long-term average of net CO2 sequestration
        long_term_average_net = (
            joined_yearly_tCO2ecapture_baseline["net_tCO2e_sequestration"].mean() * 0.8
        )
        print("LTA :", long_term_average_net)

        # Calculate the average of baseline tCO2e
        baseline_average = joined_yearly_tCO2ecapture_baseline["baseline_tCO2e"].mean()
        print("Baseline average:", baseline_average)

        # Plot the bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(
            joined_yearly_tCO2ecapture_baseline["measurement_year"],
            joined_yearly_tCO2ecapture_baseline["net_tCO2e_sequestration"],
            color="green",
            label="Net CO2e Sequestration",
        )

        # Plot the long-term average line
        plt.axhline(
            y=long_term_average_net,
            color="orange",
            linestyle="--",
            linewidth=2,
            label="Long-term Average - Net with buffer 20%",
        )

        # Plot the baseline line
        plt.axhline(
            y=baseline_average,
            color="red",
            linestyle="-",
            linewidth=2,
            label="Baseline tCO2e",
        )

        # Add label over the long-term average line
        plt.text(
            joined_yearly_tCO2ecapture_baseline["measurement_year"].min() - 0.5,
            long_term_average_net
            - joined_yearly_tCO2ecapture_baseline["net_tCO2e_sequestration"].max()
            / 30,  #  to make them visually better joined_yearly_tCO2ecapture_baseline['net_tCO2e_sequestration'].max()/30
            f"{long_term_average_net:.2f}",
            color="orange",
            fontsize=10,
            fontweight="bold",
        )

        # Add label over the baseline line
        plt.text(
            joined_yearly_tCO2ecapture_baseline["measurement_year"].min() - 0.5,
            baseline_average
            - joined_yearly_tCO2ecapture_baseline["net_tCO2e_sequestration"].max()
            / 30,  #  to make them visually better joined_yearly_tCO2ecapture_baseline['net_tCO2e_sequestration'].max()/30
            f"{baseline_average:.2f}",
            color="red",
            fontsize=10,
            fontweight="bold",
        )

        plt.xlabel("Year")
        plt.ylabel("Net CO2e Sequestration (Tonnes)")
        plt.title("Net CO2 stored by trees over 30 years (baseline subtracted)")
        plt.legend(title="tCO2e", loc="upper left", fontsize="small", markerscale=0.5)
        if location_save != "":
            plt.savefig(
                location_save, dpi=300, bbox_inches="tight"
            )  # Save as JPEG with high resolution and tight bounding box
        plt.show()
