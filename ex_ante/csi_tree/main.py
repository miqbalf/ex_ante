import math
import os

import numpy as np
import pandas as pd

from .utils import (
    allowed_cut_harvest,
    calculate_and_merge_proportions,
    calculate_proportion,
    create_nested_dict,
    fill_harvest_cycle,
    harvest_cycle_rotation,
    is_species_allowed_to_harvest,
    merge_dictionaries_zone,
    num_trees_harvest_allowed,
    num_trees_retained,
    re_planting,
    simulate_all_proportions_for_species,
    transform_plot_base_dict,
)

filtered_column = [
                "Plot_ID",
                "area_ha",
                "species",
                "year",
                "total_csu_tCO2e_species",
                "rotation_year",
                "harvest_time",
                "remnant_trees",
                "allowed_cut_proportion_okay",
                "remnant_level",
                "num_trees",
                "num_trees_harvested",
                "num_trees_retained",
                "allowed_cut_csu",
                "tCO2e_retained",
                "cycle_harvest",
                "tCO2e_harvest_allowed_species",
                "tco2_per_tree",
            ]

# Custom exception
class BreakAllLoops(Exception):
    pass

class CSIExante:
    def __init__(
        self,
        # plot_seedling: dict,
        plot_sum,
        melt_plot_species,
        input_scenario_species: dict,
        config: dict,
        df_tco2_selected: pd.DataFrame,
        override_avg_tree_perha='' # new for the thinning stop
    ):

        # self.plot_seedling = plot_seedling
        # self.plot_sum = plot_seedling["plot_sum"]
        # self.melt_plot_species = plot_seedling["melt_plot_species"]
        self.plot_sum = plot_sum
        self.melt_plot_species = melt_plot_species

        self.config = config
        self.input_scenario_species = input_scenario_species
        self.duration_project = config["duration_project"]
        self.gap_harvest = config["gap_harvest"]
        self.df_tco2_selected = df_tco2_selected

        self.dict_plot_scenario = {}
        self.dict_plot_harvest_year_list = {}
        self.dict_min_harvest_year_plot = {}
        self.dict_plot_start_year = {}
        self.dict_plot_scenario = {}

        self.override_avg_tree_perha = override_avg_tree_perha

        melt_plot_species = self.melt_plot_species
        # re-work on the dict_plot_scenario
        # Define the base groupby columns
        groupby_columns = ["zone", "Plot_ID", "year_start"]

        # Check if 'is_replanting' exists in the DataFrame
        if "is_replanting" in self.melt_plot_species.columns:
            groupby_columns.insert(
                2, "is_replanting"
            )  # Add 'is_replanting' to groupby columns if it exists
        else:
            melt_plot_species["is_replanting"] = False
            groupby_columns.insert(
                2, "is_replanting"
            )  # now we will normalize this to add in the keys

        # Group by the specified columns and aggregate 'species'
        # Filter for species with num_trees greater than 0
        melt_plot_species = melt_plot_species[melt_plot_species["num_trees"] > 0]

        self.melt_plot_species = melt_plot_species.copy()
        # self.melt_plot_species = self.melt_plot_species[self.melt_plot_species['num_trees']> 0]

        dict_group_species = (
            melt_plot_species.groupby(groupby_columns)["species"]
            .apply(list)  # Aggregate species into a list
            .to_dict()  # Convert to dictionary
        )

        # Generate the new dictionary for dict_plot_scenario later
        new_dict = {}
        update_all_scenario = {}
        if (
            "non_replanting" in self.input_scenario_species.keys()
        ):  # try to work on previous version vs updated version of input_scenario
            update_all_scenario = self.input_scenario_species
        else:
            update_all_scenario["non_replanting"] = self.input_scenario_species
        self.updated_input_scenario = update_all_scenario

        # we will eliminate this by make a default value (False) for non-exist replanting
        for key, value in dict_group_species.items():
            if len(key) == 4:
                zone, plot_id, is_replanting, year_start = key
                # Use plot_id, is_replanting, and zone here
                # Determine the scenario: 'replanting' or 'non_replanting'
                scenario = "replanting" if is_replanting else "non_replanting"
            elif len(key) == 3:
                zone, plot_id, year_start = key
                # Use plot_id and zone here
                # You might need to handle the missing 'is_replanting' based on your specific use case
                scenario = "non_replanting"
            else:
                print("Unexpected key length:", len(key))

            # print(scenario)

            # Access the corresponding zone data in the second dictionary
            zone_data = update_all_scenario[scenario].get(zone, {})
            species_list = value

            # Construct the new dictionary structure
            new_dict[key] = {
                species: zone_data.get(species, {}) for species in species_list
            }

        self.dict_plot_scenario = new_dict

        # print('NOW THE DICTIONARY PLOT SCENARIO --> \n',self.dict_plot_scenario)
        if self.dict_plot_scenario == {}:
            raise ValueError(
                "dictionary is empty, please check the distribution seedling data!"
            )

        zone_plot_dict = create_nested_dict(self.melt_plot_species)
        # self.zone_plot_dict = zone_plot_dict

        # Assuming your dictionaries are named zone_dict and non_replanting_dict
        zone_plot_dict_scenario = merge_dictionaries_zone(
            zone_plot_dict, self.updated_input_scenario
        )
        self.zone_plot_dict_scenario = zone_plot_dict_scenario

        self.plot_grouped_dict_scenario = transform_plot_base_dict(
            self.zone_plot_dict_scenario
        )

        # # k is plot, v is their value in scenario
        for k, v in self.dict_plot_scenario.items():
            print(k, v)
            # now we want to only get the min year of harvest cycle
            harvest_list = []
            for species, conf in v.items():
                harvest_list.append(conf["harvesting_year"])

            sorted_harvest_list = sorted(list(set(harvest_list)))
            # print('HEEEY CHECCK', sorted_harvest_list)
            # self.dict_plot_harvest_year_list[k] = sorted_harvest_list
            self.dict_min_harvest_year_plot[k] = min(harvest_list)

            all_harvest_list = (
                []
            )  # Ensure the next cycle years are added (e.g., 10, 20, 30)

            for harvest_start in sorted_harvest_list:
                # Repeat the multiplication process for each cycle within the project duration
                a = 1
                b = 0  # Reset b for each harvest_start

                while (harvest_start * a) <= self.duration_project:
                    # If gap_harvest is True, apply the gap logic
                    if self.gap_harvest:
                        next_harvest_year = (harvest_start * a) + b
                        # Add the result to the list if it's not already present
                        if next_harvest_year not in all_harvest_list:
                            all_harvest_list.append(next_harvest_year)
                        b += 1  # Increment gap for the next cycle
                    else:
                        # For regular cycles without gap
                        next_harvest_year = harvest_start * a
                        if next_harvest_year not in all_harvest_list:
                            all_harvest_list.append(next_harvest_year)
                    a += 1  # Increment the cycle multiplier

            # Sort the list and remove duplicates
            all_harvest_list = sorted(set(all_harvest_list))
            all_harvest_list = sorted(
                list(set(all_harvest_list))
            )  # make sure its unique

            # print(k)
            # this means that the unique_label in the conf will be a unique
            # plot_label = plot_sum[plot_sum['Plot_ID']==int(i)]['unique_label'].unique()[0]
            # print(plot_label)
            # year_start = self.plot_sum[self.plot_sum['Plot_ID']==k]['year_start'].unique()[0]   # THIS IS CONTEXTUAL TO SRA with different start year PLEASE CHECK AGAIN LATER -
            year_start = k[-1]
            # print(year_start)

            # NOW WE WILL ADD A NUMBER OF START YEAR TO MAKE THIS RELEVANT - recursive, chunk iteration of the for loop, CONTEXTUAL SRA OR SOME OTHER CONFIGURATION THAT HAS DIFFERENT TIME YEAR START PLANTING
            self.dict_min_harvest_year_plot[k] = min(harvest_list) + year_start - 1
            sorted_harvest_list_added_start_year = [
                i + year_start - 1
                for i in all_harvest_list
                if i + year_start - 1 <= self.duration_project
            ]

            self.dict_plot_harvest_year_list[k] = sorted(
                sorted_harvest_list_added_start_year
            )

            # plot_sum[(plot_sum['Plot_ID']==i) & (plot_sum['Plot_ID']==i)]

            # adding the project end duration
            for k, v in self.dict_plot_harvest_year_list.items():
                if self.duration_project not in v:
                    v.append(self.duration_project)

            # # Ensure project_end is part of the list
            # if self.duration_project not in all_harvest_list:
            #    all_harvest_list.append(self.duration_project)
            # self.dict_plot_harvest_year_list[k] = all_harvest_list

            self.dict_plot_start_year = {
                self.plot_sum["Plot_ID"][i]: self.plot_sum["year_start"][i]
                for i in range(len(self.plot_sum["Plot_ID"]))
            }

            new_format_dict_plot_harvest_list = {}
            for (
                zone,
                plot_id,
                is_replanting,
                year_start,
            ), harvest_years in self.dict_plot_harvest_year_list.items():
                if plot_id not in new_format_dict_plot_harvest_list:
                    new_format_dict_plot_harvest_list[
                        (plot_id, is_replanting, year_start)
                    ] = {}
                new_format_dict_plot_harvest_list[
                    (plot_id, is_replanting, year_start)
                ] = harvest_years
            self.new_dict_plot_harvest_year_list = new_format_dict_plot_harvest_list

        plot_group_dict_scenario = self.plot_grouped_dict_scenario
        new_dict_plot_harvest_year_list = self.new_dict_plot_harvest_year_list

        for plot_id, plot_data in plot_group_dict_scenario.items():
            #     print(plot_id)
            for is_replanting, scenario_data in plot_data.items():
                list_harvest_year_start = []
                for year_start in scenario_data["year_start_list"]:
                    key_tuple = (plot_id, is_replanting, year_start)
                    # print(key_tuple)
                    if key_tuple in new_dict_plot_harvest_year_list:
                        list_harvest_year_start.append(
                            new_dict_plot_harvest_year_list[key_tuple]
                        )

                #         print(list_harvest_year_start)
                scenario_data["harvest_list"] = list_harvest_year_start

        self.plot_grouped_dict_scenario = plot_group_dict_scenario

        self.config["dict_plot_scenario"] = self.dict_plot_scenario
        self.config["dict_plot_harvest_year_list"] = self.dict_plot_harvest_year_list
        self.config["dict_min_harvest_year_plot"] = self.dict_min_harvest_year_plot
        self.config["dict_plot_start_year"] = self.dict_plot_start_year
        self.config["dict_plot_scenario"] = self.dict_plot_scenario
        self.config["plot_grouped_dict_scenario"] = self.plot_grouped_dict_scenario
        self.config["new_dict_plot_harvest_year_list"] = (
            self.new_dict_plot_harvest_year_list
        )

    def plot_carbon_melt(self):

        plot_carbon = pd.merge(
            self.melt_plot_species,
            self.df_tco2_selected,
            left_on="species",
            right_on="species",
            how="left",
        )

        print("DICTPLOT SCENARIO: \n", self.dict_plot_scenario)

        # proportion_per_trees = biomass_per_tree_year.copy()
        plot_carbon = plot_carbon.copy()

        if self.config.get("thinning_stop", False):
            # Use the simulation function WITH the density stop logic
            plot_carbon = calculate_and_merge_proportions(
                plot_carbon,  # Pass the DataFrame
                self.dict_plot_scenario,  # Pass the config dict
                simulate_all_proportions_for_species,  # Pass the correct simulation function
                override_avg_tree_perha =self.override_avg_tree_perha
            )
            

        else:  # try to maintain the legacy version without thinning stop
            plot_carbon["proportion_per_trees"] = plot_carbon.apply(
                lambda row: calculate_proportion(row, self.dict_plot_scenario), axis=1
            )
        
        # ENSURE THE NA PROPORTION PER TREES IS REMOVED BECAUSE THE CASE OF TWO ZONE THE SAME SPECIES CASE!
        plot_carbon = plot_carbon.dropna(subset=["proportion_per_trees"])

        # calculating biomass (ton) in actual proportion after thinning, and natural thinning
        plot_carbon["tCo2e_per_trees_proportion"] = (
            plot_carbon["tco2_per_tree"] * plot_carbon["proportion_per_trees"]
        )

        plot_carbon["total_csu_tCO2e_species"] = (
            plot_carbon["num_trees"] * plot_carbon["tCo2e_per_trees_proportion"]
        )

        plot_carbon["rotation_year"] = plot_carbon["year"]  # first rotation only
        plot_carbon["year"] = (
            plot_carbon["year"] + plot_carbon["year_start"] - 1
        )  # adjusted for the year started, if they planted at the year 1 meaning 1+1-1, but if started at year 2, meaning 1+2-1, the actual year planted at year 2
        plot_carbon["harvest_time"] = plot_carbon.apply(
            lambda x: is_species_allowed_to_harvest(x, self.dict_plot_scenario), axis=1
        )

        # plot_carbon['species'].unique()

        plot_carbon["cycle_harvest"] = 1
        plot_carbon["remnant_trees"] = False

        plot_carbon["allowed_cut_proportion_okay"] = False

        return {"plot_carbon": plot_carbon}

    # performing the harvest min. 40% retaining algorithm which recursively harvest the remnant (left over) trees in the following year
    def acquire_remnant_harvest(
        self,
        all_df,
        start_year,
        end_year,
        gap_harvest,
        dict_plot_scenario,
        k,
        tCO2e_retained=1,
        a=0,
        result_df=None,
        # dict_year_species_retained = {},
        # dict_year_species_num_trees_harvested = {},
        prev_num_trees_harvested=1,
        level_forloop=0,
    ):
        zone, plot_id, is_replanting, year_start = k
        try:
            # first we will for loop (iterate) over the plotIDs
            plot = k
            level_forloop = level_forloop  # i =0 is already used in the for loop in the main script, i=0 is the first cycle script

            # for displaying - debugging only
            filtered_column = [
                "Plot_ID",
                "area_ha",
                "species",
                "year",
                "total_csu_tCO2e_species",
                "rotation_year",
                "harvest_time",
                "remnant_trees",
                "allowed_cut_proportion_okay",
                "remnant_level",
                "num_trees",
                "num_trees_harvested",
                "num_trees_retained",
                "allowed_cut_csu",
                "tCO2e_retained",
                "cycle_harvest",
                "tCO2e_harvest_allowed_species",
                "tco2_per_tree",
            ]

            if result_df is None:
                result_df = pd.DataFrame()

            if start_year <= end_year:
                dict_all_species_tCO2e = {}
                list_all_species_tCO2e = []

                # dict_year_species_retained = {}
                species_retain_dict = {}

                # dict_year_species_num_trees_harvested = {}
                # species_num_trees_harvested = {}

                list_df_remnant = []
                replant_df_list = []

                # all_df_replant =  pd.DataFrame()

                list_condition = []

                for index_species, species_name in enumerate(
                    list(dict_plot_scenario[k].keys())
                ):
                    harvest_year = dict_plot_scenario[k][species_name][
                        "harvesting_year"
                    ]
                    rotation_year = harvest_cycle_rotation(
                        start_year, gap_harvest, harvest_year
                    ) - (
                        (year_start) - 1
                    )  # its coded now the key includes the info of year_start
                    print(f'year {start_year} of rotation year: {rotation_year} of species {species_name}')

                    # detect if species is harvested on the specific year before start year (actual year)
                    print('detect if species is harvested on the specific year before start year (actual year)')
                    if start_year - harvest_year >= 0:

                        filtered_df_prev_year = all_df[
                            (all_df["year"] == start_year - 1)
                            & (all_df["species"] == species_name)
                            & (all_df["Plot_ID"] == plot_id)
                            & (all_df["harvest_time"] == True)
                            & (all_df["year_start"] == year_start)
                            & (all_df["is_replanting"] == is_replanting)
                        ]

                        print(f"check query all_df[(all_df['year']=={start_year}-1) & (all_df['species']=={species_name}) & \n (all_df['Plot_ID']=={plot_id}) & (all_df['harvest_time']==True)] & (all_df['year_start']=={year_start}) & (all_df['is_replanting']=={is_replanting})")

                        # display(filtered_df_prev_year)

                        # when we are on the cycle 3 to the end of the project, case of inprosula, we will have multiple query of remnant trees and harvest time (prev. cycle), so we will skip the remnant_trees query
                        # if a > 0:

                        #     filtered_df_prev_year = all_df[(all_df['year']==start_year-1) & (all_df['species']==species_name) &
                        #                                               (all_df['Plot_ID']==k) & (all_df['remnant_trees']==True) & (all_df['harvest_time']==True)]

                        #     print("adding another query all_df['remnant_trees']==True")

                        # print('this is prev year for above query')
                        # # display(filtered_df_prev_year[filtered_column])
                        #
                        print(f'\nIteration recursive counter!!  a : {a}  LOOP NUMBER {a+1}')

                        # display(filtered_df_prev_year[filtered_column])

                        if not filtered_df_prev_year.empty:
                            exist_check = True
                            list_condition.append(exist_check)

                            # display(filtered_df_prev_year)

                            print(f'start with {species_name} at year {start_year} when the rotation of {rotation_year} in cycle {math.ceil(start_year/harvest_year)}')
                            print('still in acquire_remnant_harvest')
                            print(f"query of all_df[(all_df['year']=={start_year-1}) & (all_df['species']=={species_name}) & \n (all_df['Plot_ID']=={plot_id})]")

                            # this is only assume that this is the only row that come up of the same year, same species, but there are case when remnant trees replanted in the following year and the remnant harvest need to be harvested
                            # prev_harvest_time = filtered_df_prev_year['harvest_time'].iloc[0]
                            # tCO2e_retained = float(filtered_df_prev_year['tCO2e_retained'].iloc[0])
                            # prev_num_trees_harvested = int(filtered_df_prev_year['num_trees_harvested'].iloc[0])
                            # print('in a gap plant and harvest, rotation 0 means, we take a break, let the soil quality regenerated')
                            

                            # this is already check above, if the query is empty or not
                            list_prev_tCO2e_retained = filtered_df_prev_year[
                                "tCO2e_retained"
                            ].to_list()
                            list_prev_num_trees_harvested = filtered_df_prev_year[
                                "num_trees_harvested"
                            ].to_list()
                            list_prev_num_trees = filtered_df_prev_year[
                                "num_trees"
                            ].to_list()
                            list_prev_harvest_time = filtered_df_prev_year[
                                "harvest_time"
                            ].to_list()

                            list_prev_trees_retained = filtered_df_prev_year[
                                "num_trees_retained"
                            ].to_list()

                            # index the row, if we find the two query result (harvest time at the same year). example is in the second cycle and its remnant (delayed) replanting
                            dict_index_species_tCO2e = {}
                            dict_index_species_tree_retained = {}

                            # display(list_prev_num_trees_harvested)
                            # display(filtered_df_prev_year[filtered_column])

                            # now we will need to iterate to those item in the query result that assume to be more than one for some specific case (eg beginning of third cycle Inprosula)
                            # print(f'NOW WE WILL DO AN ITERATION OF THE QUERY RESULT: FOUND {len(list_prev_num_trees_harvested)} record')
                            for i in range(len(list_prev_num_trees_harvested)):
                                remnant_reset = []

                                print(f'this is year of {start_year} that have the data:')
                                print('----------------------------------------------------------------------')
                                print('prev TCO2e retained: ', list_prev_tCO2e_retained[i])
                                tCO2e_retained = list_prev_tCO2e_retained[i]
                                dict_index_species_tCO2e[i] = tCO2e_retained

                                print(f'process of {i+1} out of {len(list_prev_num_trees_harvested)} rows')
                                print(f'tCO2e_retained: {tCO2e_retained}')
                                print(f'species name: {species_name}')

                                prev_num_trees_harvested = (
                                    list_prev_num_trees_harvested[i]
                                )
                                # print('prev num harvested: ', prev_num_trees_harvested)
                                # prev_harvest_time = list_prev_harvest_time[i]

                                trees_retained = list_prev_trees_retained[i]
                                dict_index_species_tree_retained[i] = trees_retained

                                prev_num_trees = list_prev_num_trees[i]

                                # trees retained, no longer used
                                # species_retain_dict[species_name] = trees_retained
                                # dict_year_species_retained[start_year] = species_retain_dict

                                # trees harvested, not yet used
                                # species_num_trees_harvested[species_name] = prev_num_trees_harvested
                                # dict_year_species_num_trees_harvested[start_year] = species_num_trees_harvested

                                # will be used later to get sum(list_all_species_tCO2e), let's just put in the same list, this will take all later in aggregation, since the data is the same year
                                list_all_species_tCO2e.append(tCO2e_retained)

                                filtered_first_year = all_df[
                                    (all_df["year"] == year_start)
                                    & (all_df["species"] == species_name)
                                    & (all_df["Plot_ID"] == plot_id)
                                    & (all_df["is_replanting"] == is_replanting)
                                ]

                                if prev_num_trees_harvested > 0:
                                    df_remnant_standing = filtered_df_prev_year.copy()
                                    # additional adjustment - to avoid the double list row appending
                                    df_remnant_standing = df_remnant_standing[
                                        df_remnant_standing["tCO2e_retained"]
                                        == tCO2e_retained
                                    ]  # we used and assumed that tCO2e retained will never be the same
                                    trees_retained_batch = trees_retained

                                    # print(dict_year_species_retained)

                                    # # not relevant anymore since we do a recursive
                                    # if rotation_year != 1:
                                    #     trees_retained_batch = dict_year_species_retained[1][species_name]
                                    override_planting = False
                                    override_number = 0

                                    if a > 0:
                                        # since we do a recursive, next recursive will be used trees_retained directly from the remnant harvest
                                        # first_num_trees = float(filtered_first_year['num_trees'].iloc[0])
                                        # trees_retained_batch = first_num_trees - dict_year_species_retained[start_year][species_name]

                                        override_planting = True
                                        override_number = prev_num_trees_harvested

                                    print('trees_retained----------------------------------------------------\n\n', trees_retained_batch , '\n--------------------------------------------------')

                                    # print('displaying all df:')
                                    # display(all_df)

                                    print('----------------------------\n')
                                    #since we do a recursive, next recursive will be used trees_retained directly,so that the trees harvested are equal to the trees retained

                                    end_cycle_adjust = (
                                        start_year + harvest_year - 1
                                    )  # this is a default for non gap_harvest, eg 1,2,3,4,5,6,7,8,9,10 means that if start year 13 + 10-1 = [13,...,22] 22 is the harvest year
                                    if gap_harvest:
                                        end_cycle_adjust = start_year + harvest_year

                                    # print('starting to replant')
                                    all_df_replant = re_planting(
                                        self.config,
                                        all_df,
                                        gap_harvest,
                                        start_year,
                                        harvest_year,
                                        end_cycle_adjust,
                                        plot,
                                        species_name,
                                        trees_retained,
                                        override_planting=override_planting,
                                        override_number=override_number,
                                        a=a,
                                    )

                                    # display(all_df_replant[filtered_column])

                                    replant_df_list.append(all_df_replant)
                                    # print('replant data is appended')

                                    if tCO2e_retained > 0:
                                        df_remnant_standing["num_trees"] = (
                                            trees_retained
                                        )
                                        df_remnant_standing[
                                            "total_csu_tCO2e_species"
                                        ] = tCO2e_retained
                                        df_remnant_standing["year"] = start_year
                                        df_remnant_standing["remnant_trees"] = True
                                        df_remnant_standing["remnant_level"] = a + 1

                                        list_df_remnant.append(df_remnant_standing)

                                # print(f'now we will do a summary of the tCO2e of all the listed in the plot {k}')
                                # if index_species == len(dict_plot_scenario[k].keys())-1:

                            else:
                                # print(f'QUERY species: {species_name} IS NOT FOUND LETS CONTINUE THE LOOP in another species')
                                exist_check = False
                                list_condition.append(exist_check)

                                continue

                is_any_true = any(list_condition)

                if is_any_true or start_year == end_year:

                    # print(f'CALCULATING - AND SUMMARIZING OF YEAR {start_year} PLOT {k} - LOOOOP {a}')
                    result_df = pd.concat(
                        [all_df] + list_df_remnant + replant_df_list, ignore_index=True
                    )
                    # result_df = pd.concat([all_df]+list_df_remnant, ignore_index=True)

                    result_df["harvest_time"] = result_df.apply(
                        lambda x: is_species_allowed_to_harvest(
                            x, self.dict_plot_scenario
                        ),
                        axis=1,
                    ).astype(bool)

                    # print(len(dict_plot_scenario[k].keys())-1)
                    # print(dict_plot_scenario[k].keys())

                    # This is to get the Total carbon per CSU, categorized by year
                    # plot_carbon_csu = result_df.groupby(["Plot_ID", "year"])["total_csu_tCO2e_species"].agg([np.sum]).rename(columns=dict(sum='total_tCO2e'))
                    plot_carbon_csu = (
                        result_df.groupby(["Plot_ID", "year",'year_start','is_replanting'])[
                            "total_csu_tCO2e_species"
                        ]
                        .agg(["sum"])
                        .rename(columns=dict(sum="total_tCO2e"))
                    )
                    dict_csu_plot_year = plot_carbon_csu.to_dict("index")

                    # allowed cut is the 0.599 of harvest- 'able' trees (harvest_time) of the total CSU
                    result_df["allowed_cut_csu"] = result_df.apply(
                        lambda x: allowed_cut_harvest(
                            x,
                            x["Plot_ID"],
                            x["year"],
                            x["species"],
                            x['year_start'],
                            x['is_replanting'],
                            dict_csu_plot_year,
                            harvest_time=x["harvest_time"],
                            config=self.config,
                        ),
                        axis=1,
                    )

                    con_harvest_year_plot = (
                        (result_df["harvest_time"] == True)
                        & (result_df["year"] == start_year)
                        & (result_df["Plot_ID"] == plot_id)
                        & (result_df["year_start"] == year_start)
                        & (result_df["is_replanting"] == is_replanting)
                    )

                    filtered_df_harvest = result_df[con_harvest_year_plot]

                    filtered_df_harvest = (
                        filtered_df_harvest.groupby(["Plot_ID", "year", "year_start"])[
                            "total_csu_tCO2e_species"
                        ]
                        .sum()
                        .reset_index()
                        .rename(columns=dict(sum="total_tCO2e_harvest_cycle_csu"))
                    )
                    num_harvested_tCO2e = filtered_df_harvest["total_csu_tCO2e_species"]

                    # we have a problem if the df is empty so that we will handle in if
                    if not num_harvested_tCO2e.empty:
                        dict_all_species_tCO2e[start_year] = filtered_df_harvest[
                            "total_csu_tCO2e_species"
                        ].to_list()[0]

                    else:
                        dict_all_species_tCO2e[start_year] = sum(
                            list_all_species_tCO2e
                        )  # since at the last year, the list retained is not there so that we will use the merge join instead

                    result_df.loc[
                        con_harvest_year_plot, "total_tCO2e_harvest_cycle_csu"
                    ] = dict_all_species_tCO2e[start_year]

                    # PERFORM THE WEIGHING HARVEST!!
                    result_df["tCO2e_harvest_allowed_species"] = (
                        result_df["total_csu_tCO2e_species"]
                        / result_df["total_tCO2e_harvest_cycle_csu"]
                    ) * result_df["allowed_cut_csu"]

                    result_df["tCO2e_harvest_allowed_species"] = result_df.apply(
                        lambda x: (
                            x["total_csu_tCO2e_species"]
                            if x["tCO2e_harvest_allowed_species"]
                            >= x["total_csu_tCO2e_species"]
                            else (
                                (
                                    x["total_csu_tCO2e_species"]
                                    / x["total_tCO2e_harvest_cycle_csu"]
                                )
                                * x["allowed_cut_csu"]
                                if x["total_tCO2e_harvest_cycle_csu"] != 0
                                else x["total_tCO2e_harvest_cycle_csu"]
                            )
                        ),
                        axis=1,
                    )

                    result_df["allowed_cut_proportion_okay"] = result_df.apply(
                        lambda x: (
                            True
                            if x["allowed_cut_csu"]
                            >= x["total_tCO2e_harvest_cycle_csu"]
                            else False
                        ),
                        axis=1,
                    )

                    result_df["num_trees_harvested"] = result_df.apply(
                        lambda x: num_trees_harvest_allowed(
                            x["tCO2e_harvest_allowed_species"], x["tco2_per_tree"]
                        ),
                        axis=1,
                    )
                    result_df["num_trees_harvested"] = result_df[
                        "num_trees_harvested"
                    ].fillna(0)

                    # remember to fillna the num trees since csv has NaN
                    # result_df['num_trees'] = result_df['num_trees'].fillna(0)

                    result_df["num_trees_retained"] = result_df.apply(
                        lambda x: num_trees_retained(
                            x["num_trees"],
                            x["proportion_per_trees"],
                            x["num_trees_harvested"],
                            remnant_stock=x["remnant_trees"],
                        ),
                        axis=1,
                    ).fillna(0)
                    result_df["num_trees_retained"] = result_df[
                        "num_trees_retained"
                    ].fillna(0)

                    result_df["tCO2e_retained"] = (
                        result_df["num_trees_retained"] * result_df["tco2_per_tree"]
                    )
                    result_df["tCO2e_retained"] = result_df["tCO2e_retained"].fillna(0)

                    print(f'---------------------HEEEEYYYYY this the sum of all species tCO2e in plot {k}')
                    print(sum(list_all_species_tCO2e))
                    print('--------------------------------')
                    print('--------------------------------')

                    # display(result_df[(result_df['species']=='Anthocephalus cadamba | White Jabon - West Java') & (result_df['year']==11)])
                    # result_df['allowed_cut_proportion_okay'] = result_df.apply(lambda x: True if x['allowed_cut_csu'] > x['total_tCO2e_harvest_cycle_csu'] else False, axis=1)

                    # filtered_result = result_df[(result_df['year']<30) & ((result_df['species']=='Paraserianthes falcataria | Sengon - Java'))]
                    # filtered_result = result_df[(result_df['year']==12) & (result_df['species']=='Paraserianthes falcataria | Sengon - Java')]

                    # filtered_result = result_df[(result_df['year']<=21) & ((result_df['species']=='Anthocephalus cadamba | White Jabon - West Java'))]

                    # display(filtered_result[filtered_column])

                    # display(result_df)

                    # if start_year == 16:  #DEBUG
                    #     return result_df
                    #     raise BreakAllLoops  # Break out of all loops for testing and checking

                    # this is required to return the result to prevent None Object!!
                    if start_year == end_year:
                        return result_df

                    result_df = self.acquire_remnant_harvest(
                        result_df,
                        start_year + 1,
                        end_year,
                        gap_harvest,
                        dict_plot_scenario,
                        k,
                        tCO2e_retained=tCO2e_retained,
                        a=a + 1,
                        #  dict_year_species_retained = dict_year_species_retained,
                        #  dict_year_species_num_trees_harvested = dict_year_species_num_trees_harvested,
                        prev_num_trees_harvested=prev_num_trees_harvested,
                        level_forloop=level_forloop,
                    )

                elif start_year != end_year:
                    # print(f'Loop {a+1} does not exist!!!!!!!!! YEAR: {start_year}')
                    # print('MOVING TO ANOTHER QUERY in the next loop!!!!')

                    return self.acquire_remnant_harvest(
                        all_df,
                        start_year + 1,
                        end_year,
                        gap_harvest,
                        dict_plot_scenario,
                        k,
                        tCO2e_retained=tCO2e_retained,
                        a=a + 1,
                        # dict_year_species_retained = dict_year_species_retained,
                        # dict_year_species_num_trees_harvested = dict_year_species_num_trees_harvested,
                        prev_num_trees_harvested=prev_num_trees_harvested,
                        level_forloop=level_forloop,
                    )

                return result_df

        except BreakAllLoops:
            pass

    def ex_ante(self) -> pd.DataFrame:
        # EXECUTING for loop, iterating to the calc, recursively for the remnant trees harvesting cycle

        plot_carbon_func = self.plot_carbon_melt()
        plot_carbon = plot_carbon_func["plot_carbon"]
        # plot_sum = self.plot_seedling["plot_sum"]
        plot_sum = self.plot_sum

        duration_project = self.config["duration_project"]
        gap_harvest = self.config["gap_harvest"]

        # hotfix for the zone
        plot_id_zone = (
            plot_sum[["Plot_ID", "zone"]]
            .drop_duplicates()
            .set_index("Plot_ID")["zone"]
            .to_dict()
        )

        # for loop over the harvesting cycle, after the first cycle, nested for loop on the list of different cycle in plot
        list_df_per_year_csu = []

        try:
            # filtered_df = plot_carbon.copy()
            for (
                plot_id,
                is_replanting_scenario,
            ) in (
                self.plot_grouped_dict_scenario.items()
            ):  # k is Plot_ID, loop over the csu

                # print('this is the iteration',plot_id)
                plot_id_harvest_list_pair = {}
                list_one_csu_grouped_df = []
                for is_replanting, scenario in is_replanting_scenario.items():
                    #         print(is_replanting)
                    year_start_list = scenario["year_start_list"]
                    #         print(year_start_list)
                    harvest_list = scenario["harvest_list"]
                    #         print(harvest_list)

                    harvest_dict = {}
                    # reset the object after the different plot id iterated
                    filtered_df = None
                    for i in range(len(year_start_list)):
                        # print(harvest_list[i])
                        harvest_dict[year_start_list[i]] = harvest_list[i]
                        for z in range(len(harvest_list[i])):
                            if z == 0:  # first cycle
                                filtered_df = plot_carbon.copy()
                                filtered_df = filtered_df[
                                    (filtered_df["year"] <= harvest_list[i][z])
                                    & (filtered_df["Plot_ID"] == plot_id)
                                    & (filtered_df["is_replanting"] == is_replanting)
                                    & (filtered_df["year_start"] == year_start_list[i])
                                ]
                                # print(f"query ==> filtered_df[(filtered_df['year'] <= {harvest_list[i][z]}) &  \
                                #                         (filtered_df['Plot_ID'] == {plot_id}) & \
                                #                         (filtered_df['is_replanting'] == {is_replanting}) & \
                                #                         (filtered_df['year_start'] == {year_start_list[i] }) ]")

                                #                     display(filtered_df)
                                list_one_csu_grouped_df.append(filtered_df)
                            else:
                                pass
                    #
                    # print(harvest_dict)

                    # to reuse again later
                    plot_id_harvest_list_pair[is_replanting] = harvest_dict

                df_csu_first_cycle = pd.concat(
                    list_one_csu_grouped_df, ignore_index=True
                )
                filtered_df = df_csu_first_cycle.copy()
                # display(df_csu_first_cycle)

                # restructure and create dict object of the group by csu # do not reset index of this
                # plot_carbon_csu = df_csu_first_cycle.groupby(["Plot_ID","year"])["total_csu_tCO2e_species"].agg([np.sum]).rename(columns=dict(sum='total_tCO2e')) # its okay, still works, this will generate only one plotid, and total its csu tco2e
                plot_carbon_csu = (
                    df_csu_first_cycle.groupby(["Plot_ID", "year","year_start","is_replanting"])[  # we will include the additional field, to match up the percentage algorithm
                        "total_csu_tCO2e_species"
                    ]
                    .agg(["sum"])
                    .rename(columns=dict(sum="total_tCO2e"))
                )  # its okay, still works, this will generate only one plotid, and total its csu tco2e
                dict_csu_plot_year = plot_carbon_csu.to_dict("index")

                # IN ORDER TO USE THE HARVEST_TIME FIELD WE SHOULD ADD IT TO THE ARGUMENT
                filtered_df["allowed_cut_csu"] = filtered_df.apply(
                    lambda x: allowed_cut_harvest(
                        x,
                        x["Plot_ID"],
                        x["year"],
                        x["species"],
                        x['year_start'],
                        x['is_replanting'],
                        dict_csu_plot_year,
                        harvest_time=x["harvest_time"],
                        config=self.config,
                    ),
                    axis=1,
                )
                # display(filtered_df)

                # #     # Filter the dataframe based on the allowed species. # SET THE YEAR INTO ROTATION_YEAR!
                # #     filtered_df_allowed_species = filtered_df[filtered_df.apply(lambda row: is_species_allowed_to_harvest(self.dict_plot_scenario, row['Plot_ID'],row['species'], row['rotation_year']), axis=1).fillna(False)]

                # #     #input_scenario_species
                # #     list_df_filtered_harvest_only_species = []

                # #     input_scenario_species_per_plot = self.dict_plot_scenario[k]

                # #     for species_name,conf in input_scenario_species_per_plot.items():
                # #         if input_scenario_species_per_plot[species_name]['harvesting_year'] <= duration_project:  # we might need to comeback to this and aggregate the info later
                # #             list_df_filtered_harvest_only_species.append(filtered_df[(filtered_df['species']==species_name) & (filtered_df['rotation_year']==input_scenario_species_per_plot[species_name]['harvesting_year'])])

                plot_carbon_harvest_cycle_only = filtered_df.copy()
                plot_carbon_harvest_cycle_only = plot_carbon_harvest_cycle_only[
                    plot_carbon_harvest_cycle_only["harvest_time"] == True
                ]

                # grouping based on plot, only to generate the total harvestable tCO2e
                # plot_carbon_harvest_cycle_only_sum = plot_carbon_harvest_cycle_only.groupby(["Plot_ID","Plot_Name", "year"])["total_csu_tCO2e_species"].agg([np.sum]).rename(columns=dict(sum='total_tCO2e_harvest_cycle_csu'))
                # plot_carbon_harvest_cycle_only_sum = plot_carbon_harvest_cycle_only.groupby(["Plot_ID", "year","year_start"])["total_csu_tCO2e_species"].agg([np.sum]).rename(columns=dict(sum='total_tCO2e_harvest_cycle_csu'))
                plot_carbon_harvest_cycle_only_sum = (
                    plot_carbon_harvest_cycle_only.groupby(
                        ["Plot_ID", "year", "year_start","is_replanting"]
                    )["total_csu_tCO2e_species"]
                    .agg(["sum"])
                    .rename(columns=dict(sum="total_tCO2e_harvest_cycle_csu"))
                )
                plot_carbon_harvest_cycle_only_sum = (
                    plot_carbon_harvest_cycle_only_sum.reset_index()
                )

                # display(plot_carbon_harvest_cycle_only_sum)
                # display(filtered_df)
                filtered_df_harvest = pd.merge(
                    filtered_df,
                    plot_carbon_harvest_cycle_only_sum,
                    how="left",
                    left_on=["Plot_ID", "year", "year_start",'is_replanting'],
                    right_on=["Plot_ID", "year", "year_start",'is_replanting'],
                )
                # display(filtered_df_harvest)

                filtered_df = pd.merge(
                    filtered_df,
                    filtered_df_harvest[
                        [
                            "Plot_ID",
                            "year",
                            "year_start",
                            'is_replanting',
                            "species",
                            "total_tCO2e_harvest_cycle_csu",
                        ]
                    ],
                    how="left",
                    left_on=["Plot_ID", "species", "year", "year_start",'is_replanting'],
                    right_on=["Plot_ID", "species", "year", "year_start",'is_replanting'],
                )

                filtered_df = filtered_df.copy()
                # display(filtered_df)

                # IMPLEMENTING THE WEIGHING PROPORTION!!!
                filtered_df["tCO2e_harvest_allowed_species"] = (
                    filtered_df["total_csu_tCO2e_species"]
                    / filtered_df["total_tCO2e_harvest_cycle_csu"]
                ) * filtered_df["allowed_cut_csu"]

                # num_trees in the df is num_trees init or in the first year, not reflected yet in certain year, need to adjust based on the proportion, or we should use TTB per tree, NOT TCO2e per tree proportion
                filtered_df["num_trees_harvested"] = filtered_df.apply(
                    lambda x: num_trees_harvest_allowed(
                        x["tCO2e_harvest_allowed_species"], x["tco2_per_tree"]
                    ),
                    axis=1,
                )
                filtered_df["num_trees_harvested"] = filtered_df[
                    "num_trees_harvested"
                ].fillna(0)

                # this is to ensure that the NaN is converted to 0
                # filtered_df['num_trees'] = filtered_df['num_trees'].fillna(0)
                filtered_df["num_trees_retained"] = filtered_df.apply(
                    lambda x: num_trees_retained(
                        x["num_trees"],
                        x["proportion_per_trees"],
                        x["num_trees_harvested"],
                        remnant_stock=x["remnant_trees"],
                    ),
                    axis=1,
                ).fillna(0)
                filtered_df["num_trees_retained"] = filtered_df[
                    "num_trees_retained"
                ].fillna(0)

                filtered_df["tCO2e_retained"] = (
                    filtered_df["num_trees_retained"] * filtered_df["tco2_per_tree"]
                )
                filtered_df["tCO2e_retained"] = filtered_df["tCO2e_retained"].fillna(0)
                filtered_df["remnant_level"] = 0

                # # now we modify the previous formula, just change with the harvest cycle tree species only, not entire total_csu_tCO2e_species
                # filtered_df['allowed_cut_proportion_okay'] = filtered_df.apply(lambda x: True if x['allowed_cut_csu'] > x['total_tCO2e_harvest_cycle_csu'] else False, axis=1)
                filtered_df["allowed_cut_proportion_okay"] = filtered_df.apply(
                    lambda x: (
                        False
                        if pd.isna(x["allowed_cut_csu"])
                        or pd.isna(x["total_tCO2e_harvest_cycle_csu"])
                        or x["allowed_cut_csu"] < x["total_tCO2e_harvest_cycle_csu"]
                        else True
                    ),
                    axis=1,
                )

                # cleaning the data if number of trees is 0 means that in that particular plot, the seedling is not planted, which means the data row is irrelevant
                # Remove rows where 'num_trees' is equal to 0
                filtered_df = filtered_df[filtered_df["num_trees"] != 0]
                filtered_df = filtered_df.copy()
                # change the number to be make sense, if total_csu species > existing stock then it will be the same to the existing stock -> because the formula is from allowed_harvest / emission factor
                filtered_df["tCO2e_harvest_allowed_species"] = filtered_df.apply(
                    lambda x: (
                        x["total_csu_tCO2e_species"]
                        if x["tCO2e_harvest_allowed_species"]
                        >= x["total_csu_tCO2e_species"]
                        or pd.isna(x["allowed_cut_csu"])
                        else (
                            (
                                x["total_csu_tCO2e_species"]
                                / x["total_tCO2e_harvest_cycle_csu"]
                            )
                            * x["allowed_cut_csu"]
                            if x["total_tCO2e_harvest_cycle_csu"] != 0
                            else x["total_tCO2e_harvest_cycle_csu"]
                        )
                    ),
                    axis=1,
                )

                # for displaying - debugging only
                filtered_column = [
                    "Plot_ID",
                    "Plot_Name",
                    "area_ha",
                    "species",
                    "year",
                    'year_start',
                    'is_replanting',
                    "total_csu_tCO2e_species",
                    "rotation_year",
                    "harvest_time",
                    "remnant_trees",
                    "allowed_cut_proportion_okay",
                    "remnant_level",
                    "num_trees",
                    "num_trees_harvested",
                    "num_trees_retained",
                    "allowed_cut_csu",
                    "tCO2e_retained",
                    "cycle_harvest",
                    "tCO2e_harvest_allowed_species",
                ]

                # "['remnant_level', 'num_trees_harvested', 'num_trees_retained', 'tCO2e_retained', 'tCO2e_harvest_allowed_species']

                # display(filtered_df[filtered_column])
                # break
                ##################################

                print("first cycle is done iterated for plot ", plot_id)
                print(
                    "------------------------------------------------------------------------------------------"
                )

                ## debugging check for PT SRA updated 3 year_start
                # if k not in ['720_Nr Large Tree Expost_2', '702_Nr Tree Evidence Expost_3', '729_Nr Tree Evidence Expost_3']:
                #     continue # we will move to only above plot for debugging purpose

                ########################################
                # THIS IS SECOND HARVEST CYCLE and next following CYCLE
                # iterated to the next item in the list based on what?
                # print(f'the next cycle check based on this {plot_id_harvest_list_pair}')
                print(
                    f'next-cycle for {self.config["harvesting_max_percent"]}% harvest allowed or must retained tco2 per csu stock {100-self.config["harvesting_max_percent"]}%'
                )
                for is_replanting, dict_harvest in plot_id_harvest_list_pair.items():
                    for year_start, harvest_list in dict_harvest.items():
                        for i in range(len(harvest_list)):
                            if i == len(harvest_list) - 1:
                                continue
                            else:
                                year_start_arr = [
                                    year_start
                                    for k, v in plot_id_harvest_list_pair.items()
                                    for year_start, harvest_list in v.items()
                                ]
                                start_year_arr = [
                                    harvest_list[z] + 1
                                    for k, v in plot_id_harvest_list_pair.items()
                                    for year_start, harvest_list in v.items()
                                    for z in range(len(harvest_list))
                                    if z == i
                                ]
                                end_year_arr = [
                                    harvest_list[z]
                                    for k, v in plot_id_harvest_list_pair.items()
                                    for year_start, harvest_list in v.items()
                                    for z in range(len(harvest_list))
                                    if z == i + 1
                                ]
                                print(f'dealing with the next. harvest cycle, from year array year_start {year_start_arr} of starting year {start_year_arr} \
                                        \n to array {end_year_arr} in plot {plot_id} and is_replanting {is_replanting} \n-----------------------------------------------------------------')
                                start_year = harvest_list[i] + 1
                                end_year = harvest_list[i + 1]
                                print(f'dealing with the next. harvest cycle, from {start_year} to {end_year} in plot {plot_id} \n-----------------------------------------------------------------')

                                filtered_df_i = plot_carbon.copy()
                                filtered_df_i = filtered_df_i[
                                    (filtered_df_i["year"] >= start_year)
                                    & (filtered_df_i["year"] <= end_year)
                                    & (filtered_df_i["Plot_ID"] == plot_id)
                                    & (filtered_df_i["is_replanting"] == is_replanting)
                                    & (filtered_df_i["year_start"] == year_start)
                                ]

                                # apply recursively adding/ concat
                                filtered_df = pd.concat(
                                    [filtered_df, filtered_df_i], ignore_index=True
                                )
                                filtered_df = self.acquire_remnant_harvest(
                                    filtered_df,
                                    start_year,
                                    end_year,
                                    gap_harvest,
                                    self.dict_plot_scenario,
                                    (
                                        plot_id_zone[plot_id],
                                        plot_id,
                                        is_replanting,
                                        year_start,
                                    ),
                                    level_forloop=i,
                                )
                                # display(filtered_df)
                                # print('done function acquire_remnant_harvest')

                                # if i == 1: # adding another +1 in i because we want to proceed to filtered_df
                                #     raise BreakAllLoops  # Break out of all loops for testing and checking

                                # print(f'all-recursively remnant identified in year from {start_year} to {end_year} \n----------------------------------------------------------\n')
                                # print('------------------------------------------------------------------------------------------------------------------------------------------')

                # # store to the list and merged it later
                list_df_per_year_csu.append(filtered_df)
                print("done next plot!")

            # concat the list into one single df from above list df
            all_df_merged = pd.concat(list_df_per_year_csu, ignore_index=True)
            print("===============================================================")
            print("merging all the result")
            # all_df_merged.columns
            all_df_merged = all_df_merged.drop(
                "tco2_per_tree", axis=1
            )  # ensure that the TTB formula is using the correct rotation year

            return all_df_merged

        except BreakAllLoops:
            pass
