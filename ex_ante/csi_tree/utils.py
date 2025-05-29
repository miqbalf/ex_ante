import math

import numpy as np
import pandas as pd


def create_nested_dict(df):
    nested_dict = {}
    for (zone, is_replanting, year_start, plot_id), group in df.groupby(
        ["zone", "is_replanting", "year_start", "Plot_ID"]
    ):
        species_list = group["species"].tolist()
        if zone not in nested_dict:
            nested_dict[zone] = {}
        if is_replanting not in nested_dict[zone]:
            nested_dict[zone][is_replanting] = {}
        if year_start not in nested_dict[zone][is_replanting]:
            nested_dict[zone][is_replanting][year_start] = {}
        nested_dict[zone][is_replanting][year_start][plot_id] = species_list
    return nested_dict


def merge_dictionaries_zone(zone_dict, dict_scenario):
    adjusted_format = {}
    new_name = False
    for k, v in dict_scenario.items():
        if k == "non_replanting":
            new_name = False
        elif k == "replanting":
            new_name = True
        adjusted_format[new_name] = v
    merged_dict = {}
    try:
        for zone, zone_data in zone_dict.items():
            merged_dict[zone] = {}
            for is_replanting, year_data in zone_data.items():
                merged_dict[zone][is_replanting] = {}
                for year, plot_data in year_data.items():
                    merged_dict[zone][is_replanting][year] = {}
                    for plot_id, species_list in plot_data.items():
                        merged_dict[zone][is_replanting][year][plot_id] = {}
                        for species in species_list:
                            merged_dict[zone][is_replanting][year][plot_id][species] = (
                                adjusted_format[is_replanting][zone][species]
                            )
    except Exception as e:
        raise ValueError(
            f" {e} \n make sure that all the plot distribution have the scenario, especially if in ex-post there are trees species in other zone (example in inprosula!)"
        )

    return merged_dict


def transform_plot_base_dict(data):
    new_data = {}
    for zone_type, zone_data in data.items():
        for is_replanting, year_start_data in zone_data.items():
            for year_start, plot_data in year_start_data.items():
                for plot_id, species_data in plot_data.items():
                    if plot_id not in new_data:
                        new_data[plot_id] = {}
                    if is_replanting not in new_data[plot_id]:
                        new_data[plot_id][is_replanting] = {
                            "year_start_list": [],
                            "species_list_scenario": [],
                        }
                    new_data[plot_id][is_replanting]["year_start_list"].append(
                        year_start
                    )
                    new_data[plot_id][is_replanting]["species_list_scenario"].append(
                        species_data
                    )
    return new_data


def is_species_allowed_to_harvest_calc(
    dict_plot_scenario, plot_id, species_name, rotation_year
):
    if dict_plot_scenario[plot_id][species_name]["harvesting_year"] == rotation_year:
        return True
    else:
        return False


def is_species_allowed_to_harvest(row, dict_plot_scenario):
    tuple_columns_series = (
        row["zone"],
        row["Plot_ID"],
        row["is_replanting"],
        row["year_start"],
    )

    is_harvest = is_species_allowed_to_harvest_calc(
        dict_plot_scenario, tuple_columns_series, row["species"], row["rotation_year"]
    )
    return is_harvest


# creating a calculation of proportion per tree of dataframe row
def calc_proportion_per_tree(input_dict, plot_id, species_name, year):
    output = None
    dict_species_in_plot = input_dict[plot_id]
    # ensure the calculation only on those species
    for species, conf in dict_species_in_plot.items():
        if species == species_name:
            if (
                year > dict_species_in_plot[species]["harvesting_year"]
            ):  # this one look at the existing input_scenario_species
                output = None
            else:
                # applying mortality rate and natural thinning
                if year >= 1:
                    output = (
                        1
                        - (
                            dict_species_in_plot[species]["mortality_percent"] / 100
                        )  # * 1
                    ) * (
                        1 - (dict_species_in_plot[species]["natural_thinning"] / 100)
                    ) ** (
                        year - 1
                    )

                    # now we want to apply the manual thinning approach
                    if dict_species_in_plot[species]["frequency_manual_thinning"] >= 1:
                        for i in range(
                            dict_species_in_plot[species]["frequency_manual_thinning"]
                        ):
                            if (
                                year
                                >= dict_species_in_plot[species][
                                    f"thinning_cycle{i+1}_year:"
                                ]
                            ):  # is not a good idea for a description widget to put ':' here, get the uneccessary silly bug if you forgot to add ':'
                                output = output * (
                                    1
                                    - dict_species_in_plot[species][
                                        f"thinning_cycle{i+1}_percent:"
                                    ]
                                    / 100
                                )

    return output


def calculate_proportion(row, dict_plot_scenario):
    tuple_columns_series = (
        row["zone"],
        row["Plot_ID"],
        row["is_replanting"],
        row["year_start"],
    )

    proportion = calc_proportion_per_tree(
        dict_plot_scenario, tuple_columns_series, row["species"], row["year"]
    )
    return proportion


def simulate_all_proportions_for_species(
    species_conf, initial_num_trees, density_threshold_trees
):
    """
    Simulates proportions year-by-year up to harvesting year for one species/plot,
    including initial mortality and density-dependent natural thinning stop.

    Args:
        species_conf (dict): Configuration dictionary for the species.
        initial_num_trees (float): Initial number of trees planted.
        density_threshold_trees (float): Minimum tree count required for natural thinning.

    Returns:
        dict: {year: proportion}, or {} if config error occurs.
    """
    results = {}
    # --- Get config values safely with .get() and type conversion ---
    try:
        harvesting_year = int(
            species_conf.get("harvesting_year", 0)
        )  # Default 0 if missing
        # Handle missing required values gracefully by returning {}
        if harvesting_year <= 0:
            print(
                f"Warning/Error: Invalid or missing harvesting_year ({harvesting_year}). Cannot simulate."
            )
            return {}

        initial_mortality_rate = float(species_conf.get("mortality_percent", 0)) / 100.0
        natural_thinning_rate = float(species_conf.get("natural_thinning", 0)) / 100.0
        frequency = int(species_conf.get("frequency_manual_thinning", 0))
        if frequency < 0:
            frequency = 0  # Ensure non-negative

        manual_thinnings = {}
        if frequency > 0:
            for i in range(frequency):
                # !!! ASSUMPTION: Keys DO NOT have trailing colon (:) !!!
                # If they DO, change these keys back e.g. f"thinning_cycle{i+1}_year:"
                year_key = f"thinning_cycle{i+1}_year"
                percent_key = f"thinning_cycle{i+1}_percent"

                print('species_conf: ', species_conf)

                thin_year_val = species_conf.get(year_key)
                thin_percent_val = species_conf.get(percent_key)

                # Check if keys exist before trying to convert
                if thin_year_val is None:
                    print(
                        f"Warning: Missing manual thinning key '{year_key}'. Skipping cycle {i+1}."
                    )
                    continue
                if thin_percent_val is None:
                    print(
                        f"Warning: Missing manual thinning key '{percent_key}'. Skipping cycle {i+1}."
                    )
                    continue

                # Try conversion after checking existence
                try:
                    thin_year = int(thin_year_val)
                    thin_rate = float(thin_percent_val) / 100.0
                    if thin_year >= 0 and thin_rate >= 0:
                        manual_thinnings[thin_year] = manual_thinnings.get(
                            thin_year, 1.0
                        ) * (1.0 - thin_rate)
                    else:
                        print(
                            f"Warning: Invalid value for manual thinning cycle {i+1} (Year: {thin_year}, Rate: {thin_rate}). Skipping."
                        )
                except (ValueError, TypeError) as conv_err:
                    print(
                        f"Warning: Error converting manual thinning data for cycle {i+1}: {conv_err}. Skipping."
                    )
                    continue

    except (ValueError, TypeError, KeyError) as config_err:
        # Catch potential errors during config reading
        print(f"Error reading configuration: {config_err}. Cannot simulate.")
        return {}  # Return empty dict

    # --- Handle initial state and edge cases ---
    if initial_num_trees <= 0:
        print("Warning: Initial number of trees is zero or negative.")
        # Return 0 proportion for all years up to harvesting year
        return {yr: 0.0 for yr in range(harvesting_year + 1)}

    # Year 0: Proportion is 1.0
    results[0] = 1.0
    current_proportion = 1.0

    # Apply initial mortality (affects proportion at the *start* of year 1)
    current_proportion *= 1.0 - initial_mortality_rate
    # Ensure proportion doesn't go below zero right away
    current_proportion = max(0.0, current_proportion)

    # --- Simulation Loop ---
    # Loop from year 1 up to harvesting year
    for year_step in range(1, harvesting_year + 1):

        # State at the START of year_step (after previous year's events)
        trees_at_start_of_year = initial_num_trees * current_proportion

        # Condition for applying natural thinning this year
        apply_natural_thinning = trees_at_start_of_year >= density_threshold_trees

        # Apply Natural Thinning? (Only from Year 2 onwards)
        if apply_natural_thinning and year_step > 1:
            current_proportion *= 1.0 - natural_thinning_rate

        # Apply Manual Thinning scheduled for this exact year?
        if year_step in manual_thinnings:
            current_proportion *= manual_thinnings[year_step]

        # Ensure proportion is not negative and store result for the END of year_step
        current_proportion = max(0.0, current_proportion)
        results[year_step] = current_proportion

        # Optimization: if zero, fill remaining years and break
        if current_proportion == 0.0:
            for yr in range(year_step + 1, harvesting_year + 1):
                results[yr] = 0.0
            break

    return results  # Return dictionary {year: proportion}


def calculate_and_merge_proportions(plot_carbon_df, scenario_dict, simulation_func):
    """
    Pre-calculates proportions using a simulation function and merges them
    onto the main DataFrame.

    Args:
        plot_carbon_df (pd.DataFrame): The main DataFrame with plot/species/year rows.
                                        Must contain columns for plot ID tuple components,
                                        species, year, area_ha, avgtrees_per_ha (initial density?),
                                        calc_max_density.
        scenario_dict (dict): Dictionary containing species configuration keyed by plot_id_tuple.
        simulation_func (callable): The simulation function to call, e.g.,
                                    simulate_all_proportions_for_species.
                                    Expected signature: (species_conf, initial_num_trees, density_threshold_trees) -> dict[int, float]

    Returns:
        pd.DataFrame: The input DataFrame with the 'proportion_per_trees' column added/updated.
    """
    print(f"Starting pre-calculation using {simulation_func.__name__}...")
    all_proportions_list = []
    processed_combinations = set()

    # Define columns needed to identify unique plot/species and for calculation
    # Adjust these names if they differ in your DataFrame!
    id_cols_for_tuple = ["zone", "Plot_ID", "is_replanting", "year_start"]
    species_col = "species"
    area_col = "area_ha"
    initial_density_col = "avgtrees_per_ha"  # !! VERIFY THIS !!
    max_density_col = "calc_max_density"
    year_col = "year"  # Must exist in plot_carbon_df

    required_row_cols = id_cols_for_tuple + [
        species_col,
        area_col,
        initial_density_col,
        max_density_col,
    ]

    # Check required columns exist in input df
    if not all(col in plot_carbon_df.columns for col in required_row_cols + [year_col]):
        missing = [
            col
            for col in required_row_cols + [year_col]
            if col not in plot_carbon_df.columns
        ]
        print(f"Error: Input DataFrame is missing required columns: {missing}")
        # Return original df or raise error? Let's add NaN column and return
        plot_carbon_df["proportion_per_trees"] = np.nan
        return plot_carbon_df

    # Use .drop_duplicates to get unique plot/species combos
    unique_plot_species_rows = plot_carbon_df.drop_duplicates(
        subset=id_cols_for_tuple + [species_col]
    )
    print(
        f"Processing {len(unique_plot_species_rows)} unique plot/species combinations..."
    )

    for _, row in unique_plot_species_rows.iterrows():
        try:
            print('unique_plot_species_rows: ', unique_plot_species_rows)
            # Construct plot_id_tuple
            plot_id_tuple = tuple(row[col] for col in id_cols_for_tuple)
            species = row[species_col]
            combination_key = (plot_id_tuple, species)

            if combination_key in processed_combinations:
                continue

            species_conf = scenario_dict.get(plot_id_tuple, {}).get(species)
            if not species_conf:
                print(f"Warning: No config found for {combination_key}. Skipping.")
                continue

            # Calculate inputs for simulation
            initial_num_trees = row[initial_density_col] * row[area_col]
            density_threshold_trees = row[max_density_col] * row[area_col]
            area_ha = row[area_col]  # For validation

            if (
                np.isnan(initial_num_trees)
                or np.isnan(density_threshold_trees)
                or area_ha <= 0
            ):
                print(
                    f"Warning: Skipping {combination_key} due to invalid inputs (NaN/Zero Area)."
                )
                continue

            # Run simulation for all years
            yearly_proportions = simulation_func(
                species_conf, initial_num_trees, density_threshold_trees
            )

            # Store results
            for year, proportion in yearly_proportions.items():
                result_row = {
                    col: row[col] for col in id_cols_for_tuple
                }  # Copy identifiers
                result_row[species_col] = species
                result_row[year_col] = year
                result_row["proportion_per_trees"] = proportion
                all_proportions_list.append(result_row)

            processed_combinations.add(combination_key)

        except Exception as e:
            # Log more specific errors if possible (e.g., KeyError for missing columns)
            print(
                f"Warning: Error processing combination {combination_key} (Row index might be misleading due to drop_duplicates): {e}"
            )
            continue  # Skip combination on error

    if not all_proportions_list:
        print("Warning: No proportions were calculated. Adding NaN column.")
        plot_carbon_df["proportion_per_trees"] = np.nan
        return plot_carbon_df

    # Create DataFrame from results
    proportions_df = pd.DataFrame(all_proportions_list)

    # --- Merge results back ---
    merge_cols = id_cols_for_tuple + [species_col, year_col]

    # Ensure 'year' column is correct type in both DFs
    if year_col not in plot_carbon_df.columns:
        # This check is technically redundant due to earlier check, but safe
        raise ValueError(
            f"Main DataFrame 'plot_carbon_df' must have a '{year_col}' column."
        )
    plot_carbon_df[year_col] = plot_carbon_df[year_col].astype(int)
    if not proportions_df.empty:
        proportions_df[year_col] = proportions_df[year_col].astype(int)

    # Remove old proportion column if it exists to avoid duplicate columns after merge
    if "proportion_per_trees" in plot_carbon_df.columns:
        plot_carbon_df = plot_carbon_df.drop(columns=["proportion_per_trees"])

    # Perform the merge
    merged_df = pd.merge(plot_carbon_df, proportions_df, on=merge_cols, how="left")
    print("Merge completed.")
    if merged_df["proportion_per_trees"].isnull().any():
        print(
            "Warning: Some rows did not match calculated proportions (check merge keys and year ranges)."
        )
        # merged_df['proportion_per_trees'].fillna(0, inplace=True) # Optional: Fill unmatched rows

    return merged_df


# Assume simulate_all_proportions_for_species is defined (with density stop logic)
# Assume simulate_all_proportions_NO_STOP is defined (without density stop logic) - PLACEHOLDER


def simulate_all_proportions_NO_STOP(
    species_conf, initial_num_trees, density_threshold_trees
):
    # Placeholder: Implement simulation WITHOUT the density check
    print(f"Warning: {simulate_all_proportions_NO_STOP.__name__} not implemented.")
    harvesting_year = species_conf.get("harvesting_year", 0)
    return {yr: 0.5 for yr in range(harvesting_year + 1)}  # Return dummy data


# following tree c-sink extending the next cycle process
def allowed_cut_harvest(
    df, plot_id, year, species, dict_csu_plot_year, harvest_time=False, config={}
):
    # creating the conditional based on the above condition (harvest gap) and get the mod method, to ensure the repetitive cycle, next replanting is accounted
    csu = plot_id

    duration_project = config["duration_project"]
    dict_plot_scenario = config["dict_plot_scenario"]
    harvesting_max_percent = config["harvesting_max_percent"]

    if harvest_time:
        # should be retained >= 40% -> allowed cut - < 60%, in csu level summary
        prop_60_percent_csu = dict_csu_plot_year[(csu, year)]["total_tCO2e"] * float(
            harvesting_max_percent / 100.00
        )
        return prop_60_percent_csu

    else:
        return 0.0


def num_trees_harvest_allowed(tco2_harvest, tco2_per_tree):
    num_trees = None
    if pd.notna(tco2_harvest):
        if tco2_per_tree != 0:
            num_trees = max(
                1, round((tco2_harvest / tco2_per_tree) * 100000000 / 100000000)
            )  # if there is carbon, we consider as 1 tree
        else:
            num_trees = 0
    return num_trees


def num_trees_retained(
    num_trees, proportion_share_time, harvest_num, remnant_stock=False
):  # here the proportion_share_time will not work on the remnant trees, need to re-adjust
    num_trees_retained = None
    if pd.notna(harvest_num):
        if remnant_stock == False:
            num_trees_retained = max(
                0,
                round(
                    (num_trees * proportion_share_time - harvest_num)
                    * 100000000
                    / 100000000
                ),
            )  # 100000000/100000000 is for rounding precision

        elif pd.notna(harvest_num) & remnant_stock == True:
            num_trees_retained = max(
                0, round((num_trees - harvest_num) * 100000000 / 100000000)
            )  # 100000000/100000000 is for rounding precision

        # if we do clear cut (the other trees cover > 40% tCO2e in the same csu)
        if num_trees_retained <= 0:
            num_trees_retained = 0
    return num_trees_retained


def harvest_cycle_rotation(actual_year, gap_harvest, harvest_year):
    rotation_year = 1
    if gap_harvest:
        rotation_year = actual_year % (harvest_year + 1)
        return rotation_year
    else:
        rotation_year = actual_year % (harvest_year)
        return rotation_year


# (11-11) % 20
# function utils
def fill_harvest_cycle(harvest_year, rotation_year):
    if rotation_year == harvest_year:
        return True
    else:
        return False


def re_planting(
    config,
    all_df,
    gap_harvest,
    actual_year,
    harvest_year,
    end_year,
    k,
    species_name,
    trees_retained,
    replanting_data=None,
    override_planting=False,
    override_number=0,
    harvest_time=False,
    a=0,
):

    # k is combination of unique info
    zone, plot_id, is_replanting, year_start = k

    duration_project = config["duration_project"]
    dict_plot_start_year = config["dict_plot_start_year"]
    dict_plot_scenario = config["dict_plot_scenario"]

    # printing and testing purpose
    filtered_column = [
        "Plot_ID",
        "area_ha",
        "species",
        "year",
        "total_csu_tCO2e_species",
        "allowed_cut_csu",
        "tCO2e_harvest_allowed_species",
        "rotation_year",
        "harvest_time",
        "remnant_trees",
        "allowed_cut_proportion_okay",
        "remnant_level",
        "num_trees",
        "num_trees_harvested",
        "num_trees_retained",
        "tCO2e_retained",
        "cycle_harvest",
    ]

    if replanting_data is None:
        replanting_data = pd.DataFrame()

    # global variable check
    if (actual_year <= end_year) and (
        actual_year <= duration_project
    ):  # maximum replanting until year 30 only (if global variable duration_project is year 30)
        # all_df = all_df
        actual_year = actual_year
        cycle_harvest = math.ceil(((actual_year - a) - (year_start - 1)) / harvest_year)
        if gap_harvest:
            cycle_harvest = math.ceil(
                ((actual_year - a - 1) - (year_start - 1)) / harvest_year
            )
        rotation_year = harvest_cycle_rotation(
            actual_year - a - (year_start - 1), gap_harvest, harvest_year
        )  # adjusted the rotation status, following the recursive counter (a) and also with the delayed start planting field column
        # df_replant_cycle = filtered_df_prev_year.copy()
        # print('actual_year: ',actual_year)
        # print('rotation_year: ', rotation_year)
        # print('cycle_harvest: ',cycle_harvest)
        # print('year_queried: ', actual_year - a - (harvest_year+1) + (dict_plot_start_year[k]-1))
        # print('rotation_year_queried: ',rotation_year)

        if gap_harvest:  # gap harvest is enabled
            # harvesting cycle number
            if (
                rotation_year == 0
            ):  # if year is harvest_year (harvesting cycle) + one year, this will give 0 since we apply the gap, meaning here is the 1 year after harvesting, mod result is 0
                # print('HERE')

                df_add = all_df[
                    (all_df["Plot_ID"] == plot_id)
                    & (all_df["species"] == species_name)
                    & (all_df["year"] == harvest_year + (year_start - 1))
                    & (all_df["remnant_trees"] == False)
                    & (all_df["rotation_year"] == harvest_year)
                    & (all_df["is_replanting"] == is_replanting)
                    & (all_df["year_start"] == year_start)
                ]  # no need to exact x%j number, we will update this anyway
                df_add2 = df_add.copy()

                # print(1+(harvest_year)*(cycle_harvest-2))
                print(
                    f"applying query: all_df[(all_df['Plot_ID']=={k}) & (all_df['species']=={species_name}) & (all_df['year']=={harvest_year + (year_start-1)})  & \
                                (all_df['remnant_trees']=={False}) & (all_df['rotation_year']=={harvest_year}) (all_df['is_replanting']==is_replanting) & (all_df['year_start']==year_start) ] "
                )
                # display(df_add2[filtered_column])

                (
                    df_add2["year"],
                    df_add2["tco2_per_tree"],
                    df_add2["proportion_per_trees"],
                    df_add2["biomass_per_trees_proportion"],
                    df_add2["tCarbon_per_trees_proportion"],
                    df_add2["tCo2e_per_trees_proportion"],
                    df_add2["allowed_cut_proportion_okay"],
                    df_add2["total_csu_tCO2e_species"],
                    df_add2["remnant_trees"],
                    df_add2["num_trees"],
                    df_add2["harvest_time"],
                ) = zip(
                    *[
                        (
                            actual_year,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            False,
                            0.0,
                            False,
                            0,
                            False,
                        )
                    ]
                )

                df_add2["cycle_harvest"] = (
                    cycle_harvest  # x is current year, harvest_year is harvesting year
                )
                df_add2["rotation_year"] = rotation_year

                # display(df_add[filtered_column])

                # list_df.append(df_add2)
                # all_df = all_df.append(df_add2, ignore_index=True)

                # all_df = pd.concat([all_df,df_add2], ignore_index=True)
                replanting_data = pd.concat(
                    [replanting_data, df_add2], ignore_index=True
                )

                # return all_df

            elif rotation_year > 0 and rotation_year < harvest_year:
                # if x % j == 0: #if year is j (harvesting cycle)
                df_add = all_df[
                    (all_df["Plot_ID"] == plot_id)
                    & (all_df["species"] == species_name)
                    & (all_df["year"] == actual_year - a - (harvest_year + 1))
                    & (all_df["remnant_trees"] == False)
                    & (all_df["rotation_year"] == rotation_year)
                    & (all_df["is_replanting"] == is_replanting)
                    & (all_df["year_start"] == year_start)
                ]  # num 1 is the baseline,  since replanting started at cycle harvest num. 1, for the gap harvest only
                df_add2 = df_add.copy()

                # print(f"applying query: all_df[(all_df['Plot_ID']=={k}) & (all_df['species']=={species_name}) & (all_df['year']=={actual_year - a - (harvest_year+1)}) & \
                #                                           (all_df['remnant_trees']=={False}) & (all_df['rotation_year']=={rotation_year}) &(all_df['is_replanting']==is_replanting) & (all_df['year_start']==year_start) ]")

                # display(df_add2[filtered_column])

                df_add2["num_trees"] = (
                    df_add2["num_trees"] - trees_retained
                )  # avoid recursive, since the number will be the same after the first adjustment
                if override_planting:
                    df_add2["num_trees"] = override_number
                df_add2["year"] = actual_year
                df_add2["cycle_harvest"] = (
                    cycle_harvest  # z is current year, harvest_year is harvesting year
                )
                df_add2["total_csu_tCO2e_species"] = (
                    df_add2["num_trees"] * df_add2["tCo2e_per_trees_proportion"]
                )
                df_add2["remnant_trees"] = False

                df_add2["rotation_year"] = rotation_year
                df_add2["harvest_time"] = df_add2.apply(
                    lambda x: is_species_allowed_to_harvest(x, dict_plot_scenario),
                    axis=1,
                ).astype(bool)

                # all_df = pd.concat([all_df,df_add2], ignore_index=True)
                replanting_data = pd.concat(
                    [replanting_data, df_add2], ignore_index=True
                )

                # display(df_add[filtered_column])

                # return all_df

            elif rotation_year == harvest_year:  # harvesting year cycle
                # if x % j == 0: #if year is j (harvesting cycle)
                df_add = all_df[
                    (all_df["Plot_ID"] == plot_id)
                    & (all_df["species"] == species_name)
                    & (all_df["year"] == actual_year - a - (harvest_year + 1))
                    & (all_df["remnant_trees"] == False)
                    & (all_df["rotation_year"] == rotation_year)
                    & (all_df["is_replanting"] == is_replanting)
                    & (all_df["year_start"] == year_start)
                ]  # num 1 is the baseline, since replanting started at cycle harvest num. 1, for the gap harvest only
                df_add2 = df_add.copy()
                df_add2["num_trees"] = (
                    df_add2["num_trees"] - trees_retained
                )  # avoid recursive, since the number will be the same after the first adjustment
                if override_planting:
                    df_add2["num_trees"] = override_number
                df_add2["year"] = actual_year
                df_add2["cycle_harvest"] = (
                    cycle_harvest  # z is current year, harvest_year is harvesting year
                )
                df_add2["total_csu_tCO2e_species"] = (
                    df_add2["num_trees"] * df_add2["tCo2e_per_trees_proportion"]
                )
                df_add2["remnant_trees"] = False

                df_add2["rotation_year"] = rotation_year
                df_add2["harvest_time"] = df_add2.apply(
                    lambda x: is_species_allowed_to_harvest(x, dict_plot_scenario),
                    axis=1,
                ).astype(bool)

                # all_df = pd.concat([all_df,df_add2], ignore_index=True)
                replanting_data = pd.concat(
                    [replanting_data, df_add2], ignore_index=True
                )

        # if no gap_harvest, direct planting in the following year
        elif gap_harvest == False:
            # harvesting cycle number
            if rotation_year == 0:  # if year is j (harvesting cycle)
                df_add = all_df[
                    (all_df["Plot_ID"] == plot_id)
                    & (all_df["species"] == species_name)
                    & (
                        all_df["year"]
                        == harvest_year
                        + (harvest_year) * (cycle_harvest - 2)
                        + (year_start - 1)
                    )
                    & (all_df["remnant_trees"] == False)
                    & (all_df["rotation_year"] == harvest_year)
                    & (all_df["is_replanting"] == is_replanting)
                    & (all_df["year_start"] == year_start)
                ]

                # print(f"applying query: all_df[(all_df['Plot_ID']=={k}) & (all_df['species']=={species_name}) & (all_df['year']=={harvest_year+ (harvest_year)*(cycle_harvest-2) + (year_start-1)})& \
                #                                           (all_df['remnant_trees']==False) & (all_df['rotation_year']=={harvest_year})]')")
                #
                df_add2 = df_add.copy()
                # display(df_add2[filtered_column])
                df_add2["num_trees"] = df_add2["num_trees"] - trees_retained
                if override_planting:
                    df_add2["num_trees"] = override_number
                df_add2["year"] = actual_year
                df_add2["cycle_harvest"] = (
                    cycle_harvest  # z is current year, harvest_year is harvesting year
                )
                df_add2["total_csu_tCO2e_species"] = (
                    df_add2["num_trees"] * df_add2["tCo2e_per_trees_proportion"]
                )
                df_add2["remnant_trees"] = False

                df_add2["rotation_year"] = (
                    harvest_year  # THIS IS NEEDED, since the mod calculation without gap means this number
                )
                # df_add2['harvest_time'] = df_add2.apply(lambda x: fill_harvest_cycle(harvest_year, x['rotation_year']), axis = 1).astype(bool)
                df_add2["harvest_time"] = df_add2.apply(
                    lambda x: is_species_allowed_to_harvest(x, dict_plot_scenario),
                    axis=1,
                ).astype(bool)

                # all_df = pd.concat([all_df,df_add2], ignore_index=True)
                replanting_data = pd.concat(
                    [replanting_data, df_add2], ignore_index=True
                )

                # print('TESTING TESTING TESTING')
                # print(f'we are replanting in year {actual_year} of species: {species_name}, its harvest cycle is {harvest_year}')
                # print(f'HERE THE ACTUAL YEAR : {actual_year} for the planting and ROTATION YEAR IS {rotation_year} HARVESTING IN PREVIOUS ROTATION (10 Jabon, Sengon)')
                # print(f'OVERRIDE NUMBER TREES IS {override_number} ----------------------------------------------')
                # # print(f'ADJUSTED_YEAR_ACTUAL: {actual_year - (cycle_harvest-2)}')
                # print(f'WE ARE QUERYING OR COPYING THE ROW YEAR DATA FROM {harvest_year+ (harvest_year)*(cycle_harvest-2)-a}')
                # print('TESTING TESTING --------------------------------------------------------------------------')
                # display(df_add2[filtered_column])

                # list_df.append(df_add2)
                # return all_df

            elif rotation_year != 0:
                df_add = all_df[
                    (all_df["Plot_ID"] == plot_id)
                    & (all_df["species"] == species_name)
                    & (
                        all_df["year"]
                        == rotation_year
                        + (harvest_year) * (cycle_harvest - 2)
                        + (year_start - 1)
                    )
                    & (all_df["remnant_trees"] == False)
                    & (all_df["rotation_year"] == rotation_year)
                    & (all_df["is_replanting"] == is_replanting)
                    & (all_df["year_start"] == year_start)
                ]
                df_add2 = df_add.copy()
                # display(df_add2[filtered_column])
                df_add2["num_trees"] = df_add2["num_trees"] - trees_retained
                if override_planting:
                    df_add2["num_trees"] = override_number
                df_add2["year"] = actual_year
                df_add2["cycle_harvest"] = (
                    cycle_harvest  # z is current year, harvest_year is harvesting year
                )
                df_add2["total_csu_tCO2e_species"] = (
                    df_add2["num_trees"] * df_add2["tCo2e_per_trees_proportion"]
                )
                df_add2["remnant_trees"] = False
                df_add2["rotation_year"] = rotation_year

                # print('TESTING TESTING TESTING')
                # print(f'we are replanting in year {actual_year} of species: {species_name}, its harvest cycle is {harvest_year}')
                # print(f'HERE THE ACTUAL YEAR : {actual_year} for the planting and ROTATION YEAR IS {rotation_year}')
                # print(f'OVERRIDE NUMBER TREES IS {override_number} ----------------------------------------------')
                # # print(f'ADJUSTED_YEAR_ACTUAL: {actual_year - (cycle_harvest-2)}')
                # print(f'WE ARE QUERYING OR COPYING THE ROW YEAR DATA FROM {rotation_year+(harvest_year)*(cycle_harvest-2)-a}')
                # print('TESTING TESTING --------------------------------------------------------------------------')
                # display(df_add2[filtered_column])

                # df_add2['harvest_time'] = df_add2.apply(lambda x: fill_harvest_cycle(harvest_year, x['rotation_year']), axis = 1).astype(bool)
                df_add2["harvest_time"] = df_add2.apply(
                    lambda x: is_species_allowed_to_harvest(x, dict_plot_scenario),
                    axis=1,
                ).astype(bool)

                replanting_data = pd.concat(
                    [replanting_data, df_add2], ignore_index=True
                )

        # perform a recursive re_planting
        replanting_data = re_planting(
            config,
            all_df,
            gap_harvest,
            actual_year + 1,
            harvest_year,
            end_year,
            k,
            species_name,
            trees_retained,
            replanting_data=replanting_data,
            override_planting=override_planting,
            override_number=override_number,
            harvest_time=harvest_time,
            a=a,
        )

    return replanting_data
