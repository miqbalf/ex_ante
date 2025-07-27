import os

import pandas as pd


def num_tco_years(
    df_ex_ante: pd.DataFrame = pd.DataFrame(),
    is_save_to_csv: str = "",
    override_num_trees_0: bool = False,
    mortality_csu_df: pd.DataFrame = None,
    distribution_seedling: str = "",  # if we use the existing distribution_seedling path manually, otherwise we will use the same directory from is_save_to_csv
):
    # df_ex_ante = pd.read_csv(config['csv_file_exante'])
    if 'managementUnit' in df_ex_ante.columns:
        df_ex_ante = df_ex_ante.rename(columns={'Plot_ID': 'Plot_ID_exante', "zone": "plotZone"})
    else:
        df_ex_ante = df_ex_ante.rename(
            columns={
                "Plot_ID": "Plot_ID_exante",
                "mu": "managementUnit",
                "zone": "plotZone",
            }
        )

    # Creating a pivot table, ex_ante adjustment for num_trees and tco2e
    pivot_df_num_trees_tco2e = pd.pivot_table(
        df_ex_ante,
        values=["total_csu_tCO2e_species", "num_trees_adjusted"],
        index=["Plot_ID_exante", "species", "is_replanting", "year_start"],
        columns="year",
        aggfunc="sum",
    )

    if distribution_seedling == "" and is_save_to_csv != "":
        output_main_dir = os.path.dirname(os.path.dirname(is_save_to_csv))
        distribution_seedling = os.path.join(
            output_main_dir,
            f"{os.path.basename(is_save_to_csv)}_distribution_trees_seedling.csv",
        )

    distribution_seedling_df = pd.read_csv(distribution_seedling)

    # if there is no replanting (initial model or expost without replanting)
    distribution_seedling_df["is_replanting"] = distribution_seedling_df.apply(
        lambda x: (
            x["is_replanting"]
            if "is_replanting" in distribution_seedling_df.columns
            else False
        ),
        axis=1,
    )

    distribution_seedling_df = distribution_seedling_df.rename(
        columns={"mu": "managementUnit", "zone": "plotZone"}
    )
    distribution_seedling_df = distribution_seedling_df[
        ["Plot_ID", "is_replanting", "year_start", "managementUnit", "plotZone"]
    ].rename(columns={"Plot_ID": "Plot_ID_exante"})

    # Group data by 'year_start' and aggregate into a dictionary  # grouping based on year_start and is_replanting as well later because there are some project that has different year_start
    year_dict = {}

    for _, row in distribution_seedling_df.iterrows():
        year_start = row["year_start"]
        entry = (row["Plot_ID_exante"], row["is_replanting"])

        # Append to list under the same year_start key
        if year_start in year_dict:
            year_dict[year_start].append(entry)
        else:
            year_dict[year_start] = [entry]

    # filling the data frame with the grouped data of is_replanting and Plot_ID
    frame_d = pd.DataFrame(None)
    for year_start, list_plot_is_replanting in year_dict.items():
        for plotid, is_replanting in list_plot_is_replanting:
            filtered = df_ex_ante[
                (df_ex_ante["year"] == year_start)
                & (df_ex_ante["Plot_ID_exante"] == plotid)
                & (df_ex_ante["is_replanting"] == is_replanting)
            ]
            frame_d = pd.concat([frame_d, filtered])

    # the filter will be dynamically generated based on the trees distribution data
    filtered_num_year = frame_d

    ## now we put automated year_0 based on num_trees data above and year_start based on plot distribution seedling
    pivot_num_trees_0 = pd.pivot_table(
        filtered_num_year,
        values=["num_trees"],
        index=[
            "year_start",
            "is_replanting",
            "plotZone",
            "managementUnit",
            "Plot_ID_exante",
            "species",
        ],
        columns="year",
        aggfunc="sum",  # margins=True
    )
    # starting the filter
    for k, v in year_dict.items():
        pivot_num_trees_0[("num_trees_adjusted", k - 1)] = pivot_num_trees_0[
            ("num_trees", k)
        ]
        pivot_num_trees_0 = pivot_num_trees_0.drop(columns=[("num_trees", k)])

    # join all the important aggregation columns num_trees and tco2e over years, by plot and species
    joined_pivot_num_trees_tco2e_all = pd.merge(
        pivot_df_num_trees_tco2e,
        pivot_num_trees_0,
        left_index=True,
        right_index=True,
        how="outer",
    )

    # Now re-engineer just to choose of the num_trees without NaN
    # Step 1: check the columns dynamically
    x_columns = [
        col
        for col in joined_pivot_num_trees_tco2e_all.columns
        if col[0] == "num_trees_adjusted_x"
    ]
    y_columns = [
        col
        for col in joined_pivot_num_trees_tco2e_all.columns
        if col[0] == "num_trees_adjusted_y"
    ]

    # Step 2: Extract unique 'year' levels dynamically
    years = sorted(set(col[1] for col in x_columns + y_columns))

    # Step 3: Combine values from _x and _y for each 'year' level
    for year in years:
        x_col = ("num_trees_adjusted_x", year)
        y_col = ("num_trees_adjusted_y", year)
        target_col = ("num_trees_adjusted", year)

        # Combine values without hardcoding years
        if (
            x_col in joined_pivot_num_trees_tco2e_all.columns
            and y_col in joined_pivot_num_trees_tco2e_all.columns
        ):
            joined_pivot_num_trees_tco2e_all[target_col] = (
                joined_pivot_num_trees_tco2e_all[x_col].combine_first(
                    joined_pivot_num_trees_tco2e_all[y_col]
                )
            )
        elif x_col in joined_pivot_num_trees_tco2e_all.columns:
            joined_pivot_num_trees_tco2e_all[target_col] = (
                joined_pivot_num_trees_tco2e_all[x_col]
            )
        elif y_col in joined_pivot_num_trees_tco2e_all.columns:
            joined_pivot_num_trees_tco2e_all[target_col] = (
                joined_pivot_num_trees_tco2e_all[y_col]
            )

    # Step 4: Drop the original columns with suffixes
    joined_pivot_num_trees_tco2e_all = joined_pivot_num_trees_tco2e_all.drop(
        columns=[
            col
            for col in joined_pivot_num_trees_tco2e_all.columns
            if col[0].endswith("_x") or col[0].endswith("_y")
        ]
    )

    # Step 5: Sort the columns by 'year' to ensure correct order if needed
    joined_pivot_num_trees_tco2e_all = joined_pivot_num_trees_tco2e_all.sort_index(
        axis=1, level=1
    )

    # will equal to this, but this one is manual creation of year_0,
    # pivot_num_trees_0 = pd.pivot_table(df_ex_ante[df_ex_ante['year']==1], values=['num_trees'], index=['Plot_ID','species'], columns='year', aggfunc='sum')
    # pivot_num_trees_0[('num_trees_adjusted',0)] = pivot_num_trees_0[('num_trees',1)]
    # pivot_num_trees_0 = pivot_num_trees_0.drop(columns = [('num_trees',1)])
    ## join all the important aggregation columns num_trees and tco2e over years, by plot and species
    # joined_pivot_num_trees_tco2e_all = pd.merge(pivot_df_num_trees_tco2e,pivot_num_trees_0, left_index=True, right_index=True)

    ## Extract 'species' from the MultiIndex and convert it into a Series
    species_series = pd.Series(
        joined_pivot_num_trees_tco2e_all.index.get_level_values("species"),
        index=joined_pivot_num_trees_tco2e_all.index,
    )
    # joined_pivot_num_trees_tco2e_all['species_treeocloud'] = species_series.apply(lambda species_name: species_match_coredb_treeocloud(species_name, species_json))
    # temp only:
    joined_pivot_num_trees_tco2e_all["species_series"] = species_series

    joined_pivot_num_trees_tco2e_all = joined_pivot_num_trees_tco2e_all.reset_index()
    # joined_pivot_num_trees_tco2e_all = joined_pivot_num_trees_tco2e_all.set_index(['Plot_ID_exante','species_treeocloud'])
    # temp
    joined_pivot_num_trees_tco2e_all = joined_pivot_num_trees_tco2e_all.set_index(
        [
            "year_start",
            "is_replanting",
            "plotZone",
            "managementUnit",
            "Plot_ID_exante",
            "species",
        ]
    )

    # Sort the DataFrame based on the MultiIndex
    joined_pivot_num_trees_tco2e_all = joined_pivot_num_trees_tco2e_all.sort_index(
        level=[
            "year_start",
            "is_replanting",
            "plotZone",
            "managementUnit",
            "Plot_ID_exante",
            "species",
        ]
    )

    # re-order position, only for visualization
    columns = list(joined_pivot_num_trees_tco2e_all.columns)
    print("columns: ", columns)
    columns.remove(("num_trees_adjusted", 0))
    columns = [("num_trees_adjusted", 0)] + columns
    joined_pivot_num_trees_tco2e_all = joined_pivot_num_trees_tco2e_all[columns]

    # flatten level for num_trees
    joined_pivot_num_trees_all = joined_pivot_num_trees_tco2e_all["num_trees_adjusted"]

    # implement override_num_trees_0 to present the number explicitly in the result num_trees_over_the_years
    if override_num_trees_0 == True:
        # this is hot fix for overriding from csu species mort. many not use mu (managementUnit) combination yet
        joined_pivot_num_trees_all = joined_pivot_num_trees_all.reset_index()
        joined_pivot_num_trees_all = joined_pivot_num_trees_all.set_index(
            ["is_replanting", "year_start", "Plot_ID_exante", "species"]
        )

        try:

            # change the index structure from mortality rate (adding is_replanting as False) to match the index with this module
            # Step 1: Extract the current levels of the MultiIndex
            current_tuples = (
                mortality_csu_df.index.tolist()
            )  # Convert MultiIndex to list of tuples

            # Step 2: Add the new level (`False`) to the beginning of each tuple
            # new_tuples = [(False,) + tup for tup in current_tuples]
            # let's put at the first time when is_replanting put in the csu_mort_df too, in the previous mortality analysis:
            new_tuples = current_tuples

            # Step 3: Construct a new MultiIndex with the updated levels
            new_index = pd.MultiIndex.from_tuples(
                new_tuples,
                names=["is_replanting", "year_start", "Plot_ID_expost", "species"],
            )
        except Exception as e:
            raise ValueError(
                "Please add the following columns in expost data: \n "
                "['is_replanting','year_start' ]",
                f"error {e}",
            )

        # Print the new MultiIndex
        # print(new_index)

        # Assign the new MultiIndex to the DataFrame
        mortality_csu_df.index = new_index

        # we will use directly the series since it will be expected to have the same multi-index structure
        joined_pivot_num_trees_all = joined_pivot_num_trees_all.copy()
        # the implementation to override the num_trees year 0 with the monitoring planted (expected) year 0 before mortality
        joined_pivot_num_trees_all[0] = mortality_csu_df["num_trees_All"]

        # Calculate the grand total for each numeric column
        grand_total_num_trees = joined_pivot_num_trees_all.sum(numeric_only=True)

        # Create a new row with the grand total values and the appropriate index
        grand_total_row = pd.DataFrame(
            [grand_total_num_trees],
            index=pd.MultiIndex.from_tuples(
                [("Grand Total", "", "", "")],
                names=joined_pivot_num_trees_all.index.names,
            ),
        )
    else:
        # Calculate the grand total for each numeric column
        grand_total_num_trees = joined_pivot_num_trees_all.sum()

        # Create a new row with the grand total values and the appropriate index
        grand_total_row = pd.DataFrame(
            [grand_total_num_trees],
            index=pd.MultiIndex.from_tuples(
                [("Grand Total", "", "", "", "", "")],
                names=joined_pivot_num_trees_all.index.names,
            ),
        )

    # Append the grand total row to the DataFrame
    exante_num_trees_yrs = pd.concat([joined_pivot_num_trees_all, grand_total_row])
    exante_num_trees_yrs = exante_num_trees_yrs

    # for all the trees just planted (year_start 0) it will be considered as 0
    joined_pivot_num_trees_tco2e_all[("total_csu_tCO2e_species", 0)] = 0
    # flatten level for the tco2e
    joined_pivot_tco2e_all = joined_pivot_num_trees_tco2e_all["total_csu_tCO2e_species"]

    # re-order position, only for visualization
    columns_tco2 = list(joined_pivot_tco2e_all.columns)
    columns_tco2.remove(0)
    columns_tco2 = [0] + columns_tco2

    joined_pivot_tco2e_all = joined_pivot_tco2e_all[columns_tco2]

    ## Calculate the grand total for each numeric column
    grand_total_tco2 = joined_pivot_tco2e_all.sum()

    # Create a new row with the grand total values and the appropriate index
    grand_total_row = pd.DataFrame(
        [grand_total_tco2],
        index=pd.MultiIndex.from_tuples(
            [("Grand Total", "", "", "", "", "")],
            names=joined_pivot_tco2e_all.index.names,
        ),
    )

    # Append the grand total row to the DataFrame
    exante_tco2e_yrs = pd.concat([joined_pivot_tco2e_all, grand_total_row])

    if is_save_to_csv != "":
        exante_num_trees_yrs.to_csv(f"{is_save_to_csv}_num_trees_years.csv")
        exante_tco2e_yrs.to_csv(f"{is_save_to_csv}_tco2_years.csv")

    return {
        "exante_num_trees_yrs": exante_num_trees_yrs,
        "exante_tco2e_yrs": exante_tco2e_yrs,
        "joined_pivot_tco2e_all": joined_pivot_tco2e_all,
        "joined_pivot_num_trees_all": joined_pivot_num_trees_all,
    }
