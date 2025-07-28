import os

import pandas as pd

def pop_num_trees(df, seedling_csu, base_year):
    distribution_seedling_df = seedling_csu

    pivot_df_num_trees = pd.pivot_table(
        df,
        values=["num_trees_adjusted"],
        index=["Plot_ID_exante", "species", "is_replanting", "year_start"],
        columns="year",
        aggfunc="sum",
        )

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

    # print(year_dict) #debug

    # set the filter based on the dictionary list
    frame_d = pd.DataFrame(None)
    for year_start, list_plot_is_replanting in year_dict.items():
        for plotid, is_replanting in list_plot_is_replanting:
            filtered = df[
                (df["year"] == year_start + base_year)
                & (df["Plot_ID_exante"] == plotid)
                & (df["is_replanting"] == is_replanting)
            ]
            frame_d = pd.concat([frame_d, filtered])

    frame_d['num_trees'] = frame_d.apply(lambda x: x['num_trees'] if x['rotation_year']==1 else 0, axis=1)

    filtered_num_year = frame_d

    # print('filtered_num_year')
    # display(filtered_num_year)

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

    for k, v in year_dict.items():
        pivot_num_trees_0[("num_trees_adjusted", base_year + k -1)] = pivot_num_trees_0[
            ("num_trees", base_year + k)
        ]

        # display(pivot_num_trees_0)
        pivot_num_trees_0 = pivot_num_trees_0.drop(columns=[("num_trees", base_year + k )])

    # join all the important aggregation columns num_trees over years, by plot and species
    joined_pivot_num_trees_all = pd.merge(
        pivot_df_num_trees,
        pivot_num_trees_0,
        left_index=True,
        right_index=True,
        how="outer",
    )

    # Now re-engineer just to choose of the num_trees without NaN
    # Step 1: check the columns dynamically
    x_columns = [
        col
        for col in joined_pivot_num_trees_all.columns
        if col[0] == "num_trees_adjusted_x"
    ]
    y_columns = [
        col
        for col in joined_pivot_num_trees_all.columns
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
            x_col in joined_pivot_num_trees_all.columns
            and y_col in joined_pivot_num_trees_all.columns
        ):
            joined_pivot_num_trees_all[target_col] = (
                joined_pivot_num_trees_all[x_col].combine_first(
                    joined_pivot_num_trees_all[y_col]
                )
            )
        elif x_col in joined_pivot_num_trees_all.columns:
            joined_pivot_num_trees_all[target_col] = (
                joined_pivot_num_trees_all[x_col]
            )
        elif y_col in joined_pivot_num_trees_all.columns:
            joined_pivot_num_trees_all[target_col] = (
                joined_pivot_num_trees_all[y_col]
            )

    # Step 4: Drop the original columns with suffixes
    joined_pivot_num_trees_all = joined_pivot_num_trees_all.drop(
        columns=[
            col
            for col in joined_pivot_num_trees_all.columns
            if col[0].endswith("_x") or col[0].endswith("_y")
        ]
    )

    # Step 5: Sort the columns by 'year' to ensure correct order if needed
    joined_pivot_num_trees_all = joined_pivot_num_trees_all.sort_index(
        axis=1, level=1
    )

    # debug
    # display(joined_pivot_num_trees_all) # this is fine

    # will equal to this, but this one is manual creation of year_0,
    # pivot_num_trees_0 = pd.pivot_table(df_ex_ante[df_ex_ante['year']==1], values=['num_trees'], index=['Plot_ID','species'], columns='year', aggfunc='sum')
    # pivot_num_trees_0[('num_trees_adjusted',0)] = pivot_num_trees_0[('num_trees',1)]
    # pivot_num_trees_0 = pivot_num_trees_0.drop(columns = [('num_trees',1)])
    ## join all the important aggregation columns num_trees and tco2e over years, by plot and species
    # joined_pivot_num_trees_tco2e_all = pd.merge(pivot_df_num_trees_tco2e,pivot_num_trees_0, left_index=True, right_index=True)

    ## Extract 'species' from the MultiIndex and convert it into a Series
    species_series = pd.Series(
        joined_pivot_num_trees_all.index.get_level_values("species"),
        index=joined_pivot_num_trees_all.index,
    )
    # joined_pivot_num_trees_tco2e_all['species_treeocloud'] = species_series.apply(lambda species_name: species_match_coredb_treeocloud(species_name, species_json))
    # temp only:
    joined_pivot_num_trees_all["species_series"] = species_series

    joined_pivot_num_trees_all = joined_pivot_num_trees_all.reset_index()
    # joined_pivot_num_trees_tco2e_all = joined_pivot_num_trees_tco2e_all.set_index(['Plot_ID_exante','species_treeocloud'])
    # temp
    joined_pivot_num_trees_all = joined_pivot_num_trees_all.set_index(
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
    joined_pivot_num_trees_all = joined_pivot_num_trees_all.sort_index(
        level=[
            "year_start",
            "is_replanting",
            "plotZone",
            "managementUnit",
            "Plot_ID_exante",
            "species",
        ]
    )

    # display(joined_pivot_num_trees_all) #debug

    return joined_pivot_num_trees_all

def pop_tco2(df_ex_ante, base_year=0, planting_year=0):
    # Creating a pivot table, ex_ante adjustment for num_trees and tco2e
    pivot_df_tco2e = pd.pivot_table(
        df_ex_ante,
        values=["total_csu_tCO2e_species"],
        index=["year_start",
            "is_replanting",
            "plotZone",
            "managementUnit",
            "Plot_ID_exante",
            "species"],
        columns="year",
        aggfunc="sum",
    )

    # for all the trees just planted (year_start 0) it will be considered as 0
    # pivot_df_tco2e[("total_csu_tCO2e_species", 0)] = 0
    # flatten level for the tco2e
    joined_pivot_tco2e_all = pivot_df_tco2e["total_csu_tCO2e_species"]

    # re-order position, only for visualization
    # columns_tco2 = list(joined_pivot_tco2e_all.columns)
    # columns_tco2.remove(0)
    # columns_tco2 = [0] + columns_tco2

    # joined_pivot_tco2e_all = joined_pivot_tco2e_all[columns_tco2]

    ## Calculate the grand total for each numeric column
    grand_total_tco2 = joined_pivot_tco2e_all.sum()

    # Create a new row with the grand total values and the appropriate index
    grand_total_row = pd.DataFrame(
        [grand_total_tco2],
        index=pd.MultiIndex.from_tuples(
            [("Grand Total", "", "", "",'','')],
            names=joined_pivot_tco2e_all.index.names,
        ),
    )

    # Append the grand total row to the DataFrame
    exante_tco2e_yrs = pd.concat([joined_pivot_tco2e_all, grand_total_row])

    return {'exante_tco2e_yrs':exante_tco2e_yrs, 'joined_pivot_tco2e_all': joined_pivot_tco2e_all}



def num_tco_years(
    df_ex_ante: pd.DataFrame = pd.DataFrame(),
    is_save_to_csv: str = "",
    override_num_trees_0: bool = False,
    # mortality_csu_df: pd.DataFrame = None,
    distribution_seedling: str = "",  # if we use the existing distribution_seedling path manually, otherwise we will use the same directory from is_save_to_csv
    large_tree: bool = False,
    planting_year : int = 0,
    current_gap_year: int = 0,
    num_trees_prev_exante: pd.DataFrame = pd.DataFrame(None),
    pivot_csu: pd.DataFrame = pd.DataFrame(None)
):
    ''' example parameter arg.
    self = expost
    mortality_csu_df=self.mortality_csu_df
    pivot_csu = self.pivot_plot_species

    ## parameter example
    # adjust the large tree
    large_tree = True
    planting_year = 2024 # set this later
    current_gap_year = 1
    root_folder_prev_exante = expost.root_folder_prev_exante
    override_num_trees_0 = True
    num_trees_dir = os.path.join(root_folder_prev_exante, 'num_trees_tco2_over_the_year')
    num_trees_prev_exante = os.path.join(num_trees_dir, expost.project_name_exante+'_num_trees_years.csv')
    num_trees_prev_exante = pd.read_csv(num_trees_prev_exante)
    # mortality_num_trees =
    distribution_seedling_df = expost.updated_exante.csu_seedling
    df_exante = all_df_merged
    
    '''

    df_ex_ante = df_ex_ante.copy()
    base_year = planting_year + current_gap_year

    if distribution_seedling == '' and is_save_to_csv != '':
        output_main_dir = os.path.dirname(os.path.dirname(is_save_to_csv))
        distribution_seedling = os.path.join(output_main_dir, f'{os.path.basename(is_save_to_csv)}_distribution_trees_seedling.csv')

    distribution_seedling_df = pd.read_csv(distribution_seedling)

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
    
    # display(distribution_seedling_df) #debug

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


    df_ex_ante['year'] = df_ex_ante['year'] + base_year

    if large_tree == True: # we will set the tree evidence for num_trees purpose only to get the number of trees in the previous years (year -1) because the algorithm process previously apply year_start delay for tree evidence due to no carbon yet
        df_ex_ante_for_num_trees = df_ex_ante.copy()
        df_ex_ante_for_num_trees['year'] = df_ex_ante_for_num_trees.apply(lambda x: x['year'] -1 if x['measurement_type'] == 'Nr Tree Evidence Expost' and x['is_replanting'] == False else x['year'], axis=1)

    else:
        df_ex_ante_for_num_trees = df_ex_ante.copy()

    joined_pivot_num_trees_all = pop_num_trees(df_ex_ante_for_num_trees, distribution_seedling_df, base_year)
    joined_pivot_num_trees_all.columns = joined_pivot_num_trees_all.columns.droplevel(0) # remove unecessary num_trees_adjusted column (level 0)
    # display(joined_pivot_num_trees_all)
    # update mortality analysis

    tco2 = pop_tco2(df_ex_ante=df_ex_ante)
    joined_pivot_tco2e_all=tco2['joined_pivot_tco2e_all']
    exante_tco2e_yrs = tco2['exante_tco2e_yrs']

    if override_num_trees_0 == True:
        print('selecting override num_trees_0')
        # this is hot fix for overriding from csu species mort. may not use mu (managementUnit) combination yet
        # joined_pivot_num_trees_all = joined_pivot_num_trees_all.reset_index() # Remove this line
        unique_index = ["is_replanting", "year_start", "Plot_ID_exante", "species", "managementUnit", "plotZone"]

        joined_pivot_num_trees_all = joined_pivot_num_trees_all.reset_index()
        joined_pivot_num_trees_all = joined_pivot_num_trees_all.set_index(unique_index)


        # now we will assume that this is next year planting
        if planting_year != base_year:
            num_trees_prev_exante = num_trees_prev_exante.copy() # Remove reset_index()
            num_trees_prev_exante =num_trees_prev_exante.iloc[:-1]
            num_trees_prev_exante = num_trees_prev_exante.reset_index()
            num_trees_prev_exante['year_start'] = num_trees_prev_exante['year_start'].astype(int)
            num_trees_prev_exante = num_trees_prev_exante[unique_index+ ['0']]
            num_trees_prev_exante= num_trees_prev_exante.rename(columns={'0':planting_year})

            # joined_pivot_num_trees_all[planting_year] = num_trees_prev_exante['0']
            # num_trees_prev_exante = num_trees_prev_exante[unique_index+['0']]
            # num_trees_prev_exante[planting_year] = num_trees_prev_exante['0']
            # num_trees_prev_exante = num_trees_prev_exante.set_index(unique_index)
            # joined_pivot_num_trees_all[planting_year] = num_trees_prev_exante['0']

            joined_pivot_num_trees_all = joined_pivot_num_trees_all.reset_index()

            joined_pivot_num_trees_all = pd.merge(joined_pivot_num_trees_all, num_trees_prev_exante, on=unique_index, how='outer')

            # melt_df = pd.melt(pivot_csu,id_vars=["Plot_ID", "is_replanting", "year_start", "managementUnit", "plotZone"], value_vars=[col for col in pivot_csu.columns if col.endswith('_num_trees')])
            # melt_df = melt_df.rename(columns={'Plot_ID':'Plot_ID_exante'})
            # melt_df['species'] = melt_df.apply(lambda x: x['species'].replace('_num_trees',''),axis=1)
            # melt_df['value'] = melt_df['value'].astype(float)
            # melt_df = melt_df.rename(columns={'value': base_year})

            melt_df = pd.melt(
                pivot_csu,
                id_vars=["Plot_ID", "is_replanting", "year_start", "managementUnit", "plotZone"],
                value_vars=[col for col in pivot_csu.columns if col.endswith('_num_trees')],
                var_name='species' # Name the new column 'species' directly
            )

            # Rename 'Plot_ID' using the correct dictionary syntax
            melt_df = melt_df.rename(columns={'Plot_ID': 'Plot_ID_exante'})

            # Clean the species names and rename the 'value' column
            melt_df['species'] = melt_df['species'].str.replace('_num_trees', '')
            melt_df['value'] = melt_df['value'].astype(float)
            melt_df = melt_df.rename(columns={'value': base_year})

            joined_pivot_num_trees_all = joined_pivot_num_trees_all.drop(columns=[base_year])

            joined_pivot_num_trees_all = pd.merge(joined_pivot_num_trees_all, melt_df, on=unique_index, how='outer')
            joined_pivot_num_trees_all = joined_pivot_num_trees_all.drop(columns=['']) # not sure why we have '' column
            joined_pivot_num_trees_all = joined_pivot_num_trees_all.set_index(unique_index)

        # Calculate the grand total for each numeric column
        grand_total_num_trees = joined_pivot_num_trees_all.sum(numeric_only=True)

        # Create a new row with the grand total values and the appropriate index
        grand_total_row = pd.DataFrame(
            [grand_total_num_trees],
            index=pd.MultiIndex.from_tuples(
                [("Grand Total", "", "", "",'','')],
                names=joined_pivot_num_trees_all.index.names,
            ),
        )
    else:
        # Calculate the grand total for each numeric column
        grand_total_num_trees = joined_pivot_num_trees_all.sum(numeric_only=True)

        # Create a new row with the grand total values and the appropriate index
        grand_total_row = pd.DataFrame(
            [grand_total_num_trees],
            index=pd.MultiIndex.from_tuples(
                [("Grand Total", "", "", "",'','')],
                names=joined_pivot_num_trees_all.index.names,
            ),
        )

        
    # display(grand_total_row)
    # display(joined_pivot_num_trees_all)

    # Append the grand total row to the DataFrame
    exante_num_trees_yrs = pd.concat([joined_pivot_num_trees_all, grand_total_row])
    columns_sort = exante_num_trees_yrs.columns.tolist()
    columns_sort = [col for col in columns_sort if isinstance(col, str)] + sorted([col for col in columns_sort if isinstance(col, int)])
    exante_num_trees_yrs = exante_num_trees_yrs[columns_sort]
    exante_num_trees_yrs = exante_num_trees_yrs

    if is_save_to_csv != "":
        exante_num_trees_yrs.to_csv(f"{is_save_to_csv}_num_trees_years.csv")
        exante_tco2e_yrs.to_csv(f"{is_save_to_csv}_tco2_years.csv")

    # debugging
    display(exante_num_trees_yrs)
    display(exante_tco2e_yrs)

    return {
        "exante_num_trees_yrs": exante_num_trees_yrs,
        "exante_tco2e_yrs": exante_tco2e_yrs,
        "joined_pivot_tco2e_all": joined_pivot_tco2e_all,
        "joined_pivot_num_trees_all": joined_pivot_num_trees_all,
    }
