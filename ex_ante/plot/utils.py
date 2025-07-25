import pandas as pd


def calc_plot_csu(gdrive_location_seedling, override_avg_tree_perha=''):
    # read the csv
    plot_sum = pd.read_csv(gdrive_location_seedling)
    # if we want to import back from pandas object (previously exported to csv), we need to make sure this column is not included
    unnamed_columns = [col for col in plot_sum.columns if "Unnamed" in col]
    plot_sum = plot_sum.drop(columns=unnamed_columns, errors="ignore")
    # return plot_sum

    # print('displaying the proportion, seedling distribution from percentage sharing option')
    # display(plot_sum)

    list_field_species_prop = [
        x for x in plot_sum.columns.to_list() if x.endswith("_num_trees")  # in x
    ]

    # ensure the blank is fillin with 0 for num_Trees
    # plot_sum[list_field_species_prop] =  plot_sum[list_field_species_prop].fillna(0) # this one takes ages

    # # let's try this for fillna # this one also takes ages
    # for col in plot_sum.columns:
    #     if "_num_trees" in col:
    #         plot_sum[col].fillna(0, inplace=True)
    # Select all columns that contain '_num_trees' and fillna(0) for them in one go

    # 1. Create a boolean mask for columns ending with "_num_trees"
    mask = plot_sum.columns.str.endswith("_num_trees")

    plot_sum.loc[:, mask] = plot_sum.loc[
        :, plot_sum.columns.str.contains("_num_trees")
    ].fillna(0)

    plot_sum["total_num_trees"] = plot_sum.loc[:, mask].sum(axis=1)

    plot_sum["avgtrees_per_ha"] = plot_sum["total_num_trees"] / plot_sum["area_ha"]
    if override_avg_tree_perha != '':
        plot_sum["avgtrees_per_ha"] = override_avg_tree_perha # this is needed for expost thinning_stop =True

    # melt_plot_species = pd.melt(plot_sum, id_vars=["Plot_ID",	"Plot_Name",'mu', 'zone',"area_ha", 'year_start' ],
    melt_plot_species = pd.melt(
        plot_sum,
        id_vars=[x for x in plot_sum.columns.to_list() if "_num_trees" not in x],
        value_vars=list_field_species_prop,
    ).rename(columns=dict(variable="species_suf", value="num_trees"))

    # let's try this for fillna # no need if above is executed (plot_sum)
    # melt_plot_species['num_trees'] = melt_plot_species['num_trees'].fillna(0) # at the end, there is code that need to be ensure that 0 is excluded in plot_sum instead

    def change_species(x):
        return x.replace("_num_trees", "")

    melt_plot_species["species"] = melt_plot_species["species_suf"].apply(
        lambda x: change_species(x)
    )

    return {"plot_sum": plot_sum, "melt_plot_species": melt_plot_species}
