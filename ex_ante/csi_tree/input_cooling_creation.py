import matplotlib.pyplot as plt
import pandas as pd

from ..utils.helper import adding_zero_meas


def input_cooling(
    all_df_merged, growth_melt, name_species_growth, base_year, conversion_tco2=44 / 12
):
    # input adjustment of GCS calculation, adding all columns from growth data
    all_df_input = all_df_merged.copy()

    all_df_input = pd.merge(
        all_df_input,
        growth_melt,
        left_on=["species", "rotation_year"],
        right_on=[name_species_growth, "year"],
        how="left",
        suffixes=("_input", "_growthdata"),
    )

    all_df_input = all_df_input.drop(columns="year_growthdata")
    all_df_input = all_df_input.rename(columns={"year_input": "year"})
    all_df_input["year"] = all_df_input["year"].astype(int)

    # year_start_plant
    all_df_input["planting_year"] = (
        all_df_input["year_start"] + base_year - 1
    )  # because year is start at 1
    all_df_input["measurement_year"] = (
        all_df_input["planting_year"]
        + all_df_input["year"]
        - (all_df_input["year_start"] - 1)
    )  # since year_start is start from 1 not from 0

    all_df_input["plot_area_ha"] = all_df_input["area_ha"]
    all_df_input["plot_id"] = all_df_input["Plot_ID"]
    all_df_input["co2_tree_captured_tonnes"] = all_df_input["total_csu_tCO2e_species"]
    all_df_input["tree_dbh_mm"] = all_df_input["DBH"] * 10
    all_df_input["tree_total_biomass_tonnes"] = (
        all_df_input["total_csu_tCO2e_species"] / (conversion_tco2) / 0.47
    )  # convert to biomass from tCO2e
    all_df_input["measurement_id"] = all_df_input.index

    all_df_input["num_trees_init"] = all_df_input["num_trees"] 

    # planting_year	measurement_year	plot_area_ha	plot_id	species	co2_tree_captured_tonnes	measurement_id	tree_dbh_mm	tree_total_biomass_tonnes
    # max year per plot
    max_measurement_years = (
        all_df_input.groupby("plot_id")["measurement_year"].max().reset_index()
    )
    min_max_per_plot = max_measurement_years[
        "measurement_year"
    ].min()  # we need this to limit the measurement and to make the c-sink input works

    min_measurement_years = (
        all_df_input.groupby(["plot_id", "plot_area_ha"])["measurement_year"]
        .min()
        .reset_index()
    )
    min_measurement_all = all_df_input["measurement_year"].min()

    list_measurement_list = all_df_input["measurement_year"].to_list()

    all_df_input = all_df_input.copy()
    all_df_input = all_df_input[all_df_input["measurement_year"] <= min_max_per_plot]

    # min_measurement_years
    list_df_zero = []
    for i in range(len(min_measurement_years["plot_id"])):
        plot_id = min_measurement_years["plot_id"][i]
        plot_area = min_measurement_years["plot_area_ha"][i]
        min_measurement = min_measurement_years["measurement_year"][i]
        # print(plot_id)

        range_iter = (
            min_measurement - min_measurement_all
        )  # example if 2025 - 2023, means we will iterate with 2023 and 2024 and add them later
        for j in range(range_iter):
            print(plot_id, min_measurement_all + j)
            df_zero = adding_zero_meas(plot_id, plot_area, min_measurement_all + j)
            list_df_zero.append(df_zero)

    all_df_input = pd.concat(list_df_zero + [all_df_input], ignore_index=True)
    input_gcs = all_df_input[
        [
            "planting_year",
            "measurement_year",
            "is_replanting",
            "year_start",
            "plot_area_ha",
            "num_trees_init",
            "plot_id",
            "species",
            "co2_tree_captured_tonnes",
            "measurement_id",
            "tree_dbh_mm",
            "tree_total_biomass_tonnes",
        ]
    ]

    return input_gcs


def plot_co2_species(input_gcs: pd.DataFrame, location_save=""):
    # Create the pivot table
    pivot_df = input_gcs.pivot_table(
        index="measurement_year",
        columns="species",
        values="co2_tree_captured_tonnes",
        aggfunc="sum",
    ).fillna(0)

    # Summarize data based on year
    yearly_summary = (
        input_gcs.groupby("measurement_year")["co2_tree_captured_tonnes"]
        .sum()
        .reset_index()
    )

    # Calculate the long-term average of the yearly summed CO2 captured
    long_term_average = yearly_summary["co2_tree_captured_tonnes"].mean()
    print("LTA capture :", long_term_average)

    # Plot the stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))  # Create a new figure and axis

    pivot_df.plot(kind="bar", stacked=True, ax=ax)  # Use the created axis

    # Plot the long-term average line
    ax.axhline(
        y=long_term_average,
        color="orange",
        linestyle="--",
        linewidth=2,
        label="Long-term Average captured CO2",
    )

    # Add label over the long-term average line
    x_position = ax.get_xticks()[0]  # Get the x-coordinate for the first bar
    ax.text(
        x_position,
        long_term_average - 350,
        f"{long_term_average:.2f}",
        color="orange",
        fontsize=10,
        fontweight="bold",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("CO2 Captured (Tonnes)")
    ax.set_title("CO2 Captured by Trees Over 30 Years")
    ax.legend(title="Species")

    # Adjusting legend properties
    ax.legend(loc="upper left", fontsize="small", markerscale=0.5)

    if location_save != "":
        plt.savefig(
            location_save, dpi=300, bbox_inches="tight"
        )  # Save before plt.show()

    plt.show()
