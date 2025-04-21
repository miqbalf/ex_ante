import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import fsolve


# Logistic growth function definition
def logistic_growth(t, K, r, t0):
    """
    Parameters:
    - t: Time or age (e.g., in years)
    - r: Growth rate (controls how quickly DBH grows)
    - t0: Inflection point (year when growth rate is fastest)
    - K: Carrying capacity or max DBH the tree can achieve (asymptote)
    """
    return K / (1 + np.exp(-r * (t - t0)))


# Estimate growth rate 'r' so that the model starts close to 'initial_dbh' at year 1
def estimate_growth_rate(initial_dbh, max_dbh, inflection_point_guess):
    def dbh_at_one_year(r):
        # Set target DBH at year 1 to initial_dbh for estimation
        return logistic_growth(1, max_dbh, r, inflection_point_guess) - initial_dbh

    # Use fsolve to estimate r, starting with an initial guess
    r_initial_guess = 0.1
    r = fsolve(dbh_at_one_year, r_initial_guess)[0]
    return r


def remodel_growth(
    *args: str,
    initial_dbh: float = 1.0,
    max_dbh: float = 40,
    inflection_point_guess: float = 10,
    projected_years: int = 35,
    species: str = None,
    species_df: pd.DataFrame = None,
) -> pd.DataFrame:
    # Calculate the growth rate 'r'
    r_estimated = estimate_growth_rate(initial_dbh, max_dbh, inflection_point_guess)
    print(
        f"Species: {species} | Estimated growth rate r: {r_estimated:.4f} | Inflection point year: {inflection_point_guess}"
    )

    # Extend years for projection
    extended_years = np.arange(1, projected_years + 1)
    dbh_projection = logistic_growth(
        extended_years, max_dbh, r_estimated, inflection_point_guess
    )

    # Create a dictionary for growth projections
    dict_growth = {"year": extended_years, "dbh_projection": dbh_projection}

    if species_df is not None:
        for arg in args:
            if arg in species_df.columns:
                dict_growth[f"{arg}_extended"] = np.concatenate(
                    [
                        species_df[arg].values,
                        [species_df[arg].iloc[-1]]
                        * (len(extended_years) - len(species_df)),
                    ]
                )

        # Create projection DataFrame
        projection_df = pd.DataFrame(dict_growth)

        # Extend `species_df` to match the projected years
        species_df = species_df.reindex(range(len(extended_years))).fillna(
            method="ffill"
        )
    else:
        # Use only projection data if `species_df` is None
        projection_df = pd.DataFrame(dict_growth)
        species_df = projection_df.copy()

    species_df["year"] = extended_years
    species_df["sigmoid_dbh_cm"] = dbh_projection

    # Define color mapping
    color_map = {
        0: (255 / 255, 0, 0),  # Red
        1: (0, 255 / 255, 0),  # Green
        2: (0, 0, 255 / 255),  # Blue
        3: (255 / 255, 255 / 255, 0),  # Yellow
        4: (255 / 255, 165 / 255, 0),  # Orange
        5: (128 / 255, 0, 128 / 255),  # Purple
    }
    default_color = (128 / 255, 128 / 255, 128 / 255)  # Gray

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        extended_years,
        dbh_projection,
        label="Projected DBH Growth (Sigmoid) -modelled",
        linewidth=2,
        color="orange",
    )

    if species_df is not None:
        for i, arg in enumerate(args):
            color = color_map.get(i, default_color)
            plt.plot(
                extended_years,
                projection_df[f"{arg}_extended"],
                "--",
                label=f"Model {arg}_extended",
                color=color,
            )

    plt.scatter(
        1,
        initial_dbh,
        color="red",
        zorder=5,
        label=f"Initial DBH = {initial_dbh} cm at Year 1",
    )
    plt.xlabel("Years")
    plt.ylabel("DBH (cm)")
    plt.title(f"Projected DBH Growth for {species}")
    plt.legend()
    plt.grid(True)
    plt.show()

    display(species_df)

    return species_df
