import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import fsolve, curve_fit

# --- All 3 helper functions remain the same ---

def logistic_growth(t, K, r, t0):
    """The core mathematical model for the S-curve."""
    return K / (1 + np.exp(-r * (t - t0)))

def estimate_growth_rate(initial_dbh, max_dbh, inflection_point_guess):
    """Helper for the OLD one-point method (solves for 'r')."""
    def dbh_at_one_year(r):
        return logistic_growth(1, max_dbh, r, inflection_point_guess) - initial_dbh
    r = fsolve(dbh_at_one_year, 0.1)[0]
    return r

def calibrate_model_from_data(time_points, dbh_points, max_dbh, initial_guess_t0=10):
    """Helper for the NEW two-point method (solves for 'r' and 't0')."""
    model_to_fit = lambda t, r, t0: logistic_growth(t, max_dbh, r, t0)
    initial_guesses = [0.1, initial_guess_t0]
    popt, pcov = curve_fit(model_to_fit, time_points, dbh_points, p0=initial_guesses, maxfev=5000)
    return popt[0], popt[1] # r_optimized, t0_optimized


# --- NEW MASTER VISUALIZER FUNCTION ---
def remodel_growth(
    *args: str,
    # Parameters for the one-point model
    initial_dbh: float,
    inflection_point_guess: float,
    # Parameters for the two-point model
    dbh_year_1: float,
    dbh_year_2: float,
    # Common parameters
    max_dbh: float,
    projected_years: int = 35,
    species: str = None,
    species_df: pd.DataFrame = None,
    return_full_df=True
) -> pd.DataFrame:
    """
    Calculates, plots, and compares two logistic growth models on the same graph:
    1. A model based on a single initial data point.
    2. A model calibrated with two data points.
    """
    print(f"--- Comparing Growth Scenarios for: {species} ---")
    
    # --- 1. Calculate parameters for the One-Point Model (the original method) ---
    r_one_point = estimate_growth_rate(initial_dbh, max_dbh, inflection_point_guess)
    t0_one_point = inflection_point_guess # t0 is a fixed guess in this model
    print(f"\n[Model 1: One-Point Estimate]")
    print(f"  - Est. Growth Rate (r): {r_one_point:.4f}")
    print(f"  - Guessed Inflection (t0): {t0_one_point:.2f} years")

    # --- 2. Calculate parameters for the Two-Point Model (the new, more accurate method) ---
    time_data = np.array([1, 2])
    dbh_data = np.array([dbh_year_1, dbh_year_2])
    r_two_point, t0_two_point = calibrate_model_from_data(time_data, dbh_data, max_dbh, initial_guess_t0=inflection_point_guess)
    print(f"\n[Model 2: Two-Point Calibration]")
    print(f"  - Calibrated Growth Rate (r): {r_two_point:.4f}")
    print(f"  - Calibrated Inflection (t0): {t0_two_point:.2f} years")

    # --- 3. Generate both growth projections ---
    extended_years = np.arange(1, projected_years + 1)
    projection_one_point = logistic_growth(extended_years, max_dbh, r_one_point, t0_one_point)
    projection_two_point = logistic_growth(extended_years, max_dbh, r_two_point, t0_two_point)

    # --- 4. Assemble the comparison DataFrame ---
    comparison_data = {
        'year': extended_years,
        'one_point_model_dbh': projection_one_point,
        'two_point_model_dbh': projection_two_point,
    }

    if species_df is not None:
        for arg in args:
            if arg in species_df.columns:
                comparison_data[f"{arg}_extended"] = np.concatenate([
                    species_df[arg].values,
                    [species_df[arg].iloc[-1]] * (len(extended_years) - len(species_df))
                ])
                
    comparison_df = pd.DataFrame(comparison_data)

    # --- 5. Plot everything together ---
    plt.figure(figsize=(12, 8))
    
    # Plot Model 2 (Two-Point, more accurate)
    plt.plot(extended_years, projection_two_point, label="Model from 2 Points (Calibrated)", linewidth=2.5, color="orange")
    plt.scatter(time_data, dbh_data, color="red", zorder=10, s=80, label="Field Data Points (Y1 & Y2)")

    # Plot Model 1 (One-Point)
    plt.plot(extended_years, projection_one_point, label="Model from 1 Point (Guessed t0)", linestyle=':', linewidth=2.5, color="blue")
    plt.scatter(1, initial_dbh, color="blue", marker='x', zorder=10, s=80, label="Initial DBH Point (Y1)")

    # Plot existing data from the DataFrame if provided
    if species_df is not None:
        color_map = {0: 'g', 1: 'purple'}
        default_color = 'grey'
        for i, arg in enumerate(args):
            color = color_map.get(i, default_color)
            if f"{arg}_extended" in comparison_df.columns:
                plt.plot(extended_years, comparison_df[f"{arg}_extended"], "--", label=f"Existing Data: {arg}", color=color)

    plt.xlabel("Years")
    plt.ylabel("DBH (cm)")
    plt.title(f"Comparison of Growth Models for {species}")
    plt.legend()
    plt.grid(True)
    plt.show()

    # display(comparison_df)

    # --- MODIFIED: Step 6: Return the final DataFrame based on user's choice ---
    if return_full_df and species_df is not None:
        # MERGE IS NOW INSIDE: Merge new columns back into the original DataFrame
        print("\n--- Merging projection results back into the original DataFrame ---")
        
        # Ensure the input dataframe has a 'year' column if it's not just an index
        # This makes the merge more robust
        df_to_merge = species_df.copy()
        if 'year' not in df_to_merge.columns:
             df_to_merge['year'] = df_to_merge.index + 1 # Assuming index starts at 0

        final_df = pd.merge(
            df_to_merge,
            comparison_df[['year', 'one_point_model_dbh', 'two_point_model_dbh']],
            on='year',
            how='left'
        )
        display(final_df)
        return final_df
    else:
        # Original behavior: return the smaller comparison dataframe
        display(comparison_df)
        return comparison_df

    return comparison_df



# # Estimate growth rate 'r' so that the model starts close to 'initial_dbh' at year 1
# def estimate_growth_rate(initial_dbh, max_dbh, inflection_point_guess):
#     def dbh_at_one_year(r):
#         # Set target DBH at year 1 to initial_dbh for estimation
#         return logistic_growth(1, max_dbh, r, inflection_point_guess) - initial_dbh

#     # Use fsolve to estimate r, starting with an initial guess
#     r_initial_guess = 0.1
#     r = fsolve(dbh_at_one_year, r_initial_guess)[0]
#     return r


# def remodel_growth(
#     *args: str,
#     initial_dbh: float = 1.0,
#     max_dbh: float = 40,
#     inflection_point_guess: float = 10,
#     projected_years: int = 35,
#     species: str = None,
#     species_df: pd.DataFrame = None,
# ) -> pd.DataFrame:
#     # Calculate the growth rate 'r'
#     r_estimated = estimate_growth_rate(initial_dbh, max_dbh, inflection_point_guess)
#     print(
#         f"Species: {species} | Estimated growth rate r: {r_estimated:.4f} | Inflection point year: {inflection_point_guess}"
#     )

#     # Extend years for projection
#     extended_years = np.arange(1, projected_years + 1)
#     dbh_projection = logistic_growth(
#         extended_years, max_dbh, r_estimated, inflection_point_guess
#     )

#     # Create a dictionary for growth projections
#     dict_growth = {"year": extended_years, "dbh_projection": dbh_projection}

#     if species_df is not None:
#         for arg in args:
#             if arg in species_df.columns:
#                 dict_growth[f"{arg}_extended"] = np.concatenate(
#                     [
#                         species_df[arg].values,
#                         [species_df[arg].iloc[-1]]
#                         * (len(extended_years) - len(species_df)),
#                     ]
#                 )

#         # Create projection DataFrame
#         projection_df = pd.DataFrame(dict_growth)

#         # Extend `species_df` to match the projected years
#         species_df = species_df.reindex(range(len(extended_years))).fillna(
#             method="ffill"
#         )
#     else:
#         # Use only projection data if `species_df` is None
#         projection_df = pd.DataFrame(dict_growth)
#         species_df = projection_df.copy()

#     species_df["year"] = extended_years
#     species_df["sigmoid_dbh_cm"] = dbh_projection

#     # Define color mapping
#     color_map = {
#         0: (255 / 255, 0, 0),  # Red
#         1: (0, 255 / 255, 0),  # Green
#         2: (0, 0, 255 / 255),  # Blue
#         3: (255 / 255, 255 / 255, 0),  # Yellow
#         4: (255 / 255, 165 / 255, 0),  # Orange
#         5: (128 / 255, 0, 128 / 255),  # Purple
#     }
#     default_color = (128 / 255, 128 / 255, 128 / 255)  # Gray

#     # Plot
#     plt.figure(figsize=(10, 6))
#     plt.plot(
#         extended_years,
#         dbh_projection,
#         label="Projected DBH Growth (Sigmoid) -modelled",
#         linewidth=2,
#         color="orange",
#     )

#     if species_df is not None:
#         for i, arg in enumerate(args):
#             color = color_map.get(i, default_color)
#             plt.plot(
#                 extended_years,
#                 projection_df[f"{arg}_extended"],
#                 "--",
#                 label=f"Model {arg}_extended",
#                 color=color,
#             )

#     plt.scatter(
#         1,
#         initial_dbh,
#         color="red",
#         zorder=5,
#         label=f"Initial DBH = {initial_dbh} cm at Year 1",
#     )
#     plt.xlabel("Years")
#     plt.ylabel("DBH (cm)")
#     plt.title(f"Projected DBH Growth for {species}")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     display(species_df)

#     return species_df
