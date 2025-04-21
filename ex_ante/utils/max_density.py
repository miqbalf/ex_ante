# --- Define the lookup function ---
def get_max_density(input_dbh, lookup_df, threshold_col, value_col):
    """
    Finds the max density where the DBH threshold > input_dbh.
    Mimics =MAX(IF(input_dbh < thresholds, densities))
    """
    # Filter the lookup dataframe for rows where the threshold is greater than the input
    eligible_rows = lookup_df[lookup_df[threshold_col] > input_dbh]

    # Check if any rows were found
    if eligible_rows.empty:
        # No threshold is greater than input_dbh. Excel's MAX(IF()) might return 0 here.
        # Return 0 or NaN based on desired behavior for out-of-range inputs.
        return 0  # Or return np.nan
    else:
        # Return the maximum value from the target column in the filtered rows
        return eligible_rows[value_col].max()
