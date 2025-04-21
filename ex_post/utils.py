import json

import numpy as np


def species_match_coredb_treeocloud(species_name, json_path):
    with open(json_path, "r") as file_species_join:
        file_species_join = json.load(file_species_join)

    return file_species_join.get(species_name, "Not found")


def species_reverse_coredb(species_treeo, json_path):
    with open(json_path, "r") as file_species_join:
        file_species_join = json.load(file_species_join)

    reversed_species_dict = {}
    for k, v in file_species_join.items():
        if isinstance(v, str):  # Correct type check
            reversed_species_dict[v] = k
        elif isinstance(v, list):  # Ensure v is iterable
            for species_treeo_cloud in v:
                reversed_species_dict[species_treeo_cloud] = k
        else:
            raise ValueError(f"Unexpected value type for key '{k}': {type(v)}")
    # print(reversed_species_dict)

    return reversed_species_dict.get(species_treeo, "Not found")


# Function to calculate nearest plot and its Plot_ID
def find_nearest_plot(plot_id_column, reference_gdf, target_gdf):
    """
    Find the nearest plot for each geometry in target_gdf from reference_gdf.
    """
    # Initialize lists to store results
    nearest_plot_ids = []
    min_distances = []

    # Iterate through each geometry in the target GeoDataFrame
    for _, target_row in target_gdf.iterrows():
        # Get the geometry of the current row
        target_geometry = target_row.geometry

        # Compute distances from the target geometry to all geometries in the reference GeoDataFrame
        distances = reference_gdf.geometry.distance(target_geometry)

        # Check if distances series is valid (non-empty and contains valid values)
        if distances.isna().all():
            # Append NaN if all distances are NaN
            nearest_plot_ids.append(np.nan)
            min_distances.append(np.nan)
            continue

        try:
            # Find the index of the minimum distance (ignoring NaN values)
            nearest_idx = distances.idxmin(skipna=True)

            # Use the index to fetch the nearest plot's ID and distance
            nearest_plot_ids.append(reference_gdf.iloc[nearest_idx][plot_id_column])
            min_distances.append(distances.iloc[nearest_idx])
        except (ValueError, IndexError, KeyError) as e:
            # Handle cases where idxmin fails or index is invalid
            nearest_plot_ids.append(np.nan)
            min_distances.append(np.nan)

    # Return the lists as columns
    return nearest_plot_ids, min_distances
