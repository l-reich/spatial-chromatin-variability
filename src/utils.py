import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

def get_centroid(row):
    all_x = []
    all_y = []
    # Collect boundary coordinates, handle NaN gracefully
    for i in range(7):  # Adjust based on the number of slices
        x_col = f"boundaryX_z{i}"
        y_col = f"boundaryY_z{i}"

        if pd.notna(row[x_col]) and pd.notna(row[y_col]):
            # Ensure we are extracting valid data
            x_values = list(map(float, str(row[x_col]).split(',')))

            y_values = list(map(float, str(row[y_col]).split(',')))

            if len(x_values) == len(y_values):  # Ensure matching x/y pairs
                all_x.extend(x_values)
                all_y.extend(y_values)

    # Proceed only if enough coordinates are available
    if len(all_x) > 2 and len(all_y) > 2:  # Ensure enough points for a polygon
        polygon = Polygon(zip(all_x, all_y))
        return polygon.centroid.x, polygon.centroid.y
        #if polygon.is_valid and not polygon.is_empty:
         #   return polygon.centroid.x, polygon.centroid.y
    
    # If there are no valid polygon points, use a fallback (average of available points)
    if len(all_x) > 0 and len(all_y) > 0:
        return np.mean(all_x), np.mean(all_y)
    return np.nan, np.nan  # Return NaN if no valid coordinates
