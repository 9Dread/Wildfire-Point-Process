
import pandas as pd
import xarray as xr
import geopandas as gpd
import numpy as np
from shapely.geometry import box

ca = (gpd.read_file("https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json")
    .loc[lambda d: d.id.str.startswith("06")]
    .to_crs(4326) #ensure same CRS
    .dissolve()) #single polygon
def get_point24deg_grid():
    """
    Makes the grid (0.24-degree resolution) to which we aggregated our covariates.
    Four cells have missing covariates (due to resolution issues). These are dropped.
    """
    #drop_missing_cov_cells = True
    pad, dx = 0.30, 0.24 #grid dimensions
    #note that the CA bounding box in epsg4326 is (-124.409591, 32.534156, -114.131211, 42.009518)
    xmin, xmax = -124.409591-pad, -114.131211+pad
    ymin, ymax = 32.534156-pad, 42.009518+pad
    cells = [box(x, y, x+dx, y+dx) #square cells
            for x in np.arange(xmin, xmax, dx)
            for y in np.arange(ymin, ymax, dx)]
    grid_gdf = gpd.GeoDataFrame({"cell_id": range(len(cells))}, geometry=cells, crs=4326)
    #california outline
    #downloading county polygons, "FIPS : 06" is Califirnia
    ca = (gpd.read_file("ca_state/CA_state.shp")
    #  .loc[lambda d: d.STATEFP.str.startswith("06")]
      .to_crs(4326) #ensure same CRS
      .dissolve()) #single polygon
    #keep only grid cells that intersect California
    grid_gdf = grid_gdf.loc[grid_gdf.geometry.within(ca.geometry.iloc[0])].reset_index(drop=True)
    grid_gdf["cell_id"] = grid_gdf.index
    #if(drop_missing_cov_cells):
    #    grid_gdf = grid_gdf.drop(index = [71, 439, 461, 521]).reset_index(drop=True)
    #    grid_gdf["cell_id"] = grid_gdf.index
    return grid_gdf
