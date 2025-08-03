import pandas as pd
import xarray as xr
import geopandas as gpd
import torch
from shapely.geometry import Point
from shapely.geometry import box
import numpy as np
import pyogrio

def get_events(year):
    """
    Reads event files from Data/EventData to get all fire events for a given year (with times and locations)
    """
    PGE_path = f'Data/EventData/PGE_{year}.xlsx'
    SCE_path = f'Data/EventData/SCE_{year}.xlsx'
    SDGE_path = f'Data/EventData/SDGE_{year}.xlsx'
    PGE_data = pd.read_excel(PGE_path)
    SCE_data = pd.read_excel(SCE_path)
    SDGE_data = pd.read_excel(SDGE_path)
fp = "Data\\Consolidated PSPS Data 20251231.gdb"
pyogrio.list_layers(fp)
psps = gpd.read_file(fp, driver="OpenFileGDB", layer='PSPS_Map_20251231')

#First make the california grid with 0.24 degree resolution
pad, dx = 0.30, 0.24 #Grid dimensions

#Note that the CA bounding box in epsg4326 is (-124.409591, 32.534156, -114.131211, 42.009518)
xmin, xmax = -124.409591-pad, -114.131211+pad
ymin, ymax = 32.534156-pad, 42.009518+pad

cells = [box(x, y, x+dx, y+dx)               # square cells
         for x in np.arange(xmin, xmax, dx)
         for y in np.arange(ymin, ymax, dx)]
grid_gdf = gpd.GeoDataFrame({"cell_id": range(len(cells))}, geometry=cells, crs=4326)

#California outline
#downloading county polygons, "FIPS : 06" is Califirnia
ca = (gpd.read_file("https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json")
      .loc[lambda d: d.id.str.startswith("06")]
      .to_crs(4326) # ensure same CRS
      .dissolve()) # single polygon

#Keep only grid cells that intersect California
grid_gdf = grid_gdf.loc[grid_gdf.geometry.intersects(ca.geometry.iloc[0])].reset_index(drop=True)

event_csv = "data\\all_samples_2024.csv"
raster_nc = "daily_gridded_CA_2024.nc"

def build_event_tensor(event_csv, raster_nc, grid_gdf):
    #Load event data
    events = pd.read_csv(event_csv)
    events = events[events['label'] == 1]
    events["geometry"] = events.apply(lambda row: Point(row["lon"], row["lat"]), axis=1)
    #Convert event time to indexable format
    events["time"] = pd.to_datetime(events["date"])
    gdf_events = gpd.GeoDataFrame(events, geometry="geometry", crs=grid_gdf.crs)

    #Spatial join to assign each event to a grid cell
    gdf_joined = gpd.sjoin(gdf_events, grid_gdf, how="left", predicate="within")
    if gdf_joined.isnull().any().any():
        raise ValueError("Some events did not match any grid cell!")

    #Compute centroids of matched cells
    centroids = grid_gdf.copy().to_crs('EPSG:26910').geometry.centroid.to_crs('EPSG:4326')
    gdf_joined["x_centroid"] = gdf_joined["index_right"].apply(lambda i: centroids.iloc[i].x)
    gdf_joined["y_centroid"] = gdf_joined["index_right"].apply(lambda i: centroids.iloc[i].y)

    #Load raster NetCDF
    ds = xr.open_dataset(raster_nc)

    
    #extract lon/lat coordinates for all cells (1D array)
    lons = ds["lon"].values  # shape: [num_cells]
    lats = ds["lat"].values  # shape: [num_cells]
    cell_ids = ds["cell"].values  # dimension over which lon/lat are indexed

    #raster variables 
    covariate_names = list(ds.data_vars)

    marks = []
    for i, row in gdf_joined.iterrows():
        t = pd.to_datetime(row["time"])
        x = row["lon"]
        y = row["lat"]

        # compute distance to each cell centroid
        dists = np.sqrt((lons - x)**2 + (lats - y)**2)
        closest_cell = cell_ids[np.argmin(dists)]  # scalar

        # extract covariates at nearest time and nearest cell
        sampled = ds.sel(time=t, method="nearest").sel(cell=closest_cell)

        values = [float(sampled[var].values) for var in covariate_names]
        marks.append(values)
    marks = torch.FloatTensor(marks)  # [seq_len, d]

    # Assemble full tensor: [time, x_centroid, y_centroid, *marks]
    time_values = torch.FloatTensor(pd.to_datetime(gdf_joined["time"]).dt.dayofyear)  #to day of year
    x_c = torch.FloatTensor(gdf_joined["x_centroid"].values)
    y_c = torch.FloatTensor(gdf_joined["y_centroid"].values)

    X_seq = torch.cat([
        time_values.unsqueeze(1),
        x_c.unsqueeze(1),
        y_c.unsqueeze(1),
        marks
    ], dim=1)  # [seq_len, data_dim]

    # Final output: batch size = 1
    X_final = X_seq.unsqueeze(0)  # [1, seq_len, data_dim]
    return X_final, covariate_names #tensor, ordered covariate names
data, covs = build_event_tensor(event_csv, raster_nc, grid_gdf)

#first, divide time by 365 to normalize it to (0,1)
data[:,:,0] = data[:,:,0] / 365.0

#GRID CENTROIDS
#compute centroids
centroids = grid_gdf.copy().to_crs('EPSG:26910').geometry.centroid.to_crs('EPSG:4326')
#extract x and y coordinates
xs = torch.tensor(centroids.x.values, dtype=torch.float32)  # [num_cells]
ys = torch.tensor(centroids.y.values, dtype=torch.float32)  # [num_cells]

#stack into a tensor
grid_cells = torch.stack([xs, ys], dim=1)  # [num_cells, 2]


#(divide space)


# training data preparation
train_dataset = TensorDataset(batch)
train_loader  = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_seq      = data[0].squeeze()

# model configurations
T          = (0., 1.)
S          = [(0., 1.), (0., 1.), (0., 1.), (0., 1.), (0., 1.)] #we have 5 covariates here
data_dim   = 8
int_config = ("mc", 2000)
init_model = DeepBasisPointProcess(
    T=T, S=S, mu=1.,
    n_basis=5, basis_dim=10, data_dim=data_dim, 
    int_config=int_config, grid_cells = grid_cells,
    init_gain=0.01, init_bias=0.01, init_std=1,
    nn_width=10)