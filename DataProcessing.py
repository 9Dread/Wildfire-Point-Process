import pandas as pd
import xarray as xr
import geopandas as gpd
import torch
from shapely.geometry import Point
from shapely.geometry import box
import numpy as np
#import pyogrio
from torch.utils.data import Dataset
from rasterstats import zonal_stats
import re, json, urllib.request

#need openpyxl for reading excel as well

#Filled in some missing lat/lon vals in SCE 2021 (since our resolution is much coarser):
#  Sun village circuit (Sun village, CA) 34.559444 -117.956667 from MapQuest
#  Trumpet Circuit 34.488541 -118.6167840 from another Trumpet Circuit event (2020)
#  2 more missing, unable to find facility/circuit location (DROPPED)

#SCE 2020 had 3 erroneous longitude values (2 missing negative sign, 1 missing -1 in the front so that lon = 16.)
#SCE 2021 had 1 missing negative sign on the longitude

#FIRE EVENT PROCESSING:
def read_event_file(utility, year):
    """
    Helper for get_events.
    Takes a filepath (to one of the event excel files) and reads it. Processes date and time columns into a datetime.
    """
    path = f'Data/EventData/{utility}_{year}.xlsx'
    if utility == "PGE":
        if year == 2020:
            data = pd.read_excel(path, usecols=[2,3,4,5], skiprows=2, header=None, names = ["Date", "Time", "Lat", "Lon"])
            data["T"] = data["Date"].astype(str) + ' ' + data["Time"].astype(str)
            #have to align formats of time for some rows (which are missing seconds)
            mask = data["T"].str.count(":") == 1
            data.loc[mask, 'T'] = data.loc[mask, 'T'] + ':00'
            data["T"] = pd.to_datetime(data["T"])
            return(data.drop(columns = ["Date","Time"], axis=1))
        else:
            data = pd.read_excel(path, usecols=[2,3,4,5], skiprows=2, header=None, names = ["Date", "Time", "Lat", "Lon"])
            data["T"] = data["Date"].astype(str) + ' ' + data["Time"].astype(str)
            data["T"] = pd.to_datetime(data["T"])
        return(data.drop(columns = ["Date","Time"], axis=1))
    if utility == "SCE":
        if year == 2020:
            #this year has different column formats
            data = pd.read_excel(path, usecols=[3,5,6,7], skiprows=1, header=None, names = ["Date", "Time", "Lat", "Lon"])
            data["T"] = data["Date"].astype(str) + ' ' + data["Time"].astype(str)
            data = data.drop(index = list(range(149, 159)), axis=0)
            data["T"] = pd.to_datetime(data["T"])
            return(data.drop(columns = ["Date","Time"], axis=1))
        else:
            data = pd.read_excel(path, usecols=[1,2,3,4], skiprows=2, header=None, names = ["Date", "Time", "Lat", "Lon"])
            data["T"] = data["Date"].astype(str) + ' ' + data["Time"].astype(str)
            data["T"] = pd.to_datetime(data["T"])
            return(data.drop(columns = ["Date","Time"], axis=1))
    if utility == "SDGE":
        if year == 2024:
            #has cols at different indices
            data = pd.read_excel(path, usecols=[4,5,6,7], skiprows=2, header=None, names = ["Date", "Time", "Lat", "Lon"])
            data["T"] = data["Date"].astype(str) + ' ' + data["Time"].astype(str)
            data["T"] = pd.to_datetime(data["T"])
            return(data.drop(columns = ["Date","Time"], axis=1))
        else:
            data = pd.read_excel(path, usecols=[1,2,3,4], skiprows=2, header=None, names = ["Date", "Time", "Lat", "Lon"])
            data["T"] = data["Date"].astype(str) + ' ' + data["Time"].astype(str)
            data["T"] = pd.to_datetime(data["T"])
            return(data.drop(columns = ["Date","Time"], axis=1))

def get_events(year):
    """
    Reads event files from Data/EventData to get all fire events for a given year (with times and locations)
    """
    PGE_data = read_event_file("PGE", year)
    SCE_data = read_event_file("SCE", year)
    SDGE_data = read_event_file("SDGE", year)
    return(pd.concat([PGE_data, SCE_data, SDGE_data], ignore_index=True).dropna())

def events_to_tensor(events_df, grid_gdf, time_res):
    """
    events_df: Dataframe from get_events()
    grid_gdf: geodataframe of grid cells from get_point24deg_grid
    time_res: either "6hr" or "daily"
    
    and converts to the gridded time/space format
    (attaches day of the year and grid cell)
    
    returns Tensor of shape (n_events, 2) with the time and space indexes of the events
    """
    assert (time_res == "daily") | (time_res == "6hr"), "Invalid time_res. Use 'daily' or '6hr'"
    #make geometry, geodataframe
    events_df["geometry"] = events_df.apply(lambda row: Point(row["Lon"], row["Lat"]), axis=1)
    gdf_events = gpd.GeoDataFrame(events_df, geometry="geometry", crs=4326)

    #spatial join to assign each event to a grid cell
    events_joined = gpd.sjoin(gdf_events, grid_gdf, how="left", predicate="within")
    if events_joined.isnull().any().any():
        raise ValueError("Some events did not match any grid cell!")
    
    #map T to a time index
    if time_res == "daily":
        year = events_joined["T"][0].year
        events_joined["T"] = events_joined["T"].dt.dayofyear - 1 #0-indexed day of year
        if year == "2020": #We dropped first day of 2020 so if the year is 2020, adjust T index
            events_joined["T"] = events_joined["T"] - 1
            events_joined = events_joined["T" >= 0]
        events_joined = events_joined.sort_values(by="T", ascending=True) #sort
    if time_res == "6hr":
        pass #not implemented yet
    spacetime = events_joined[["T", "cell_id"]]
    return torch.Tensor(spacetime.to_numpy())

def get_events_tensor_list(time_res):
    """
    Returns list of tensors of fire events for each year. Each tensor has the time and space index of each event.
    time_res: 'daily' or '6hr'
    """
    assert (time_res == "daily") | (time_res == "6hr"), "Invalid time_res. Use 'daily' or '6hr'"
    years = [2020, 2021, 2022, 2023, 2024]
    events_df_list = [get_events(year) for year in years]
    grid_gdf = get_point24deg_grid()
    return([events_to_tensor(events, grid_gdf, time_res) for events in events_df_list])

#RETURNS SPATIAL GRID WE AGGREGATE COVS TO:
def get_point24deg_grid():
    """
    Makes the grid (0.24-degree resolution) to which we aggregated our covariates.
    Four cells have missing covariates (due to resolution issues). These are dropped.
    """
    drop_missing_cov_cells = True
    pad, dx = 0.30, 0.24 #grid dimensions
    #note that the CA bounding box in epsg4326 is (-124.409591, 32.534156, -114.131211, 42.009518)
    xmin, xmax = -124.409591-pad, -114.131211+pad
    ymin, ymax = 32.534156-pad, 42.009518+pad
    cells = [box(x, y, x+dx, y+dx)               # square cells
            for x in np.arange(xmin, xmax, dx)
            for y in np.arange(ymin, ymax, dx)]
    grid_gdf = gpd.GeoDataFrame({"cell_id": range(len(cells))}, geometry=cells, crs=4326)
    #california outline
    #downloading county polygons, "FIPS : 06" is Califirnia
    ca = (gpd.read_file("https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json")
      .loc[lambda d: d.id.str.startswith("06")]
      .to_crs(4326) #ensure same CRS
      .dissolve()) #single polygon
    #keep only grid cells that intersect California
    grid_gdf = grid_gdf.loc[grid_gdf.geometry.intersects(ca.geometry.iloc[0])].reset_index(drop=True)
    grid_gdf["cell_id"] = grid_gdf.index
    if(drop_missing_cov_cells):
        grid_gdf = grid_gdf.drop(index = [71, 439, 461, 521]).reset_index(drop=True)
        grid_gdf["cell_id"] = grid_gdf.index
    return grid_gdf

#COVARIATES PROCESSING:

def tensor_gridded_covs(year):
    """
    Loads the tensor grid of the covariates for a given year. (T, C, p)
    """
    nc_dataset = xr.open_dataset(f"Data/CovsDaily/daily_gridded_CA_{year}.nc")
    array = nc_dataset.to_array()
    array = array.transpose("time", "cell", "variable") #reorder stuff
    arr = array.values #shape (T, C, p)
    return(torch.from_numpy(arr).float(), list(nc_dataset.data_vars.keys()))

def get_covs_tensor_list():
    """
    Returns a list of tensors that contain the covariates from tensor_gridded_covs()
    """

    years = [2020, 2021, 2022, 2023, 2024]
    tensor_list = []
    for year in years:
        tensor, names_list = tensor_gridded_covs(year)
        tensor_list.append(tensor)
        #return list of tensors, variable names
    return tensor_list, names_list 
    
    
def standardize_cov_tensors(list):
    """
    Given the list of the covariate tensors, return a list of standardized tensors
    list: list of tensors (1 tensor per year)
    """
    p = list[0].shape[2] #number of covariates
    #quick standardization
    all_data = torch.cat([c.view(-1, p) for c in list], dim=0)
    means = all_data.mean(dim=0) #shape (p,)
    stds  = all_data.std(dim=0) #shape (p,)
    standardized = [(cov - means) / stds for cov in list]
    return(standardized)

class WildfireDataset(Dataset):
    """
    Definitions of some things:
    T_y = time steps in the given year y (e.g. at daily resolution, 365 for 2021-2023 and 366 for 2020 and 2024)
    C = number of grid cells
    p = number of covariates
    N_y = number of events in the given year y

    covs: list, each a tensor (T_y, C, p)
    events: list, each a tensor shape (N_y, 2)
        where each row is (t_i, c_i) index for an event in that year
    """

    def __init__(self, covs, events):
        self.covs = covs
        self.events = events
    
    def __len__(self):
        return len(self.covs)
    
    def __getitem__(self, idx):
        cov = self.covs[idx] #(T_y, C, p)

        tcs = self.events[idx].long() #(N_y, 2)

        return cov, tcs



#INHIBITORY EVENTS DATA FILES PROCESSING
class WildfireInhibDataset(Dataset):
    """
    covs: list, each a tensor (T_y, C, p)
    inhib: list, each a tensor (T_y,C,2)
    events: list, each a tensor shape (N_y, 2)
        where each row is (t_i, c_i) index for an event in that year
    """

    def __init__(self, covs, inhib, events):
        self.covs = covs
        self.inhib = inhib
        self.events = events
    
    def __len__(self):
        return len(self.covs)
    
    def __getitem__(self, idx):
        cov = self.covs[idx] #(T_y, C, p)
        inh = self.inhib[idx] #(T_y,C,2)
        tcs = self.events[idx].long() #(N_y, 2)
        return cov, inh, tcs

def get_psps():
    """
    Reads the psps data file
    """ 
    fp = "Data\\Consolidated PSPS Data 20251231.gdb"
    #pyogrio.list_layers(fp)
    psps = gpd.read_file(fp, driver="OpenFileGDB", layer='PSPS_Map_20251231')
    return psps

def get_psps_events_list():
    """
    Gets list of 4 geodataframes, one for each year 2021-24. Each contains all psps events in the year that have
    start and end dates as well as a polygon of affected areas. 
    Note that 2020 has no psps events.
    """
    psps = get_psps().to_crs(4326)
    #filter to events which de-energized
    psps = psps[psps['De_Energization'] == 'Yes']
    #get start/end time, polygon
    psps = psps.loc[:,['DeEnergizationStartDate', 'FullRestorationDate', 'geometry']].rename(columns = {'DeEnergizationStartDate' : 'Start', 'FullRestorationDate': 'End'})
    psps["Start"] = psps["Start"].dt.tz_localize(None)
    psps["End"] = psps["End"].dt.tz_localize(None)

    years = [2021, 2022, 2023, 2024]
    out_list = []
    for year in years:
        out_list.append(psps[psps['Start'].dt.year == year].copy().reset_index())
    return out_list

def get_epss_events_list():
    """
    Gets list of 4 geodataframes, one for each year 2021-24. Each contains all epss events in the year that have
    start and end dates as well as a polygon of the county where the event occurred.
    Note that 2020 has no epss events.
    """
    years = [2021, 2022, 2023, 2024]
    out_list = []

    #Map county names to polygons; code from Maitreya
    #Helper
    def normalize_county_name(s):
        if pd.isna(s):
            return None
        x = str(s).strip().lower()
        x = x.replace("&","and")
        x = re.sub(r"^city and county of\s+","",x)
        x = re.sub(r"\s+county$","",x)
        x = re.sub(r"[^\w\s-]","",x)
        x = " ".join(x.split())
        return x.title()

    with urllib.request.urlopen("https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json") as resp:
        counties_json = json.load(resp)
    ca_feats = [f for f in counties_json["features"] if f["id"].startswith("06")]
    CA = gpd.GeoDataFrame.from_features(ca_feats, crs=4326)
    if "NAME" in CA.columns:
        CA = CA[["NAME","geometry"]].rename(columns={"NAME":"County"})
    elif "name" in CA.columns:
        CA = CA[["name","geometry"]].rename(columns={"name":"County"})
    CA["county_norm"] = CA["County"].map(normalize_county_name)

    for year in years:
        #Process this year's data frame
        epss = pd.read_excel(f'Data/EPSS/EPSS_{year}.xlsx').rename(columns=str.strip)
        epss = epss[epss["EPSS Outage Type"].str.upper() == "FTS"].copy().rename(columns = {'FNL' : 'Start', 'End_Date_Time' : 'End'})
        epss["Start"] = pd.to_datetime(epss["Start"], errors="coerce")
        epss["End"] = pd.to_datetime(epss["End"], errors="coerce")
        epss["county_norm"] = epss["County"].map(normalize_county_name)
        epss_joined = epss.merge(CA[["County","county_norm","geometry"]],
                    on="county_norm", how="inner",
                    suffixes=("_epss","_poly"))
        epss_joined = gpd.GeoDataFrame(epss_joined, geometry="geometry", crs=4326)
        out_list.append(epss_joined.loc[:,['Start', 'End', 'geometry']])
    return out_list

def inhib_to_tensor(psps_gdf, epss_gdf, grid_gdf):
    """
    given psps and epss gdfs, each of which have Start, End, geometry columns describing the spacetime
    occurence of inhibitory events, return a (T,C,2) indicator tensor of whether cells in grid_gdf
    currently intersect an active area.
    """
    year = psps_gdf.loc[0,'Start'].year
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    T = len(days)
    C = len(grid_gdf)

    #helper, only used here so i'm just gonna define it here
    def daily_hits(gdf, grid_gdf):
        arr = np.zeros((T, C), dtype=np.float32)
        for d_idx, d in enumerate(days):
            #select polygons active on this day
            active = gdf[(gdf["Start"] <= d) & (gdf["End"] >= d)]
            if active.empty:
                continue
            #spatial join: which cells are touched by those polygons
            joined = gpd.sjoin(
                grid_gdf[["cell_id", "geometry"]],
                active.set_geometry("geometry"),
                how="inner",
                predicate="intersects"
            )
            arr[d_idx, joined["cell_id"].to_numpy()] = 1.0
        return arr
    
    psps_arr = daily_hits(psps_gdf, grid_gdf)
    epss_arr = daily_hits(epss_gdf, grid_gdf)
    return torch.tensor(np.stack([psps_arr, epss_arr], axis=-1))

def get_inhib_tensor_list(time_res='daily'):
    """
    time_res: time resolution to get events at. currently only 'daily' is supported.
    
    Reads psps/epss events data and returns list of 5 tensors (T,C,2), one for each year. Note
    that there are no events for 2020, so this year will just have a zeros tensor.
    """
    #grid to aggregate to
    grid_gdf = get_point24deg_grid()

    #just treat time_res like daily for now, update later if needed
    psps_list = get_psps_events_list()
    epss_list = get_epss_events_list()

    out_list = [torch.zeros(size=(365,len(grid_gdf),2),dtype=torch.float)] #2020 grid of zeros (365 days since we dropped jan 1 from daily covs)
    for i in range(0, len(psps_list)):
        out_list.append(inhib_to_tensor(psps_list[i], epss_list[i], grid_gdf)) #append T,C,2 tensor for each year
    return out_list



#HELPERS FOR preproc.py 
def aggregate_tif_to_cells(tif_path, grid_gdf_aligned, stats="mean", nodata=None, all_touched=False):
    """
    Returns a 1D numpy array of length n_cells with the aggregated value per cell polygon.
    grid_gdf_aligned must be in the same order as the desired cell order. Used to aggregate
    fm100 and fm1000 from 12/31/2019 to the desired grid cells.
    """
    #compute means in each grid cell
    zs = zonal_stats(
        vectors=grid_gdf_aligned.geometry,
        raster=str(tif_path),
        stats=stats,
        nodata=nodata,
        all_touched=all_touched,
        geojson_out=False
    )
    vals = np.array([d[stats] if d[stats] is not None else np.nan for d in zs], dtype=float)
    return vals

def ensure_grid_order_matches(ds, grid_gdf, id_col="cell_id"):
    """
    Align grid_gdf rows to match ds.cell (xarray ds) coordinate order via 'id' matching.
    Returns aligned GeoDataFrame and an indexer to reindex any arrays defined on grid_gdf rows.
    Requires ds to have a 'cell' coordinate (which are ids).
    """
    cell_coord = ds["cell"].values
    #if cell is numeric 0..N-1 and grid_gdf[id_col] matches that sequence, good.
    #otherwise, treat cell_coord as the ids to align to.
    #build a mapping from id to row index in grid_gdf
    id_to_idx = {rid: i for i, rid in enumerate(grid_gdf[id_col].values)}
    try:
        idxs = np.array([id_to_idx[rid] for rid in cell_coord], dtype=int)
    except KeyError as e:
        raise ValueError(f"Found cell id {e} in ds that does not exist in grid_gdf[{id_col}].")
    return grid_gdf.iloc[idxs].reset_index(drop=True), idxs

def shift_forward_one_year(old_vals, first_day_value):
    """
    old_vals: (time, cell) array for a single year (DataArray values)
    first_day_value: (cell,) array to insert at index 0
    returns new_vals with same shape as old_vals, applying forward shift:
        new[0] = first_day_value
        new[1:] = old[:-1]
    """
    if first_day_value.shape[0] != old_vals.shape[1]:
        raise ValueError("first_day_value length does not match number of cells.")
    new_vals = np.empty_like(old_vals)
    new_vals[0, :] = first_day_value
    new_vals[1:, :] = old_vals[:-1, :]
    return new_vals

def grid_to_cell_coords(grid_gdf, metric_crs=False):
    """
    Given a GeoDataFrame `grid_gdf` with columns 'cell_id' and 'geometry',
    returns a NumPy array of shape (C, 2) where each row i is the (x, y)
    centroid of the cell with cell_id == i. Rows are ordered by ascending cell_id.
    """
    #sort by cell_id to ensure consistent ordering
    gdf_sorted = grid_gdf.sort_values("cell_id")
    #compute centroids
    if metric_crs:
        centroids = gdf_sorted.to_crs(3310).geometry.centroid
    else:
        centroids = gdf_sorted.to_crs(3310).geometry.centroid.to_crs(4326)
    #extract x, y coordinates
    xs = centroids.x.values
    ys = centroids.y.values
    #stack into an (C, 2) array of (x, y) pairs
    cell_coords = np.stack([xs, ys], axis=1)
    return cell_coords

