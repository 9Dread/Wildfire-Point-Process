import pandas as pd
import xarray as xr
import geopandas as gpd
import torch
from shapely.geometry import Point
from shapely.geometry import box
import numpy as np
#import pyogrio
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio
from rasterstats import zonal_stats
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogFormatterMathtext

#need openpyxl for reading excel as well

#Filled in some missing lat/lon vals in SCE 2021 (since our resolution is much coarser):
#  Sun village circuit (Sun village, CA) 34.559444 -117.956667 from MapQuest
#  Trumpet Circuit 34.488541 -118.6167840 from another Trumpet Circuit event (2020)
#  2 more missing, unable to find facility/circuit location (DROPPED)

#SCE 2020 had 3 erroneous longitude values (2 missing negative sign, 1 missing -1 in the front so that lon = 16.)
#SCE 2021 had 1 missing negative sign on the longitude

#CELL IDS WITH MISSING COVARIATES: 71 439 461 521

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


def get_psps():
    fp = "Data\\Consolidated PSPS Data 20251231.gdb"
    #pyogrio.list_layers(fp)
    psps = gpd.read_file(fp, driver="OpenFileGDB", layer='PSPS_Map_20251231')
    return(psps)


def get_point24deg_grid(drop_missing_cov_cells = False):
    """
    Makes the grid (0.24-degree resolution) to which we aggregated our covariates.
    Four cells have missing covariates (due to resolution issues). These can be dropped.
    """
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
    grid_gdf["cell_id"] = grid_gdf.index
    if(drop_missing_cov_cells):
        grid_gdf = grid_gdf.drop(index = [71, 439, 461, 521]).reset_index(drop=True)
        grid_gdf["cell_id"] = grid_gdf.index
    return(grid_gdf)

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
    #Make geometry, geodataframe
    events_df["geometry"] = events_df.apply(lambda row: Point(row["Lon"], row["Lat"]), axis=1)
    gdf_events = gpd.GeoDataFrame(events_df, geometry="geometry", crs=4326)

    #Spatial join to assign each event to a grid cell
    events_joined = gpd.sjoin(gdf_events, grid_gdf, how="left", predicate="within")
    if events_joined.isnull().any().any():
        raise ValueError("Some events did not match any grid cell!")
    
    #map T to a time index
    if time_res == "daily":
        events_joined["T"] = events_joined["T"].dt.dayofyear - 1 #0-indexed day of year
        events_joined = events_joined.sort_values(by="T", ascending=True) #sort
    if time_res == "6hr":
        pass #not implemented yet
    spacetime = events_joined[["T", "cell_id"]]
    return(torch.Tensor(spacetime.to_numpy()))

def get_events_tensor_list(time_res, drop_missing_cov_cells = False):
    """
    Returns list of tensors of fire events for each year. Each tensor has the time and space index of each event.
    time_res: 'daily' or '6hr'
    """
    assert (time_res == "daily") | (time_res == "6hr"), "Invalid time_res. Use 'daily' or '6hr'"
    years = [2020, 2021, 2022, 2023, 2024]
    events_df_list = [get_events(year) for year in years]
    grid_gdf = get_point24deg_grid(drop_missing_cov_cells)
    return([events_to_tensor(events, grid_gdf, time_res) for events in events_df_list])

#OLD: only gets marcos covs
def tensor_gridded_covs(year, drop_missing_cov_cells = False):
    """
    Loads the tensor grid of the covariates for a given year. (T, C, p)
    """
    nc_dataset = xr.open_dataset(f"Data/MarcosCovs/daily_gridded_CA_{year}.nc")
    if(drop_missing_cov_cells):
        missing_ids = [71, 439, 461, 521]
        nc_dataset = nc_dataset.drop_sel(cell=missing_ids)
        new_ncell = nc_dataset.sizes["cell"]
        nc_dataset = nc_dataset.assign_coords(cell=np.arange(new_ncell))
    array = nc_dataset.to_array()
    array = array.transpose("time", "cell", "variable") #reorder stuff
    arr = array.values #shape (T, C, p)
    return(torch.from_numpy(arr).float())

#new modified ver
def tensor_gridded_covs_new(year, drop_missing_cov_cells = False):
    """
    Loads the tensor grid of the covariates for a given year. (T, C, p)
    """
    nc_dataset = xr.open_dataset(f"Data/CombinedCovs/daily_gridded_CA_{year}.nc")
    if(drop_missing_cov_cells):
        missing_ids = [71, 439, 461, 521]
        nc_dataset = nc_dataset.drop_sel(cell=missing_ids)
        new_ncell = nc_dataset.sizes["cell"]
        nc_dataset = nc_dataset.assign_coords(cell=np.arange(new_ncell))
    array = nc_dataset.to_array()
    array = array.transpose("time", "cell", "variable") #reorder stuff
    arr = array.values #shape (T, C, p)
    return(torch.from_numpy(arr).float())

def get_covs_tensor_list(drop_missing_cov_cells = False, new = False):
    """
    Returns a list of tensors that contain the covariates from tensor_gridded_covs()
    """
    if new:
        #we only have 2023, 2024 for now
        years = [2023, 2024]
        return([tensor_gridded_covs_new(year, drop_missing_cov_cells) for year in years]) 
    else:
        years = [2020, 2021, 2022, 2023, 2024]
        return([tensor_gridded_covs(year, drop_missing_cov_cells) for year in years])    
    
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
    
    #change to not return mask, just return events tensor (N_y,2)
    def __getitem__(self, idx):
        cov = self.covs[idx] #(T_y, C, p)
        T_y, C, p = cov.shape

        #build an event mask the same shape (T_y, C)
        mask = torch.zeros((T_y, C), dtype=torch.bool)

        tcs = self.events[idx] #(N_y, 2)
        
        if not (torch.is_tensor(tcs) and tcs.dtype == torch.long):
            tcs = torch.as_tensor(tcs, dtype=torch.long)

        mask[tcs[:,0], tcs[:,1]] = True

        return cov, tcs

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

#VISUALIZATION:
def animate_poisson_intensity(model, cov_tensor, events, cell_coords, output_path, fps=5,
                      decay=0.8, figsize=(6,6), device=None):
    """
    model: callable mapping covariates (T, C, p) to intensity (T, C)
    cov_tensor: torch.Tensor, shape (T, C, p); all covariates for all grid cells for each day in the year
    events: int torch.Tensor, shape (N_y, 2); time step and grid cell ids of all events for the year
    cell_coords: np.ndarray shape (C,2) of (x,y) centroids for each cell
    output_path: path to save gif to
    fps: fps of the gif
    decay: float in (0,1), controls per-frame decay of event dots
    figsize: size of the figure
    """
    #check device and tensors, make sure everything is correct
    if device is None:
        if isinstance(model, torch.nn.Module):
            dev = next(model.parameters()).device
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)
    
    if not torch.is_tensor(cov_tensor):
        cov = torch.tensor(cov_tensor, dtype=torch.float32, device=dev)
    else:
        cov = cov_tensor.to(dev).float()
    if cov.ndim == 4 and cov.shape[0] == 1:
        cov = cov.squeeze(0)

    T, C, p = cov.shape
    #build an event mask the same shape (T, C)
    event_mask = torch.zeros((T, C), dtype=torch.bool, device=dev)
    if not (torch.is_tensor(events) and events.dtype == torch.long):
        events = torch.as_tensor(events, dtype=torch.long, device = dev)
    else:
        events = events.to(dev)
    if events.ndim == 3 and events.shape[0] == 1:
        events = events.squeeze(0)
    event_mask[events[:,0], events[:,1]] = True

    ev = event_mask.to(dev).float() #need float to do computations for viz
    if ev.ndim == 3:
        ev = ev.squeeze(0)
    
    with torch.no_grad():
        lam_t = model(cov)
    if lam_t.ndim == 3 and lam_t.shape[0] == 1:
        lam_t = lam_t.squeeze(0)
    if lam_t.ndim == 3 and lam_t.shape[-1] == 1:
        lam_t = lam_t.squeeze(-1)
    if lam_t.ndim != 2:
        raise ValueError(f"Expected intensity shape (T,C), got {lam_t.shape}")
    lam = lam_t.cpu().numpy()  #now (T, C)

    xs, ys = cell_coords[:,0], cell_coords[:,1]
    event_disp = np.zeros(C, dtype=float)

    #setup figure
    fig, ax = plt.subplots(figsize=figsize)
    sc_int = ax.scatter(xs, ys, c=lam[0],
                        vmin=lam.min(), vmax=lam.max(), s=20, cmap="OrRd")
    sc_evt = ax.scatter(xs, ys, s=0, alpha=0.0)
    plt.colorbar(sc_int, ax=ax, label='lambda intensity')
    ax.set_axis_off()

    #update function
    def update(t):
        nonlocal event_disp
        #update intensity colors
        arr = lam[t] #1D of length C
        sc_int.set_array(arr)
        #update decaying event size + alpha
        event_disp = event_disp * decay + ev[t].cpu().numpy()
        event_disp = np.clip(event_disp, 0.0, 1.0)
        sc_evt.set_sizes(200 * event_disp)
        sc_evt.set_alpha(event_disp)
        ax.set_title(f"Time step {t}")
        return sc_int, sc_evt

    frames = []
    for t in range(T):
        update(t)           #redraw artists for frame t
        fig.canvas.draw()        #render the canvas
        #grab the RGB buffer from the figure
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = buf.reshape(h, w, 3)
        frames.append(img)

    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Saved GIF to {output_path}")


def _log_norm_from_pos(arr, p_lo=5, p_hi=99.5, floor=1e-8):
    """
    Build a LogNorm from positive entries of arr.
    Uses percentiles to avoid outliers. Falls back to linear if no positives.

    Makes visualization of kernel intensity visible.
    """
    pos = arr[arr > 0]
    if pos.size == 0:
        #fall back: linear 0..1 to avoid crashes
        return None, Normalize(vmin=0.0, vmax=1.0, clip=True)
    vmin = max(np.percentile(pos, p_lo), floor)
    vmax = max(np.percentile(pos, p_hi), vmin * 1.001)
    return LogNorm(vmin=vmin, vmax=vmax, clip=True), None

def animate_hawkes_intensity(model, cov_tensor, events, cell_coords, output_path, fps=5,
                      decay=0.8, figsize=(6,6), separate_base_ker=False, device=None):
    """
    model: callable mapping covariates (T, C, p) and events (N,2) to intensity (T, C); should be hawkes model with separable intensity
    cov_tensor: torch.Tensor, shape (T, C, p); all covariates for all grid cells for each day in the year
    events: int torch.Tensor, shape (N_y, 2); time step and grid cell ids of all events for the year
    cell_coords: np.ndarray shape (C,2) of (x,y) centroids for each cell
    output_path: path to save gif to
    fps: fps of the gif
    decay: float in (0,1), controls per-frame decay of event dots
    figsize: size of the figure
    separate_base_ker: whether to make separate animations for the base intensity and the excitation kernel intensity
    """
    #check device and tensors, make sure everything is correct
    if device is None:
        if isinstance(model, torch.nn.Module):
            dev = next(model.parameters()).device
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)
    
    if not torch.is_tensor(cov_tensor):
        cov = torch.tensor(cov_tensor, dtype=torch.float32, device=dev)
    else:
        cov = cov_tensor.to(dev).float()
    if cov.ndim == 4 and cov.shape[0] == 1:
        cov = cov.squeeze(0)

    T, C, p = cov.shape
    #build an event mask the same shape (T, C)
    event_mask = torch.zeros((T, C), dtype=torch.bool, device=dev)
    if not (torch.is_tensor(events) and events.dtype == torch.long):
        events = torch.as_tensor(events, dtype=torch.long, device = dev)
    else:
        events = events.to(dev)
    if events.ndim == 3 and events.shape[0] == 1:
        events = events.squeeze(0)
    event_mask[events[:,0], events[:,1]] = True

    ev = event_mask.to(dev).float() #need float to do computations for viz
    if ev.ndim == 3:
        ev = ev.squeeze(0)
    #get lambda, parts
    with torch.no_grad():
        lam_t, parts = model(cov, events, True)
    if lam_t.ndim == 3 and lam_t.shape[0] == 1:
        lam_t = lam_t.squeeze(0)
    if lam_t.ndim == 3 and lam_t.shape[-1] == 1:
        lam_t = lam_t.squeeze(-1)
    if lam_t.ndim != 2:
        raise ValueError(f"Expected intensity shape (T,C), got {lam_t.shape}")

    if parts['baseline'].ndim == 3 and parts['baseline'].shape[0] == 1:
        parts['baseline'] = parts['baseline'].squeeze(0)
    if parts['excitation'].ndim == 3 and parts['excitation'].shape[0] == 1:
        parts['excitation'] = parts['excitation'].squeeze(0)
    lam = lam_t.cpu().numpy()
    base = parts['baseline'].cpu().numpy()
    exc = parts['excitation'].cpu().numpy()
    T, C = lam.shape

    xs, ys = cell_coords[:,0], cell_coords[:,1]
    if separate_base_ker:
        #setup figures
        event_disp1 = np.zeros(C, dtype=float)
        fig1, ax1 = plt.subplots(figsize=figsize)
        #norm_base_log, norm_base_lin = _log_norm_from_pos(base)
        #if norm_base_log is not None:
        #    sc_int1 = ax1.scatter(xs, ys, c=base[0], norm=norm_base_log, s=20, cmap="OrRd")
        #else:
        sc_int1 = ax1.scatter(xs, ys, c=base[0], s=20, cmap="OrRd")

        cbar1 = plt.colorbar(sc_int1, ax=ax1, label='baseline intensity')
        cbar1.formatter = ScalarFormatter(useMathText=True)
        cbar1.update_normal(sc_int1)
        sc_evt1 = ax1.scatter(xs, ys, s=0, c="#78e8ff", alpha=0.0)
        ax1.set_axis_off()

        event_disp2 = np.zeros(C, dtype=float)
        norm_exc_log, norm_exc_lin = _log_norm_from_pos(exc)
        fig2, ax2 = plt.subplots(figsize=figsize)
        if norm_exc_log is not None:
            sc_int2 = ax2.scatter(xs, ys, c=exc[0], norm=norm_exc_log, s=20, cmap="OrRd")
        else:
            sc_int2 = ax2.scatter(xs, ys, c=exc[0], norm=norm_exc_lin, s=20, cmap="OrRd")

        cbar2 = plt.colorbar(sc_int2, ax=ax2, label='excitation intensity', format=LogFormatterMathtext())
        if norm_exc_log is not None:
            sc_int2.cmap.set_bad('white')
            sc_int2.cmap.set_under('white')
        sc_evt2 = ax2.scatter(xs, ys, s=0, c="#78e8ff", alpha=0.0)
        ax2.set_axis_off()

        #update function for baseline
        def update_baseline(t):
            nonlocal event_disp1
            #update intensity colors
            arr = base[t] #1D of length C
            sc_int1.set_array(arr)
            #update decaying event size + alpha
            event_disp1 = event_disp1 * decay + ev[t].cpu().numpy()
            event_disp1 = np.clip(event_disp1, 0.0, 1.0)
            sc_evt1.set_sizes(100 * event_disp1)
            sc_evt1.set_alpha(event_disp1)
            ax1.set_title(f"Time step {t}")
            return sc_int1, sc_evt1
        def update_exc(t):
            nonlocal event_disp2
            #update intensity colors
            arr = exc[t] #1D of length C
            sc_int2.set_array(arr)
            #update decaying event size + alpha
            event_disp2 = event_disp2 * decay + ev[t].cpu().numpy()
            event_disp2 = np.clip(event_disp2, 0.0, 1.0)
            sc_evt2.set_sizes(100 * event_disp2)
            sc_evt2.set_alpha(event_disp2)
            ax2.set_title(f"Time step {t}")
            return sc_int2, sc_evt2
        
        #draw baseline
        frames = []
        for t in range(T):
            update_baseline(t) #redraw artists for frame t
            fig1.canvas.draw() #render the canvas
            #grab the RGB buffer from the figure
            w, h = fig1.canvas.get_width_height()
            buf = np.frombuffer(fig1.canvas.tostring_rgb(), dtype=np.uint8)
            img = buf.reshape(h, w, 3)
            frames.append(img)
        imageio.mimsave(output_path + "_base.gif", frames, fps=fps)
        print(f"Saved baseline GIF to {output_path}_base.gif")

        #draw excitation
        frames = []
        for t in range(T):
            update_exc(t) #redraw artists for frame t
            fig2.canvas.draw() #render the canvas
            #grab the RGB buffer from the figure
            w, h = fig2.canvas.get_width_height()
            buf = np.frombuffer(fig2.canvas.tostring_rgb(), dtype=np.uint8)
            img = buf.reshape(h, w, 3)
            frames.append(img)
        imageio.mimsave(output_path + "_exc.gif", frames, fps=fps)
        print(f"Saved excitation GIF to {output_path}_exc.gif")
    else:  
        event_disp = np.zeros(C, dtype=float)
        #setup figure
        fig, ax = plt.subplots(figsize=figsize)
        sc_int = ax.scatter(xs, ys, c=lam[0],
                        vmin=lam.min(), vmax=lam.max(), s=20, cmap="OrRd")
        sc_evt = ax.scatter(xs, ys, s=0, alpha=0.0)
        plt.colorbar(sc_int, ax=ax, label='lambda intensity')
        ax.set_axis_off()

        #update function
        def update(t):
            nonlocal event_disp
            #update intensity colors
            arr = lam[t] #1D of length C
            sc_int.set_array(arr)
            #update decaying event size + alpha
            event_disp = event_disp * decay + ev[t].cpu().numpy()
            event_disp = np.clip(event_disp, 0.0, 1.0)
            sc_evt.set_sizes(100 * event_disp)
            sc_evt.set_alpha(event_disp)
            ax.set_title(f"Time step {t}")
            return sc_int, sc_evt

        frames = []
        for t in range(T):
            update(t) #redraw artists for frame t
            fig.canvas.draw() #render the canvas
            #grab the RGB buffer from the figure
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = buf.reshape(h, w, 3)
            frames.append(img)

        imageio.mimsave(output_path, frames, fps=fps)
        print(f"Saved GIF to {output_path}")

