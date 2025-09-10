import torch
import torch.nn
#import pyogrio
from torch.utils.data import DataLoader
from DataProcessing import get_covs_tensor_list, get_events_tensor_list, standardize_cov_tensors, WildfireInhibDataset, get_point24deg_grid, grid_to_cell_coords, get_inhib_tensor_list
from Modeling import HawkesDiffusionLinbasePSPSEPSSLin
from Modeling import train_model
from torch.optim.lr_scheduler import LambdaLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#get data
covars = get_covs_tensor_list(True)
events = get_events_tensor_list('daily', True)
#events = [events[3], events[4]] #2023, 2024
#mean/stddev standardization
covars = standardize_cov_tensors(covars)
inhibs = get_inhib_tensor_list(True)

#data loader
dataset = WildfireInhibDataset(covars, inhibs, events)
loader = DataLoader(dataset, batch_size=1, shuffle=True) #KEEP BATCH SIZE = 1

p = covars[0].shape[2] #number of covariates
grid_gdf = get_point24deg_grid(True)
cell_coords = grid_to_cell_coords(grid_gdf, True) / 1000
model = HawkesDiffusionLinbasePSPSEPSSLin(num_covariates=p, cell_coords=cell_coords)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

train_model(model, optimizer, loader, 300, device, print_iter=1)

save_path = "SavedModels/inhib_testing.pth"
torch.save(model.state_dict(), save_path)

model = model.eval()

from Visualization import animate_intensity
cell_coords_viz = grid_to_cell_coords(grid_gdf) #latlon, not metric
dataset = WildfireInhibDataset(covars, inhibs, events)
loader = DataLoader(dataset, batch_size=1, shuffle=False)
data_list = [[covs, inhibs, events] for covs, inhibs, events in loader]
covs = data_list[4][0].to("cuda")
inhs = data_list[4][1].to("cuda")
evs = data_list[4][2].to("cuda")
with torch.no_grad():
    lam, parts = model(covs, inhs, evs, return_parts=True)
inh = parts['inhib']
animate_intensity(inh, 'inh intensity', evs, cell_coords_viz, "Viz/test.gif", decay=0.9)
