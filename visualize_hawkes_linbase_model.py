from Visualization import animate_intensity
import torch
from DataProcessing import get_covs_tensor_list, get_events_tensor_list, standardize_cov_tensors, WildfireDataset, get_point24deg_grid, grid_to_cell_coords
from Modeling import HawkesDiffusionLinbase
from torch.utils.data import DataLoader

#get data
covars, var_names = get_covs_tensor_list(True, True)
events = get_events_tensor_list('daily', True)
events = [events[3], events[4]] #2023 and 2024 for now

#mean/stddev standardization
covars = standardize_cov_tensors(covars)

#data loader
dataset = WildfireDataset(covars, events)
loader = DataLoader(dataset, batch_size=1, shuffle=False)
data_list = [[covs, events] for covs, events in loader]

#Load the pretrained model
p = covars[0].shape[2] #number of covariates
grid_gdf = get_point24deg_grid(True)
cell_coords_model = grid_to_cell_coords(grid_gdf, True) / 1000
cell_coords_viz = grid_to_cell_coords(grid_gdf) #latlon, not metric

model = HawkesDiffusionLinbase(num_covariates=p, cell_coords=cell_coords_model)

#load model trained from before
state_dict = torch.load("SavedModels/hawkes_stddif_linbase_combinedcovs.pth", map_location="cpu")  
model.load_state_dict(state_dict)
model.eval() #eval mode

weights = model.linear.weight.detach().cpu().numpy().reshape(-1)  #shape (p,)
bias = model.linear.bias.detach().cpu().item()
C = model.C.detach().cpu().item()
beta = model.beta.detach().cpu().item()
sigma = model.sigma.detach().cpu().item()
print("Intercept (B):", bias)
for name, w in zip(var_names, weights):
    print(f"B for {name:30s} =", w)
print("C (excitation kernel): ", C)
print("Beta (excitation kernel): ", beta)
print("Sigma (excitation kernel; KM units): ", sigma)

#Lets do 2024 first
covs = data_list[1][0]
event = data_list[1][1]
with torch.no_grad():
    lam, parts = model(covs, event, return_parts=True)
base = parts['baseline']
exc = parts['excitation']
animate_intensity(base, 'baseline intensity', event, cell_coords_viz, "Viz/wildfire_intensity_2024_hawksstddif_combinedcovs_base.gif", decay=0.9)
animate_intensity(exc, 'excitation intensity', event, cell_coords_viz, "Viz/wildfire_intensity_2024_hawksstddif_combinedcovs_exc.gif", cmap='magma', scale='log', decay=0.9)
animate_intensity(lam, 'lambda intensity', event, cell_coords_viz, "Viz/wildfire_intensity_2024_hawksstddif_combinedcovs_lam.gif", scale='log', decay=0.9)


#marcoscovs:

covars = get_covs_tensor_list(True)
events = get_events_tensor_list('daily', True)
#mean/stddev standardization
covars = standardize_cov_tensors(covars)
dataset = WildfireDataset(covars, events)
loader = DataLoader(dataset, batch_size=1, shuffle=False)
data_list = [[covs, events] for covs, events in loader]
p = covars[0].shape[2] #number of covariates
grid_gdf = get_point24deg_grid(True)
cell_coords_model = grid_to_cell_coords(grid_gdf, True) / 1000
cell_coords_viz = grid_to_cell_coords(grid_gdf)
model = HawkesDiffusionLinbase(num_covariates=p, cell_coords=cell_coords_model)
state_dict = torch.load("SavedModels/hawkes_stddif_linbase_marcoscovs.pth", map_location="cpu")  
model.load_state_dict(state_dict)
model.eval() #eval mode

weights = model.linear.weight.detach().cpu().numpy().reshape(-1)  #shape (p,)
bias = model.linear.bias.detach().cpu().item()
C = model.C.detach().cpu().item()
beta = model.beta.detach().cpu().item()
sigma = model.sigma.detach().cpu().item()
print("Intercept (B):", bias)
for name, w in zip(["NDVI", "EVI", "fm100", "fm1000", "elevation"], weights):
    print(f"B for {name:10s} =", w)
print("C (excitation kernel): ", C)
print("Beta (excitation kernel): ", beta)
print("Sigma (excitation kernel; KM units): ", sigma)

covs = data_list[4][0]
event = data_list[4][1]
with torch.no_grad():
    lam, parts = model(covs, event, return_parts=True)
base = parts['baseline']
exc = parts['excitation']
animate_intensity(base, 'baseline intensity', event, cell_coords_viz, "Viz/wildfire_intensity_2024_hawksstddif_marcoscovs_base.gif", decay=0.9)
animate_intensity(exc, 'excitation intensity', event, cell_coords_viz, "Viz/wildfire_intensity_2024_hawksstddif_marcoscovs_exc.gif", cmap='magma', scale='log', decay=0.9)
