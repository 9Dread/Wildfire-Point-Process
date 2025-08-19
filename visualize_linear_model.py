import Functions
import torch
import torch.nn
import numpy as np
import imageio
from Modeling import PoissonLinearIntensity
from torch.utils.data import DataLoader

#MarcosCovs visualization:
#get data
covars = Functions.get_covs_tensor_list(True)
events = Functions.get_events_tensor_list('daily', True)
#mean/stddev standardization
covars = Functions.standardize_cov_tensors(covars)
#data loader
dataset = Functions.WildfireDataset(covars, events)
loader = DataLoader(dataset, batch_size=1, shuffle=False)
data_list = [[covs, events] for covs, events in loader]

#Load the pretrained linear model
p = covars[0].shape[2] #number of covariates
model = PoissonLinearIntensity(num_covariates=p)
state_dict = torch.load("SavedModels/poisson_glm_marcoscovs.pth", map_location="cpu")  
model.load_state_dict(state_dict)
model.eval() #eval mode

weights = model.linear.weight.detach().cpu().numpy().reshape(-1)  # shape (p,)
bias = model.linear.bias.detach().cpu().item()
print("Intercept (B):", bias)
for name, w in zip(["NDVI", "EVI", "fm100", "fm1000", "elevation"], weights):
    print(f"B for {name:10s} =", w)

#Lets do 2024 first
covs = data_list[4][0]
event = data_list[4][1]
#get centroids
grid_gdf = Functions.get_point24deg_grid(True)
cell_centroids = Functions.grid_to_cell_coords(grid_gdf)

Functions.animate_poisson_intensity(model, covs, event, cell_centroids, "Viz/wildfire_intensity_2024_glm_marcoscovs.gif", decay=0.9)


#CombinedCovs:
covars, var_names = Functions.get_covs_tensor_list(True, True)
events = Functions.get_events_tensor_list('daily', True)
events = [events[3], events[4]] #2023 and 2024
covars = Functions.standardize_cov_tensors(covars)
dataset = Functions.WildfireDataset(covars, events)
loader = DataLoader(dataset, batch_size=1, shuffle=False)
data_list = [[covs, events] for covs, events in loader]

p = covars[0].shape[2] #number of covariates
model = PoissonLinearIntensity(num_covariates=p)
state_dict = torch.load("SavedModels/poisson_glm_combinedcovs.pth", map_location="cpu")  
model.load_state_dict(state_dict)
model.eval() #eval mode

weights = model.linear.weight.detach().cpu().numpy().reshape(-1)  # shape (p,)
bias = model.linear.bias.detach().cpu().item()
print("Intercept (B):", bias)
for name, w in zip(var_names, weights):
    print(f"B for {name:30s} =", w)

covs = data_list[1][0]
event = data_list[1][1]
grid_gdf = Functions.get_point24deg_grid(True)
cell_centroids = Functions.grid_to_cell_coords(grid_gdf)
Functions.animate_poisson_intensity(model, covs, event, cell_centroids, "Viz/wildfire_intensity_2024_glm_combinedcovs.gif", decay=0.9)
