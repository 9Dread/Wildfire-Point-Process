import Functions
import torch
import torch.nn
import numpy as np
import imageio
from Models import PoissonLinearIntensity
from torch.utils.data import DataLoader

#get data
covars = Functions.get_covs_tensor_list(True)
events = Functions.get_events_tensor_list('daily', True)
#mean/stddev standardization
covars = Functions.standardize_cov_tensors(covars)
#data loader
dataset = Functions.WildfireDataset(covars, events)
loader = DataLoader(dataset, batch_size=1, shuffle=False)
data_list = [[covs, event_mask] for covs, event_mask in loader]

#Load the pretrained linear model
p = covars[0].shape[2] #number of covariates
model = PoissonLinearIntensity(num_covariates=p)
state_dict = torch.load("SavedModels/poisson_glm.pth", map_location="cpu")  
model.load_state_dict(state_dict)
model.eval() #eval mode

weights = model.linear.weight.detach().cpu().numpy().reshape(-1)  # shape (p,)
bias = model.linear.bias.detach().cpu().item()
print("Intercept (B):", bias)
for name, w in zip(["NDVI", "EVI", "fm100", "fm1000", "elevation"], weights):
    print(f"B for {name:10s} =", w)

#Lets do 2024 first
covs = data_list[4][0]
events = data_list[4][1]
#get centroids
grid_gdf = Functions.get_point24deg_grid(True)
cell_centroids = Functions.grid_to_cell_coords(grid_gdf)

Functions.animate_poisson_intensity(model, covs, events, cell_centroids, "Viz/wildfire_intensity_2024_glm.gif", decay=0.9)
