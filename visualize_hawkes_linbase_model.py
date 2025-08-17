import Functions
import torch
from Modeling import HawkesDiffusionLinbase
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
grid_gdf = Functions.get_point24deg_grid(True)
cell_coords_model = Functions.grid_to_cell_coords(grid_gdf, True) / 1000
cell_coords_viz = Functions.grid_to_cell_coords(grid_gdf)

model = HawkesDiffusionLinbase(num_covariates=p, cell_coords=cell_coords_model)

#load model trained from before
state_dict = torch.load("SavedModels/hawkes_stddif_linbase.pth", map_location="cpu")  
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
print(C)
print(beta)
print(sigma)

#Lets do 2024 first
covs = data_list[4][0]
events = data_list[4][1]

Functions.animate_hawkes_intensity(model, covs, events, cell_coords_viz, "Viz/wildfire_intensity_2024_hawksstddif", decay=0.9,separate_base_ker=True)


with torch.no_grad():
    lam, parts = model(covs.squeeze(0), events.squeeze(0), return_parts=True)
base, exc = parts['baseline'], parts['excitation']
print(base.max())
print(exc.max())
print(lam.max())
