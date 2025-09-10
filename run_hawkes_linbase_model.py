import torch
import torch.nn
#import pyogrio
from torch.utils.data import DataLoader
from DataProcessing import get_covs_tensor_list, get_events_tensor_list, standardize_cov_tensors, WildfireDataset, get_point24deg_grid, grid_to_cell_coords

from Modeling import HawkesDiffusionLinbase
from Modeling import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#get data
covars, _ = get_covs_tensor_list(True, True)
events = get_events_tensor_list('daily', True)
events = [events[3], events[4]] #only 2023 and 2024 for now

#mean/stddev standardization
covars = standardize_cov_tensors(covars)

#data loader
dataset = WildfireDataset(covars, events)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

#n_nan_covs = [torch.isnan(year).sum().item() for year in covars]
#print(f"covs has {n_nan_covs} NaNs")

#model and optimizer
p = covars[0].shape[2] #number of covariates
grid_gdf = get_point24deg_grid(True)
cell_coords = grid_to_cell_coords(grid_gdf, True)
cell_coords = cell_coords/1000 #convert to km units, helps with gradients

model = HawkesDiffusionLinbase(num_covariates=p, cell_coords=cell_coords).to(device)
model.apply(lambda m: (
    torch.nn.init.kaiming_uniform_(m.weight) if hasattr(m, "weight") else None,
    torch.nn.init.zeros_(m.bias)           if hasattr(m, "bias")   else None
))
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4) #1e-8 lr seems to work for this model initially but has to be adjusted to converge faster; maybe use adam
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

#training loop
train_model(model, optimizer, loader, 25, device, print_iter=1)
#3961.2657470703125.
#over the training set: -7922.531494140625

save_path = "SavedModels/hawkes_stddif_linbase_combinedcovs.pth"
torch.save(model.state_dict(), save_path)

#MarcosCovs:
covars = get_covs_tensor_list(True)
events = get_events_tensor_list('daily', True)
covars = standardize_cov_tensors(covars)
dataset = WildfireDataset(covars, events)
loader = DataLoader(dataset, batch_size=1, shuffle=True)
p = covars[0].shape[2] #number of covariates
grid_gdf = get_point24deg_grid(True)
cell_coords = grid_to_cell_coords(grid_gdf, True)
cell_coords = cell_coords/1000
model = HawkesDiffusionLinbase(num_covariates=p, cell_coords=cell_coords).to(device)
model.apply(lambda m: (
    torch.nn.init.kaiming_uniform_(m.weight) if hasattr(m, "weight") else None,
    torch.nn.init.zeros_(m.bias)           if hasattr(m, "bias")   else None
))
optimizer = torch.optim.SGD(model.parameters(), lr=1e-8) #could also use lr scheduler here, x10 every 10 (after 1e-5 slow to 20 maybe) up to 1e-4
train_model(model, optimizer, loader, 10, device, print_iter=1)
save_path = "SavedModels/hawkes_stddif_linbase_marcoscovs.pth"
torch.save(model.state_dict(), save_path)
#-4168.789599609375.
#over the training set: -20843.947998046875