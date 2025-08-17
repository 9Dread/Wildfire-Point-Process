import torch
import torch.nn
#import pyogrio
from torch.utils.data import DataLoader
import Functions
from Modeling import HawkesDiffusionLinbase
from Modeling import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#get data
covars = Functions.get_covs_tensor_list(True, True)
events = Functions.get_events_tensor_list('daily', True)
events = [events[3], events[4]] #only 2023 and 2024 for now

#mean/stddev standardization
covars = Functions.standardize_cov_tensors(covars)

#data loader
dataset = Functions.WildfireDataset(covars, events)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

#n_nan_covs = [torch.isnan(year).sum().item() for year in covars]
#print(f"covs has {n_nan_covs} NaNs")

#model and optimizer
p = covars[0].shape[2] #number of covariates
grid_gdf = Functions.get_point24deg_grid(True)
cell_coords = Functions.grid_to_cell_coords(grid_gdf, True)
cell_coords = cell_coords/1000 #convert to km units, helps with gradients

model = HawkesDiffusionLinbase(num_covariates=p, cell_coords=cell_coords).to(device)
model.apply(lambda m: (
    torch.nn.init.kaiming_uniform_(m.weight) if hasattr(m, "weight") else None,
    torch.nn.init.zeros_(m.bias)           if hasattr(m, "bias")   else None
))
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) #1e-8 lr seems to work for this model initially but has to be adjusted to converge faster; maybe use adam
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

#training loop
train_model(model, optimizer, loader, 200, device, print_iter=1)

#For my run, we got avg NLL over all 5 years down to 4149.953662109375 so the LL is
#=-20749.768310546875, which is worse than linear model! Interesting!

save_path = "SavedModels/hawkes_stddif_linbase.pth"
torch.save(model.state_dict(), save_path)
