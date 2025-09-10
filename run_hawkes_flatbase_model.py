import torch
import torch.nn
#import pyogrio
from torch.utils.data import DataLoader
from DataProcessing import get_covs_tensor_list, get_events_tensor_list, WildfireDataset, get_point24deg_grid, grid_to_cell_coords
from Modeling import HawkesDiffusionFlatbase
from Modeling import train_model
from torch.optim.lr_scheduler import LambdaLR


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#get data
covars = get_covs_tensor_list(True)
events = get_events_tensor_list('daily', True)
dataset = WildfireDataset(covars, events) #This model doesn't actually use covars, only references spacetime shape, so we don't have to standardize
loader = DataLoader(dataset, batch_size=1, shuffle=True) 

#model and optimizer
grid_gdf = get_point24deg_grid(True)
cell_coords = grid_to_cell_coords(grid_gdf, True)
cell_coords = cell_coords/1000 #convert to km units, helps with gradients

model = HawkesDiffusionFlatbase(cell_coords=cell_coords).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6) #1e-6, x10 every 20 until 1e-5
def lambda_update(epoch):
    if 10 ** (epoch // 20) < 10:
        return 10 ** (epoch // 20)
    else:
        return 10 ** 1
scheduler = LambdaLR(optimizer, lambda_update)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

#training loop
train_model(model, optimizer, loader, 250, device, scheduler, print_iter=1)
#3961.2657470703125.
#over the training set: -7922.531494140625

save_path = "SavedModels/hawkes_stddif_linbase_combinedcovs.pth"
torch.save(model.state_dict(), save_path)