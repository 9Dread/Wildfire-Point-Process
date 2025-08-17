import Functions
import torch
import torch.nn
from Modeling import PoissonNeuralIntensity
from torch.utils.data import DataLoader

hidden_dim = 100
num_hidden_layers = 10

#get data
covars = Functions.get_covs_tensor_list(True)
events = Functions.get_events_tensor_list('daily', True)
#mean/stddev standardization
covars = Functions.standardize_cov_tensors(covars)
#data loader
dataset = Functions.WildfireDataset(covars, events)
loader = DataLoader(dataset, batch_size=1, shuffle=False)
data_list = [[covs, event_mask] for covs, event_mask in loader]

#Load the pretrained model
p = covars[0].shape[2] #number of covariates
model = PoissonNeuralIntensity(p, hidden_dim, num_hidden_layers) #trained with 20 layer width, 2 hidden layers
state_dict = torch.load(f"SavedModels/poisson_neural_{hidden_dim}_{num_hidden_layers}.pth", map_location="cpu")  
model.load_state_dict(state_dict)
model.eval() #eval mode

#Lets do 2024 first
covs = data_list[4][0]
events = data_list[4][1]
#get centroids
grid_gdf = Functions.get_point24deg_grid(True)
cell_centroids = Functions.grid_to_cell_coords(grid_gdf)

Functions.animate_poisson_intensity(model, covs, events, cell_centroids, f"Viz/wildfire_intensity_2024_mlp_{hidden_dim}_{num_hidden_layers}.gif", decay=0.9)
