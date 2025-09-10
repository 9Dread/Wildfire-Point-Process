import torch
import torch.nn
#import pyogrio
from torch.utils.data import DataLoader
from DataProcessing import get_covs_tensor_list, get_point24deg_grid, grid_to_cell_coords, get_events_tensor_list, standardize_cov_tensors, WildfireDataset, WildfireInhibDataset
from Modeling import PoissonLinearIntensity
from Modeling import PoissonNeuralIntensity
from Modeling import HawkesDiffusionFlatbase
from Modeling import HawkesDiffusionLinbase
from Modeling import cross_validation


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#get data (NO PSPS/EPSS)
covars, varnames = get_covs_tensor_list()
events = get_events_tensor_list('daily')
covars = standardize_cov_tensors(covars)
dataset = WildfireDataset(covars, events)
loader = DataLoader(dataset, batch_size=1, shuffle=True) #KEEP BATCH SIZE = 1

#n_nan_covs = [torch.isnan(year).sum().item() for year in covars]
#print(f"covs has {n_nan_covs} NaNs")

#model, optimizer, scheduler prep
p = covars[0].shape[2] #number of covariates
grid_gdf = get_point24deg_grid()
cell_coords = grid_to_cell_coords(grid_gdf, True)
cell_coords = cell_coords/1000
model = HawkesDiffusionLinbase(num_covariates=p, cell_coords=cell_coords).to(device)

#Training loop. Has early stopping if change in loss is < 0.01 by default.
#train_model(model, optimizer, loader, 1000, device, scheduler)
results = cross_validation(model, "Adam", 1e-2, covars, events, device, ['Plateau'], save_path="Results/results.csv", model_name="Hawkes_Linbase_Noinhib")


#Run cross validation!!
model = PoissonLinearIntensity(num_covariates=p).to(device)
results = cross_validation(model, "Adam", 1e-2, covars, events, device, ['Plateau'], save_path="Results/results.csv", model_name="Poisson_GLM")

#Poisson MLP 20, 2
hidden_dim = 20
num_hidden_layers = 2
model = PoissonNeuralIntensity(num_covariates=p, hidden_dim = hidden_dim, num_hidden_layers=num_hidden_layers).to(device)
results = cross_validation(model, "Adam", 1e-3, covars, events, device, ['Plateau'], save_path="Results/results.csv", model_name="Poisson_Neural_20_2")

#Poisson MLP 20, 3
hidden_dim = 20
num_hidden_layers = 3
model = PoissonNeuralIntensity(num_covariates=p, hidden_dim = hidden_dim, num_hidden_layers=num_hidden_layers).to(device)
results = cross_validation(model, "Adam", 1e-3, covars, events, device, ['Plateau'], save_path="Results/results.csv", model_name="Poisson_Neural_20_3")
