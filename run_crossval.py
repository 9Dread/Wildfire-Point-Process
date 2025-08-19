import torch
import torch.nn
#import pyogrio
from torch.utils.data import DataLoader
import Functions
from Modeling import PoissonLinearIntensity
from Modeling import cross_validation
from Modeling import PoissonNeuralIntensity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#get data
covars = Functions.get_covs_tensor_list(True)
events = Functions.get_events_tensor_list('daily', True)
covars = Functions.standardize_cov_tensors(covars)
dataset = Functions.WildfireDataset(covars, events)
loader = DataLoader(dataset, batch_size=1, shuffle=True) #KEEP BATCH SIZE = 1

#n_nan_covs = [torch.isnan(year).sum().item() for year in covars]
#print(f"covs has {n_nan_covs} NaNs")

#model, optimizer, scheduler
p = covars[0].shape[2] #number of covariates
model = PoissonLinearIntensity(num_covariates=p).to(device)
model.apply(lambda m: (
    torch.nn.init.kaiming_uniform_(m.weight) if hasattr(m, "weight") else None,
    torch.nn.init.zeros_(m.bias)           if hasattr(m, "bias")   else None
))
def lambda_update(epoch):
    if 10 ** (epoch // 100) < 100:
        return 10 ** (epoch // 100)
    else:
        return 10 ** 3

#Training loop. Has early stopping if change in loss is < 0.01 by default.
#train_model(model, optimizer, loader, 1000, device, scheduler)

#Run cross validation!!
results = cross_validation(model, "SGD", 1e-7, covars, events, device, lambda_update, save_path="Results/results.csv", model_name="Poisson_GLM_Marcoscovs")

#MLP 20, 2
hidden_dim = 20
num_hidden_layers = 2
model = PoissonNeuralIntensity(num_covariates=p, hidden_dim = hidden_dim, num_hidden_layers=num_hidden_layers).to(device)
model.apply(lambda m: (
    torch.nn.init.kaiming_uniform_(m.weight) if hasattr(m, "weight") else None,
    torch.nn.init.zeros_(m.bias) if hasattr(m, "bias")   else None
))
def lambda_update(epoch):
    #we are starting at lr=1e-9; multiply by 10 every 100 epochs until 1e-5
    if 10 ** (epoch // 200) < 100000:
        return 10 ** (epoch // 200)
    else:
        return 10 ** 4
results = cross_validation(model, "SGD", 1e-9, covars, events, device, lambda_update, save_path="Results/results.csv", model_name="Poisson_Neural_20_2_Marcoscovs")
