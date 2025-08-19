import torch
import torch.nn
#import pyogrio
from torch.utils.data import DataLoader
import Functions
from Modeling import PoissonLinearIntensity
from Modeling import train_model
from torch.optim.lr_scheduler import LambdaLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#get data
covars = Functions.get_covs_tensor_list(True)
events = Functions.get_events_tensor_list('daily', True)
#events = [events[3], events[4]] #2023, 2024
#mean/stddev standardization
covars = Functions.standardize_cov_tensors(covars)

#data loader
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
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-7)
def lambda_update(epoch):
    #we are starting at lr=1e-7; multiply by 10 every 100 epochs until 1e-4
    if 10 ** (epoch // 100) < 100:
        return 10 ** (epoch // 100)
    else:
        return 10 ** 3
scheduler = LambdaLR(optimizer, lr_lambda = lambda_update)

#Training loop. Has early stopping if change in loss is < 0.01 by default.
train_model(model, optimizer, loader, 1000, device, scheduler)
save_path = "SavedModels/poisson_glm_marcoscovs.pth"
torch.save(model.state_dict(), save_path)
#results = cross_validation(model, 1e-4, covars, events, device, save_results=False)
#MarcosCovs: avg nll to 4177.556103515625, so the LL is
#=-20887.780517578125, which is a slight improvement from the homogenous model.

#CombinedCovs:
covars, _ = Functions.get_covs_tensor_list(True, True)
events = Functions.get_events_tensor_list('daily', True)
events = [events[3], events[4]] #2023 and 2024
covars = Functions.standardize_cov_tensors(covars)
dataset = Functions.WildfireDataset(covars, events)
loader = DataLoader(dataset, batch_size=1, shuffle=True) #KEEP BATCH SIZE = 1
p = covars[0].shape[2] #number of covariates
model = PoissonLinearIntensity(num_covariates=p).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-10)
def lambda_update(epoch):
    #we are starting at lr=1e-10; multiply by 10 every 150 epochs until 1e-6
    if 10 ** (epoch // 150) < 10000:
        return 10 ** (epoch // 150)
    else:
        return 10 ** 4
scheduler = LambdaLR(optimizer, lr_lambda = lambda_update)
#Training loop. Has early stopping if change in loss is < 0.01 by default.
train_model(model, optimizer, loader, 1500, device, scheduler)
#avg nll: 4049.9710693359375
#total ll: -8099.942138671875
save_path = "SavedModels/poisson_glm_combinedcovs.pth"
torch.save(model.state_dict(), save_path)

