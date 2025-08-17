import torch
import torch.nn
#import pyogrio
from torch.utils.data import DataLoader
import Functions
from Modeling import PoissonLinearIntensity
from Modeling import train_model
from Modeling import cross_validation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#get data
covars = Functions.get_covs_tensor_list(True)
events = Functions.get_events_tensor_list('daily', True)

#mean/stddev standardization
covars = Functions.standardize_cov_tensors(covars)

#data loader
dataset = Functions.WildfireDataset(covars, events)
loader = DataLoader(dataset, batch_size=1, shuffle=True) #KEEP BATCH SIZE = 1

#n_nan_covs = [torch.isnan(year).sum().item() for year in covars]
#print(f"covs has {n_nan_covs} NaNs")

#model and optimizer
p = covars[0].shape[2] #number of covariates
model = PoissonLinearIntensity(num_covariates=p).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#Training loop. Has early stopping if change in loss is < 0.01 by default.
train_model(model, optimizer, loader, 20000, device)

results = cross_validation(model, 1e-4, covars, events, device, save_results=False)

#For my run, we got avg NLL over all 5 years down to 4103.2910, so the LL is
print("LL: " + str(-(4103.2910 * 5)))
#=-20516.455, which is a slight improvement from the homogenous model.

save_path = "SavedModels/poisson_glm.pth"
torch.save(model.state_dict(), save_path)
