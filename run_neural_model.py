import torch
import torch.nn
#import pyogrio
from torch.utils.data import DataLoader
import Functions
from Modeling import PoissonNeuralIntensity
from Modeling import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#get data
covars = Functions.get_covs_tensor_list(True)
events = Functions.get_events_tensor_list('daily', True)

#mean/stddev standardization
covars = Functions.standardize_cov_tensors(covars)

#data loader
dataset = Functions.WildfireDataset(covars, events)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

hidden_dim = 50
num_hidden_layers = 10
#model and optimizer
p = covars[0].shape[2] #number of covariates
model = PoissonNeuralIntensity(num_covariates=p, hidden_dim = hidden_dim, num_hidden_layers=num_hidden_layers).to(device)
model.apply(lambda m: (
    torch.nn.init.kaiming_uniform_(m.weight) if hasattr(m, "weight") else None,
    torch.nn.init.zeros_(m.bias) if hasattr(m, "bias")   else None
))
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.5)
optimizer2 = torch.optim.Adam(model.parameters(), lr = 1e-4)


#training loop
train_model(model, optimizer, loader, 100, device)


#For my run, we got avg NLL over all 5 years down to 3834.4760 (with 2 hidden layers of length 10), so the LL is
print("LL: " + str(-(3834.4760 * 5)))
#-19172.38, slight improvement from linear model, which got -20516.455
#for 30,3 we get -18709.4482421875

save_path = f"SavedModels/poisson_neural_{hidden_dim}_{num_hidden_layers}.pth"
torch.save(model.state_dict(), save_path)
