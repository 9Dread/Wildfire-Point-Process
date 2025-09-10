from Modeling import PoissonLinPSPSEPSSLin
from Modeling import PoissonLinPSPSEPSSLinMult
from Modeling import PoissonLinearIntensity
from Modeling import PoissonLinPSPSEPSSFlat
from Modeling import cross_validation
from Modeling import train_model
import torch
import torch.nn
#import pyogrio
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

from DataProcessing import get_covs_tensor_list, get_events_tensor_list, standardize_cov_tensors, get_inhib_tensor_list, WildfireInhibDataset, WildfireDataset, get_point24deg_grid, grid_to_cell_coords

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#get data
covars, varnames = get_covs_tensor_list()
events = get_events_tensor_list('daily')
#mean/stddev standardization
covars = standardize_cov_tensors(covars)
inhibs = get_inhib_tensor_list()
dataset = WildfireInhibDataset(covars, inhibs, events)
loader = DataLoader(dataset, batch_size=1, shuffle=True)
p = covars[0].shape[2] 

#Constant:
model = PoissonLinPSPSEPSSFlat(p)
model.psps
model.epss
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
#scheduler = ReduceLROnPlateau(optimizer, threshold_mode = 'abs', threshold = 0.001)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)
def lambda_update(epoch):
    if 10 ** (epoch // 400) < 100:
        return 10 ** (epoch // 400)
    else:
        return 40
scheduler = LambdaLR(optimizer, lr_lambda = lambda_update)
train_model(model, optimizer, loader, 2000, device, scheduler)
model.psps
model.epss

#Additive linear functions:
model = PoissonLinPSPSEPSSLin(p, [0,1,9,10,15], inhib_init_bias= -10.)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)
def lambda_update(epoch):
    if 10 ** (epoch // 400) < 100:
        return 10 ** (epoch // 400)
    else:
        return 40
scheduler = LambdaLR(optimizer, lr_lambda = lambda_update)
train_model(model, optimizer, loader, 2000, device, scheduler)
model.psps_linear.weight.detach().cpu().numpy().reshape(-1)
model.epss_linear.weight.detach().cpu().numpy().reshape(-1)

#Now try multiplicative effect:
#(doesn't trigger early stopping, prolly have to change lr scheduler)
model2 = PoissonLinPSPSEPSSLinMult(p)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=1e-7)
def lambda_update2(epoch):
    #we are starting at lr=1e-7; multiply by 10 every 100 epochs until 1e-4
    if 10 ** (epoch // 100) < 100:
        return 10 ** (epoch // 100)
    else:
        return 10 ** 3
scheduler2 = LambdaLR(optimizer2, lr_lambda = lambda_update2)
train_model(model2, optimizer2, loader, 3000, device, scheduler2)
#With only features [0,1,9,10,15]: avg nll -4126.919 (basically same as poisson glm? actually slightly worse)
#With all features: -4122.285302734375 (almost exactly the same as poisson linear!)

#What if we remove 2020, which has no psps/epss effects?
covars.remove(covars[0])
events.remove(events[0])
inhibs.remove(inhibs[0])
dataset = WildfireInhibDataset(covars, inhibs, events)
loader = DataLoader(dataset, batch_size=1, shuffle=True)
model2 = PoissonLinPSPSEPSSLinMult(p)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=1e-7)
def lambda_update2(epoch):
    #we are starting at lr=1e-7; multiply by 10 every 100 epochs until 1e-4
    if 10 ** (epoch // 100) < 100:
        return 10 ** (epoch // 100)
    else:
        return 10 ** 3
scheduler2 = LambdaLR(optimizer2, lr_lambda = lambda_update2)
train_model(model2, optimizer2, loader, 3000, device, scheduler2)
#nll -4039.7166748046875, compared to -4037.7094116210938 from regular Poisson

#normal Poisson glm for comparison
dataset2 = WildfireDataset(covars, events)
loader2 = DataLoader(dataset2, batch_size=1, shuffle=True)
model3 = PoissonLinearIntensity(p)
#optimizer3 = torch.optim.SGD(model3.parameters(), lr = 1e-7)
optimizer3 = torch.optim.Adam(model3.parameters(), lr=1e-2)
def lambda_update3(epoch):
    #we are starting at lr=1e-7; multiply by 10 every 100 epochs until 1e-4
    if 10 ** (epoch // 100) < 100:
        return 10 ** (epoch // 100)
    else:
        return 10 ** 3
#scheduler3 = LambdaLR(optimizer3, lr_lambda = lambda_update3)
scheduler3 = ReduceLROnPlateau(optimizer3, 'min', threshold=1e-2, threshold_mode = 'abs')
train_model(model3, optimizer3, loader2, 1000, device, scheduler3)
#The normal poisson linear is doing better than additive effects model (-4121.70).

#Test vis
from Visualization import animate_intensity
grid_gdf = get_point24deg_grid()
cell_coords_viz = grid_to_cell_coords(grid_gdf) #latlon, not metric
dataset = WildfireInhibDataset(covars, inhibs, events)
loader = DataLoader(dataset, batch_size=1, shuffle=False)
data_list = [[covs, inhibs, events] for covs, inhibs, events in loader]
covs = data_list[4][0].to("cuda")
inhs = data_list[4][1].to("cuda")
evs = data_list[4][2].to("cuda")
with torch.no_grad():
    lam, parts = model(covs, inhs, evs, return_parts=True)
inh = parts['inhib']
animate_intensity(inh, 'inh intensity', evs, cell_coords_viz, "Viz/test_inh_additive_initnegbias2.gif", scale="lin", decay=0.9)
animate_intensity(lam, 'lam intensity', evs, cell_coords_viz, "Viz/testlam.gif", scale="lin", decay=0.9)

print("Additive inhib intensity max effect for 2024: ", inh.max())
print("Additive inhib intensity min effect for 2024: ", inh.min())

#multiplicative model:
with torch.no_grad():
    lam, parts = model2(covs, inhs, evs, return_parts=True)
inh = parts['inhib']
print("Multiplicative inhib intensity max effect for 2024: ", inh.max())
print("Multiplicative inhib intensity min effect for 2024: ", inh.min())
animate_intensity(inh, 'inh intensity', evs, cell_coords_viz, "Viz/test_mult.gif", scale="lin", decay=0.9)
animate_intensity(lam, 'lam intensity', evs, cell_coords_viz, "Viz/testlam_mult.gif", scale="lin", decay=0.9)



