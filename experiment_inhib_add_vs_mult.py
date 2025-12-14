from Modeling import PoissonLinPSPSEPSSLin
from Modeling import PoissonLinPSPSEPSSLinMult
from Modeling import PoissonLinearIntensity
from Modeling import PoissonLinPSPSEPSSFlat
from Modeling import cross_validation
from Modeling import train_model
import torch
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
covars = standardize_cov_tensors(covars, True)
inhibs = get_inhib_tensor_list()
dataset = WildfireInhibDataset(covars, inhibs, events)
loader = DataLoader(dataset, batch_size=1, shuffle=True)
p = covars[0].shape[2] 

#Constant:
model = PoissonLinPSPSEPSSFlat(p, init_epss=-5., init_psps=-5.)
model.psps
model.epss
#optimizer = torch.optim.Adam([{'params': model.linear.parameters(), 'lr': 1e-2},
#    {'params': model.raw_psps}, {'params': model.raw_epss}], lr=10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
scheduler = ReduceLROnPlateau(optimizer, threshold_mode = 'abs', threshold = 0.01)
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)
#def lambda_update(epoch):
#    if 10 ** (epoch // 400) < 100:
#        return 10 ** (epoch // 400)
#    else:
#        return 40
#scheduler = LambdaLR(optimizer, lr_lambda = lambda_update)
train_model(model, optimizer, loader, 2000, device, scheduler)
model.psps
model.epss
#ll -4038.4524 trained on 2021-24

#Test vis
from Visualization import animate_intensity
grid_gdf = get_point24deg_grid()
cell_coords_viz = grid_to_cell_coords(grid_gdf) #latlon, not metric
loader_viz = DataLoader(dataset, batch_size=1, shuffle=False)
data_list = [[covs, inhibs, events] for covs, inhibs, events in loader_viz]
covs = data_list[4][0].to("cuda")
inhs = data_list[4][1].to("cuda")
evs = data_list[4][2].to("cuda")
with torch.no_grad():
    lam, parts = model(covs, inhs, evs, return_parts=True)
inh = parts['inhib']
animate_intensity(inh, 'inhibitory effect (intensity decrease)', evs, cell_coords_viz, "Viz/test_inh_flatmult.gif", scale="lin", decay=0.9)
animate_intensity(lam, 'lam intensity', evs, cell_coords_viz, "Viz/test_lam_inhflatmult.gif", scale="lin", decay=0.9)


#Additive linear functions:
model = PoissonLinPSPSEPSSLin(p, [0,1,9,10,15], inhib_init_bias= -10.)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
scheduler = ReduceLROnPlateau(optimizer, threshold_mode = 'abs', threshold = 0.01)
train_model(model, optimizer, loader, 2000, device, scheduler)
#ll 4119.4443
model.psps_linear.weight.detach().cpu().numpy().reshape(-1)
model.epss_linear.weight.detach().cpu().numpy().reshape(-1)
with torch.no_grad():
    lam, parts = model(covs, inhs, evs, return_parts=True)
inh = parts['inhib']
animate_intensity(inh, 'inhibitory effect (intensity decrease)', evs, cell_coords_viz, "Viz/test_inh_linadd.gif", scale="lin", decay=0.9)
animate_intensity(lam, 'lam intensity', evs, cell_coords_viz, "Viz/test_lam_inhlinadd.gif", scale="lin", decay=0.9)

#Now try multiplicative effect:
model2 = PoissonLinPSPSEPSSLinMult(p)
#optimizer2 = torch.optim.SGD(model2.parameters(), lr=1e-7)
#def lambda_update2(epoch):
#    #we are starting at lr=1e-7; multiply by 10 every 100 epochs until 1e-4
#    if 10 ** (epoch // 100) < 100:
#        return 10 ** (epoch // 100)
#    else:
#        return 10 ** 3
#scheduler2 = LambdaLR(optimizer2, lr_lambda = lambda_update2)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-2)
scheduler2 = ReduceLROnPlateau(optimizer2, threshold_mode = 'abs', threshold = 0.01)
train_model(model2, optimizer2, loader, 3000, device, scheduler2)
with torch.no_grad():
    lam, parts = model2(covs, inhs, evs, return_parts=True)
inh = parts['inhib']
animate_intensity(inh, 'inhibitory effect (intensity decrease)', evs, cell_coords_viz, "Viz/test_inh_linmult.gif", scale="lin", decay=0.9)
animate_intensity(lam, 'lam intensity', evs, cell_coords_viz, "Viz/test_lam_inhlinmult.gif", scale="lin", decay=0.9)
#With only features [0,1,9,10,15]: avg nll 4118.982080078125 (basically same as poisson glm?)
#With all features: 4122.285302734375 (almost exactly the same as poisson linear!)

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
#(-4121.70).

#Cross validation
model = PoissonLinPSPSEPSSFlat(p, init_epss=0., init_psps=0.)
cross_validation(model, "Adam", 1e-1, covars, events, device, inhs=inhibs, scheduler= ['Plateau'], save_path = "Results/results_inhib_experiment.csv", model_name = "inhib_flatmult")

model = PoissonLinPSPSEPSSLin(p, [0,1,9,10,15], inhib_init_bias= -10.)
cross_validation(model, "Adam", 1e-1, covars, events, device, inhs=inhibs, scheduler= ['Plateau'], save_path = "Results/results_inhib_experiment.csv", model_name = "inhib_linadd")

model = PoissonLinPSPSEPSSLinMult(p, [0,1,9,10,15])
cross_validation(model, "Adam", 1e-1, covars, events, device, inhs=inhibs, scheduler= ['Plateau'], save_path = "Results/results_inhib_experiment.csv", model_name = "inhib_linmult")

model = PoissonLinearIntensity(p)
cross_validation(model, "Adam", 1e-2, covars, events, device, scheduler= ['Plateau'], save_path = "Results/results_std_experiment.csv", model_name = "Poisson_GLM")

#Cross validation excluding 2020 during training 
covars.remove(covars[0])
events.remove(events[0])
inhibs.remove(inhibs[0])

model = PoissonLinPSPSEPSSFlat(p, init_epss=0., init_psps=0.)
cross_validation(model, "Adam", 1e-1, covars, events, device, inhs=inhibs, scheduler= ['Plateau'], save_path = "Results/results_inhib_experiment_exclude20.csv", model_name = "inhib_flatmult")

model = PoissonLinPSPSEPSSLin(p, [0,1,9,10,15], inhib_init_bias= -10.)
cross_validation(model, "Adam", 1e-1, covars, events, device, inhs=inhibs, scheduler= ['Plateau'], save_path = "Results/results_inhib_experiment_exclude20.csv", model_name = "inhib_linadd")

model = PoissonLinPSPSEPSSLinMult(p, [0,1,9,10,15])
cross_validation(model, "Adam", 1e-1, covars, events, device, inhs=inhibs, scheduler= ['Plateau'], save_path = "Results/results_inhib_experiment_exclude20.csv", model_name = "inhib_linmult")

model = PoissonLinearIntensity(p)
cross_validation(model, "Adam", 1e-2, covars, events, device, scheduler= ['Plateau'], save_path = "Results/results_inhib_experiment_exclude20.csv", model_name = "Poisson_GLM")



