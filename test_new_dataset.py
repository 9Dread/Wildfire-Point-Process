from Modeling import PoissonLinPSPSEPSSLinMult
from Modeling import PoissonLinearIntensity
from Modeling import HawkesDiffusionLinbasePSPSEPSSLinMult
from Modeling import cross_validation
from Modeling import train_model
import torch
#import pyogrio
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from DataProcessing import WildfireIndexDataset, get_point24deg_grid, grid_to_cell_coords

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#get data
dataset = WildfireIndexDataset(128, 64, [2020, 2021, 2022, 2023], device, True, 'daily')
#Still use batch_size = 1 in the dataset itself (training loop will accumulate gradients)
#Batch size > 1 will trigger an error since event tensor sizes are unequal.
p = len(dataset.get_varnames())
loader = DataLoader(dataset, batch_size=1, shuffle=True) 
#data = [tup for tup in loader]

grid_gdf = get_point24deg_grid()
cell_coords = grid_to_cell_coords(grid_gdf, True)
cell_coords = cell_coords/1000 #convert to km units, helps with gradients
model = HawkesDiffusionLinbasePSPSEPSSLinMult(p, cell_coords)
#optimizer2 = torch.optim.SGD(model2.parameters(), lr=1e-7)
#def lambda_update2(epoch):
#    #we are starting at lr=1e-7; multiply by 10 every 100 epochs until 1e-4
#    if 10 ** (epoch // 100) < 100:
#        return 10 ** (epoch // 100)
#    else:
#        return 10 ** 3
#scheduler2 = LambdaLR(optimizer2, lr_lambda = lambda_update2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = ReduceLROnPlateau(optimizer, threshold_mode = 'abs', threshold = 0.01)
train_model(model, optimizer, loader, 3000, device, scheduler, batch_size = 1, print_iter=1)

#So now if we want to test on 2024 we can do
means, stds = dataset.get_transformation()
dataset_test = WildfireIndexDataset(366, 1, [2024], device, True, 'daily', means, stds) #(make one sequence for whole year)
#get the sequence data
covs, inhs, evs = dataset_test.get_subset(0)
with torch.no_grad():
    model.loglik(covs, inhs, evs)
#  Seq_len, overlap, batch_size, valid_ll
#  128, 64, 1, -4480.1484
torch.save(model.state_dict(), "SavedModels/flagship.pth")

#inh = parts['inhib']
#animate_intensity(inh, 'inhibitory effect (intensity decrease)', evs, cell_coords_viz, "Viz/test_inh_linmult.gif", scale="lin", decay=0.9)
#animate_intensity(lam, 'lam intensity', evs, cell_coords_viz, "Viz/test_lam_inhlinmult.gif", scale="lin", decay=0.9)
#With only features [0,1,9,10,15]: avg nll 4118.982080078125 (basically same as poisson glm?)
#With all features: 4122.285302734375 (almost exactly the same as poisson linear!)

#normal Poisson glm for comparison
dataset2 = WildfireIndexDataset(32, 8, [2020, 2021, 2022, 2023], device, False, 'daily')
p = len(dataset2.get_varnames())
loader2 = DataLoader(dataset2, batch_size=1, shuffle=True)
model2 = PoissonLinearIntensity(p)
#optimizer3 = torch.optim.SGD(model3.parameters(), lr = 1e-7)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-2)
#scheduler3 = LambdaLR(optimizer3, lr_lambda = lambda_update3)
scheduler2 = ReduceLROnPlateau(optimizer2, 'min', threshold=1e-2, threshold_mode = 'abs')
train_model(model2, optimizer2, loader2, 1000, device, scheduler2, batch_size = 1)

means, stds = dataset2.get_transformation()
dataset_test = WildfireIndexDataset(366, 1, [2024], device, False, 'daily', means, stds)
covs, evs = dataset_test.get_subset(0)
with torch.no_grad():
    model2.loglik(covs, evs)
#Tests with 6hr:
#64, 32, 1, -5582.5205
#64, 32, 16, -5574.7261
#64, 32, 32, -5971.9219 (bad)
#128, 32, 1, -5582.1533
#128, 32, 8, -5581.0918
#128, 32, 16, -5581.6279
#128, 64, 1, -5581.8115

#Tests with daily:
#128, 64, 1, -4559.4385
#128, 64, 2, -4559.8516
#128, 64, 4, -4573.6787 (Seems reduce performance. Probably because sequences overlap a lot.)
#128, 128, 1, -4560.9507
#128, 32, 1, -4562.2900
#128, 16, 1, -4562.3037
#64, 64, 1, -4559.7148
#64, 64, 2, -4559.8809
#64, 64, 4, -4607.6455 (Weird?)
#64, 32, 1, -4561.5493
#64, 16, 1, -4562.1182
#32, 32, 1, -4562.1553
#32, 24, 1, -4560.5742
#32, 16, 1, -4565.0747
#32, 8, 1, -4560.9492


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



