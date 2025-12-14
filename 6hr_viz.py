from Modeling import PoissonLinearIntensity
from Modeling import cross_validation
import torch
from torch.utils.data import DataLoader
from Modeling import train_model
from DataProcessing import get_covs_tensor_list, get_events_tensor_list, standardize_cov_tensors, get_inhib_tensor_list, WildfireInhibDataset, WildfireDataset, get_point24deg_grid, grid_to_cell_coords
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau


#6hr data
#get data
covars, varnames = get_covs_tensor_list("6hr")
events = get_events_tensor_list('6hr')
#mean/stddev standardization
covars = standardize_cov_tensors(covars, True)
p = covars[0].shape[2] 
model = PoissonLinearIntensity(p, True, True)
cross_validation(model, "Adam", 1e-2, covars, events, device, scheduler= ['Plateau'], save_path = "Results/results_6hr_experiment.csv", model_name = "Poisson_GLM_6hr2")

#daily data
covars2, varnames2 = get_covs_tensor_list()
events2 = get_events_tensor_list()
covars2 = standardize_cov_tensors(covars2, True)
model = PoissonLinearIntensity(p)
cross_validation(model, "Adam", 1e-2, covars2, events2, device, scheduler= ['Plateau'], save_path = "Results/results_6hr_experiment.csv", model_name = "Poisson_GLM_daily")

#To compute likelihood at daily resolution
model = PoissonLinearIntensity(p)
state_dict = torch.load("SavedModels/Poisson_GLM_6hr_cv.pth", map_location="cpu")  
model.load_state_dict(state_dict)
def aggregate_time_lam(lam_fine: torch.Tensor, r: int = 4):
    """
    lam_fine: (T_f, C) expected counts per fine bin
    r: number of fine bins per coarse bin (r=4 for daily<-6hr)
    returns lam_coarse: (T_c, C)
    """
    T_f, C = lam_fine.shape
    T_c = T_f // r
    rem = T_f % r
    if rem != 0:
        #handle remainder by padding with zeros for simplicity (or handle via bin edges)
        pad = lam_fine.new_zeros((r - rem, C))
        lam_fine = torch.cat([lam_fine, pad], dim=0)
        T_f = lam_fine.shape[0]
        T_c = T_f // r
    #reshape (T_c, r, C) then sum across the r axis
    lam_coarse = lam_fine.view(T_c, r, C).sum(dim=1)  # (T_c, C)
    return lam_coarse
def aggregate_event_times_to_coarse(events_fine: torch.LongTensor, r: int):
    """
    events_fine: (N,2) with columns (T_fine_idx, cell_id)
    returns events_coarse: (N,2) with (T_coarse_idx, cell_id)
    """
    T_fine = events_fine[:,0]
    cell = events_fine[:,1]
    T_coarse = T_fine // r
    return torch.stack([T_coarse, cell], dim=1)
r = 4  #6hr -> daily
lam_fine_list = [model.forward(covars[i]) for i in range(0,5)]  #(T_f, C)
lam_coarse_from_fine_list = [aggregate_time_lam(lam_fine_list[i], r) for i in range(0,5)]  #(T_c, C)
events_coarse_list = [aggregate_event_times_to_coarse(events[i], r) for i in range(0,5)]
def compute_likelihood(events_coarse, lam_agg):
    event_T = events_coarse[:,0].long()
    event_C = events_coarse[:,1].long()
    event_lams = lam_agg[event_T, event_C]  # (N_y)
    logsum = torch.sum(torch.log(event_lams))
    integral = torch.sum(lam_agg)
    return logsum - integral
lik_list = [compute_likelihood(events_coarse_list[i], lam_coarse_from_fine_list[i]) for i in range(0,5)]
train_lik = sum(lik_list[:4])/4
valid_lik = lik_list[4]
print(train_lik, ", ", valid_lik)

#load model and make a plot of one cell?
model = PoissonLinearIntensity(p)
state_dict = torch.load("SavedModels/Poisson_GLM_6hr_cv.pth", map_location="cpu")  
model.load_state_dict(state_dict)
model.to(device)
with torch.no_grad():
    lam_2024 = model(covars[4].to(device))

#grab a specific cell
cell = lam_2024[...,400]
#restrict to first few days
cell_first_days = cell[32:64]

cell_np = cell.detach().cpu().numpy()
cell_first_days_np = cell_first_days.detach().cpu().numpy()

plt.figure(figsize=(10,4))
plt.plot(cell_np)
plt.xlabel("Time step (6hr intervals)")
plt.ylabel("Intensity")
plt.title("Predicted Poisson intensity over time for one cell: Full validation year")
plt.show()
plt.close()

plt.figure(figsize=(10,4))
plt.plot(cell_first_days_np, color='orange')
plt.xlabel("Time step (6hr intervals)")
plt.ylabel("Intensity")
plt.title("Predicted Poisson intensity over time for one cell: Days 9 - 16 (32 time steps)")
plt.show()
plt.close()

model = PoissonLinearIntensity(p)
state_dict = torch.load("SavedModels/Poisson_GLM_daily_cv.pth", map_location="cpu")  
model.load_state_dict(state_dict)
model.to(device)
with torch.no_grad():
    lam_2024 = model(covars2[4].to(device))

#grab a specific cell
cell = lam_2024[...,400]
#restrict to first few days
cell_first_days = cell[9:16]

cell_np = cell.detach().cpu().numpy()
cell_first_days_np = cell_first_days.detach().cpu().numpy()

plt.figure(figsize=(10,4))
plt.plot(cell_np)
plt.xlabel("Time step (daily intervals)")
plt.ylabel("Intensity")
plt.title("Predicted Poisson intensity over time for one cell: Full validation year")
plt.show()
plt.close()

plt.figure(figsize=(10,4))
plt.plot(cell_first_days_np, color='orange')
plt.xlabel("Time step (daily intervals)")
plt.ylabel("Intensity")
plt.title("Predicted Poisson intensity over time for one cell: Days 9 - 16")
plt.show()
plt.close()