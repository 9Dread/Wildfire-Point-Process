from DataProcessing import get_events, get_point24deg_grid
import numpy as np
#The MLE of a homogenous poisson process with discrete time and grid cells is lambda = N/(TC) where N is the number of events in
#the sequence, T is the number of time steps over the sequence, and C is the number of spatial grid cells.
#Note that, in the homogenous case, we can do this by treating all 5 years as just one long sequence. So
counts = [get_events(year).shape[0] for year in range(2020,2025)]
N = sum(counts)
T = 366*1 + 365*4 #Two leap years, but missing the first day of 2020
C = get_point24deg_grid().shape[0]
#So our MLE is
lam = N/(T*C)
print(lam)
#And our log likelihood per year is
#(Nloglam - lam(TC))/5
loglik_all = (N*np.log(lam) - lam*T*C)/5
print(loglik_all) #-4571.210789954746

#If we fit on only 2020-2023 and validate on 2024 we have
counts = [get_events(year).shape[0] for year in range(2020,2024)]
N = sum(counts)
T = 365*4
lam = N/(T*C)
print(lam)
#Our log likelihood on the training set is
loglik_train = (N*np.log(lam) - lam*T*C)/4
print(loglik_train) #-4414.1729541359
#And our log likelihood on the test set is
N_valid = get_events(2024).shape[0]
loglik_valid = N_valid * np.log(lam) - lam * 366 * C
print(loglik_valid) #-5201.884405159815

#Save to Results/results.csv
import os
from pathlib import Path
import pandas as pd
save_path = "Results/results.csv"
filepath = Path(save_path)  
filepath.parent.mkdir(parents=True, exist_ok=True)
model_name = "Homogenous_Poisson"
if os.path.exists(save_path):
    #if csv exists, just append to it
    df = pd.DataFrame({'model_name': [model_name], 'training_ll_cv': [loglik_train], 'valid_ll_cv': [loglik_valid], 'overall_ll': loglik_all, 'cv_model_save_path': [pd.NA], 'all_model_save_path': pd.NA})
    df.to_csv(save_path, index=False, header=False, mode='a')
else:
    #else make new csv
    df = pd.DataFrame({'model_name': [model_name], 'training_ll_cv': [loglik_train], 'valid_ll_cv': [loglik_valid], 'overall_ll': loglik_all, 'cv_model_save_path': [pd.NA], 'all_model_save_path': pd.NA})
    df.to_csv(save_path, index=False)

#Sort values in csv:
df = pd.read_csv(save_path)
df = df.sort_values('valid_ll_cv', ascending=False)
df.to_csv(save_path, index=False)
