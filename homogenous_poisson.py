import Functions
import numpy as np
#The MLE of a homogenous poisson process with discrete time and grid cells is lambda = N/(TC) where N is the number of events in
#the sequence, T is the number of time steps over the sequence, and C is the number of spatial grid cells.
#Note that, in the homogenous case, we can do this by treating all 5 years as just one long sequence. So
counts = [Functions.get_events(year).shape[0] for year in range(2020,2025)]
N = sum(counts)
T = 366*2 + 365*3 #Two leap years
C = Functions.get_point24deg_grid(drop_missing_cov_cells=True).shape[0]
#So our MLE is
lam = N/(T*C)
print(lam)
#And our log likelihood is
#Nloglam - lam(TC)
loglik = N*np.log(lam) - lam*T*C
print(loglik) #very low!


#For only 2023 and 2024:
counts = [Functions.get_events(year).shape[0] for year in range(2023,2025)]
N = sum(counts)
T = 366 + 365
C = Functions.get_point24deg_grid(drop_missing_cov_cells=True).shape[0]
#So our MLE is
lam = N/(T*C)
print(lam)
#And our log likelihood is
#Nloglam - lam(TC)
loglik = N*np.log(lam) - lam*T*C
print(loglik) #very low!
