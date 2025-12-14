

import torch
import torch.nn as nn
import torch.nn.functional as F
from DataProcessing import WildfireDataset, WildfireInhibDataset
from torch.utils.data import DataLoader
import pandas as pd
from copy import deepcopy
from math import isnan
from pathlib import Path
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os

def train_model(model, optimizer, loader, max_epochs, device, scheduler = None, delta_loss = 0.01, improve_epochs = 5, batch_size = 1, print_iter = 10):
    """
    Runs a training loop with early stopping.
    model: one of the model classes initialized; must have loglik function
    optimizer: an initialized optimizer. 
    loader: a data loader with the training set.
    max_epochs: the maximum number of epochs to run.
    device: the device to train on.
    scheduler: optionally, a learning rate scheduler.
    delta_loss: how much the log likelihood must improve per epoch to continue training. Controls early stopping.
    improve_epochs: how many consecutive epochs in which we must have less than delta_loss improvement in order to stop training.
    print_iter: how often the epoch loss should be printed. used to monitor training.
    
    returns the most recent loss after training has stopped.
    """
    model.to(device)
    prev_loss = None
    avg_nll = None
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs+1):
        total_nll = 0.0
        if avg_nll is not None:
            prev_loss = avg_nll #update prev_loss from previous iteration
        optimizer.zero_grad()
        for i, tup in enumerate(loader):
            #if without inhib, get cov events; else get all
            #move to device, squeeze batch‐dim
            if len(tup) == 2:
                cov, events = tup
                cov = cov.squeeze(0).to(device)
                events = events.squeeze(0).to(device)
                #negative log likelihood
                nll = -model.loglik(cov, events)
            else:
                cov, inhib, events = tup
                cov = cov.squeeze(0).to(device)
                events = events.squeeze(0).to(device)
                inhib = inhib.squeeze(0).to(device)
                #negative log likelihood
                nll = -model.loglik(cov, inhib, events)

            #backward & step
            (nll/batch_size).backward()
            if ((i + 1) % batch_size == 0) or (i+1 == len(loader)):
                optimizer.step()
                optimizer.zero_grad()

            total_nll += nll.item()
        avg_nll = total_nll / len(loader)
        if scheduler is not None:
            #step scheduler
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_nll)
            else:
                scheduler.step()
        #initialize or update change in loss
        if epoch == 1:
            change_in_loss = delta_loss+1
        else:
            change_in_loss = prev_loss - avg_nll

        #update epochs without improvement
        if change_in_loss < delta_loss:
            epochs_without_improvement += 1
        else:
            epochs_without_improvement = 0
        
        if epochs_without_improvement > improve_epochs:
            #early stop if we had less than delta_loss improvement for improve_epochs consecutive epochs
            print(f"Training converged at epoch {epoch} with average negative log likelihood {avg_nll}.")
            print("This makes the total log likelihood over the training set: " + str(-(avg_nll * len(loader))))
            return -(avg_nll * len(loader))
        
        if isnan(avg_nll):
            #break early if nan 
            print("Err: NaN loss! Breaking loop.")
            return None

        if (epoch - 1) % print_iter == 0:
            if scheduler is not None:
                print(f"Epoch {epoch:02d} — Avg NLL: {avg_nll:.4f} — LR: ", scheduler.get_last_lr())
            else:
                print(f"Epoch {epoch:02d} — Avg NLL: {avg_nll:.4f}")

        

    print(f"{max_epochs} epochs completed with most recent average negative log likelihood {avg_nll}.")
    print("This makes the total log likelihood over the training set: " + str(-(avg_nll * len(loader))))
    return -(avg_nll * len(loader))

def cross_validation(model, optimizer, lr, covars, events, device, inhs = None, scheduler = None, delta_loss = 0.01, improve_epochs = 5, print_iter = 10, save_results=True, save_path = None, model_name = None, train_on_all=True, batch_size = 1):
    """
    cross validation routine using 2024 as the test set.

    model: one of the model classes initialized
    optimizer: "SGD" or "Adam"
    lr: learning rate passed to optimizer.
    covars: a list of tensors of shape (T_y, C, p); spacetime grid of covariates
    events: a list of tensors of shape (N_y, 2) containing the time and space indices of each event occurrence
    device: a pytorch device
    inhs: inhibitory effect mask tensors (psps/epss). pass these if training an inhib model.
    scheduler: optional, either ['LambdaLR', function] or ['Plateau']; LambdaLR useful for SGD, plateau useful for Adam
    delta_loss: passed to train_model
    improve_epochs: passed to train_model
    print_iter: passed to train_model
    save_results: whether or not to save the results of the analysis. If true, will save to csv in save_path or append
        to it if it already exists.
    save_path: the path to save the csv of the results of the cross validation to. must be provided if save_results is true.
    model_name: name of the model for saving state dict. must be provided if save_results = true.
    train_on_all: whether or not to run training on the *entire* dataset after cross validation is complete. useful
        if you want to additionally save the training likelihood of the model class, not just the validation likelihood. 
    returns a list of the LL over training set, validation year, and over all 5 years when trained on all years if train_on_all=true.
    """
    if save_results:
        assert save_path is not None, "If save_results=True, make sure save_path provides a csv save path! This can be an existing csv to append to or a path to create one."
        assert model_name is not None, "If save_results=True, make sure model_name is provided!"
    assert (optimizer == "SGD") | (optimizer == "Adam"), "Optimizer must be 'SGD' or 'Adam'"
    if scheduler is not None:
        assert(scheduler[0] == "LambdaLR") | (scheduler[0] == "Plateau"), "if scheduler is used, it must be either ['LambdaLR', function] or ['Plateau']"
    out_list = []
    #make data loader for 2020-2023
    training_covars = [covars[j] for j in range(0, len(covars) - 1)]
    training_events = [events[j] for j in range(0, len(covars) - 1)]
    if inhs is not None:
        training_inhs = [inhs[k] for k in range(0, len(inhs))]
        dataset = WildfireInhibDataset(training_covars, training_inhs, training_events)
    else:
        dataset = WildfireDataset(training_covars, training_events)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    

    #train loop
    #just keep running until training converges:
    training_ll = None
    while training_ll is None:
        #if we get NaN loss, retry and keep going:
        model_copy = deepcopy(model).to(device)
        if optimizer == "Adam":
            optim = torch.optim.Adam(model_copy.parameters(), lr=lr)
        else:
            optim = torch.optim.SGD(model_copy.parameters(), lr=lr)

        sched = None
        if scheduler is not None:
            if scheduler[0] == 'LambdaLR':
                sched = LambdaLR(optim, lr_lambda=scheduler[1])
            else:
                sched = ReduceLROnPlateau(optim, 'min', threshold=1e-2, threshold_mode = 'abs')
        training_ll = train_model(model_copy, optim, loader, 1000000, device, scheduler=sched, delta_loss=delta_loss, improve_epochs = improve_epochs, batch_size = batch_size, print_iter=print_iter)
    out_list.append(training_ll / 4)
    save_cv_model_path = f"SavedModels/{model_name}_cv.pth"
    torch.save(model_copy.state_dict(), save_cv_model_path)


    #evaluate loglik on unseen data:
    if inhs is not None:
        loglik = float(model_copy.loglik(covars[len(covars)-1].to(device), inhs[len(covars)-1].to(device), events[len(covars)-1].int().to(device)))
    else:
        loglik = float(model_copy.loglik(covars[len(covars)-1].to(device), events[len(covars)-1].int().to(device)))
    out_list.append(loglik)

    save_full_model_path = None
    all_ll = None
    #train on all data if train_on_all = true
    if train_on_all:
        if inhs is not None:
            dataset = WildfireInhibDataset(covars, inhs, events)
        else:
            dataset = WildfireDataset(covars, events)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        while all_ll is None:
            #if we get NaN loss, retry and keep going:
            model_copy = deepcopy(model).to(device)
            if optimizer == "Adam":
                optim = torch.optim.Adam(model_copy.parameters(), lr=lr)
            else:
                optim = torch.optim.SGD(model_copy.parameters(), lr=lr)
            sched = None
            if scheduler is not None:
                if scheduler[0] == 'LambdaLR':
                    sched = LambdaLR(optim, lr_lambda=scheduler[1])
                else:
                    sched = ReduceLROnPlateau(optim, 'min', threshold=1e-2, threshold_mode = 'abs')
            all_ll = train_model(model_copy, optim, loader, 1000000, device, scheduler=sched, delta_loss=delta_loss, improve_epochs = improve_epochs, print_iter=print_iter)
        #append ll
        out_list.append(all_ll / 5)
        save_full_model_path = f"SavedModels/{model_name}_all.pth"
        torch.save(model_copy.state_dict(), save_full_model_path)
    
    #save results to csv
    if save_results:
        filepath = Path(save_path)  
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(save_path):
            #if csv exists, just append to it

            model_trained_all_path = [pd.NA] if save_full_model_path is None else [save_full_model_path]
            trained_all_ll = [pd.NA] if all_ll is None else [all_ll/5]
            df = pd.DataFrame({'model_name': [model_name], 'training_ll_cv': [training_ll/4], 'valid_ll_cv': [loglik], 'overall_ll': trained_all_ll, 'cv_model_save_path': [save_cv_model_path], 'all_model_save_path': model_trained_all_path})
            df.to_csv(save_path, index=False, header=False, mode='a')
        else:
            #else make new csv
            model_trained_all_path = [pd.NA] if save_full_model_path is None else [save_full_model_path]
            trained_all_ll = [pd.NA] if all_ll is None else [all_ll/5]
            df = pd.DataFrame({'model_name': [model_name], 'training_ll_cv': [training_ll/4], 'valid_ll_cv': [loglik], 'overall_ll': trained_all_ll, 'cv_model_save_path': [save_cv_model_path], 'all_model_save_path': model_trained_all_path})
            df.to_csv(save_path, index=False)
        
        #Sort values in the csv by validation performance:
        df = pd.read_csv(save_path)
        df = df.sort_values('valid_ll_cv', ascending=False)
        df.to_csv(save_path, index=False)

    return out_list



class PoissonLinearIntensity(nn.Module):
    """
    Poisson-GLM intensity model with log-link (the canonical link function)

    Input: cov of shape (T, C, p)
    Output: lam of shape (T, C)

    """
    def __init__(self, num_covariates: int, bias: bool = True, sixhr = False):
        super().__init__()
        self.linear = nn.Linear(num_covariates, 1, bias=bias)
        self.sixhr = sixhr

    def forward(self, cov: torch.Tensor) -> torch.Tensor:
        """ 
        cov: (T, C, p); batching not implemented because it's somewhat of a pain when years have variable time steps
        returns (T,C) grid of lambda intensity values for each grid cell at each time step
        """
        dims = cov.dim()
        if dims == 4 and cov.shape[0] == 1:
            cov = cov.squeeze(0)
            dims = cov.dim()
        if dims == 3:
            T, C, p = cov.shape
            flat = cov.view(-1, p) #(T*C, p) for vectorized linear computation
            nu = self.linear(flat).view(T, C) #(T, C)
        else:
            raise ValueError(f"Expected 3D input, got {dims}D")

        lam = torch.exp(nu) #log link
        return lam
    
    def loglik(self, cov: torch.Tensor, events: torch.Tensor):
        """
        cov: (T,C,p) tensor of covariates for any given year
        events: (N_y, 2) tensor of N_y events (all events that occurred in a given year) 
            with their time steps [,0] and grid cell ids [,1]
        returns the log-likelihood of the year.
        """
        lam = self.forward(cov) #(T,C)
        if self.sixhr:
            #evaluate daily likelihood
            r = 4
            T_f, C = lam.shape
            T_c = T_f // r
            rem = T_f % r
            if rem != 0:
                #handle remainder by padding with zeros for simplicity (or handle via bin edges)
                pad = lam.new_zeros((r - rem, C))
                lam_fine = torch.cat([lam, pad], dim=0)
                T_f = lam_fine.shape[0]
                T_c = T_f // r
            #reshape (T_c, r, C) then sum across the r axis
            lam_coarse = lam_fine.view(T_c, r, C).sum(dim=1)  #(T_c, C)

            #aggregate events to daily resolution
            T_fine = events[:,0]
            cell = events[:,1]
            T_coarse = T_fine // r
            events_coarse = torch.stack([T_coarse, cell], dim=1)
            event_T = events_coarse[:,0].long()
            event_C = events_coarse[:,1].long()
            event_lams = lam_coarse[event_T, event_C]  #(N_y)
            logsum = torch.sum(torch.log(event_lams))
            integral = torch.sum(lam_coarse)
            return logsum - integral
        T_ids = events[:, 0]
        C_ids = events[:, 1]
        event_lams = lam[T_ids, C_ids] #(N_y) containing lambdas of events
        logsum = torch.sum(torch.log(event_lams)) #logsum term of loglikelihood
        integral = torch.sum(lam) #integral term of loglikelihood
        return logsum - integral

class PoissonNeuralIntensity(nn.Module):
    """
    Poisson Neural intensity model

    Input: cov of shape (T, C, p)
    Output: lam of shape (T, C)

    hidden_dim: width of each hidden layer.
    num_hidden_layers: number of hidden layers (default=2).

    """
    def __init__(self, num_covariates: int, hidden_dim: int, num_hidden_layers: int = 2, bias: bool = True):
        super().__init__()
        layers = []
        in_dim = num_covariates
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        #final projection to scalar
        layers.append(nn.Linear(in_dim, 1, bias=bias))
        self.mlp = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

    def forward(self, cov: torch.Tensor) -> torch.Tensor:
        """
        cov: (T, C, p); batching not implemented because it's somewhat of a pain when years have variable time steps
        returns (T,C) grid of lambda intensity values for each grid cell at each time step
        """
        dims = cov.dim()
        if dims == 4 and cov.shape[0] == 1:
            cov = cov.squeeze(0)
            dims = cov.dim()
        if dims == 3:
            T, C, p = cov.shape
            flat = cov.view(-1, p) #(T*C, p) for vectorized forward thru MLP
            nu = self.mlp(flat).view(T, C) #(T, C)
        else:
            raise ValueError(f"Expected 3D input, got {dims}D")

        lam = torch.exp(nu) #log link
        return lam
    
    def loglik(self, cov: torch.Tensor, events: torch.Tensor):
        """
        cov: (T,C,p) tensor of covariates for any given year
        events: (N_y, 2) tensor of N_y events (all events that occurred in a given year) 
            with their time steps [,0] and grid cell ids [,1]
        returns the log-likelihood of the year.
        """
        lam = self.forward(cov) #(T,C)
        T_ids = events[:, 0]
        C_ids = events[:, 1]
        event_lams = lam[T_ids, C_ids].clamp_min(1e-12) #(N_y) containing lambdas of events, clamped before log so that we dont do log(0)
        logsum = torch.sum(torch.log(event_lams)) #logsum term of loglikelihood
        integral = torch.sum(lam) #integral term of loglikelihood
        return logsum - integral

class HawkesDiffusionLinbase(nn.Module):
    """
    Spatiotemporal Hawkes with diffusion-type kernel (Musmeci & Vere-Jones, 1992)
    + linear (log-link) baseline.

    cov: (T, C, p) covariates (float)
    events_tc: (N,2) long/float tensor with columns:
                [time_index, cell_index]
                Times are 0..T-1 (integers). Only events with time < t affect time t.
    return_parts: if True, also returns dict with 'baseline' and 'excitation'

    Notes:
    cell_coords should be in a metric CRS ideally (e.g. 3310 for CA)
    kernel parameters C, beta, sigma_x, sigma_y are positive via softplus.
    can add a `max_lag` cutoff if needed.
    """
    def __init__(self,
                 num_covariates: int,
                 cell_coords,
                 bias: bool = True,
                 init_C: float = 8., #Scale excitation intensity; init big so that gradient descent doesn't ignore it
                 init_beta: float = 0.5, #time damping
                 init_sigma: float = 5., #kilometer scale
                 max_lag: int | None = None, #optional time cutoff (in time step units), could be useful for 6hr resolution?
                 coalesce_duplicates: bool = True #combine events in the same spacetime cell using weighting, making things faster
                 ):
        super().__init__()

        #baseline (Poisson GLM with log-link)
        self.linear = nn.Linear(num_covariates, 1, bias=bias)

        #raw kernel parameters initialization (before softplus)
        self.raw_C = nn.Parameter(torch.tensor(float(init_C)))
        self.raw_beta = nn.Parameter(torch.tensor(float(init_beta)))
        self.raw_sigma = nn.Parameter(torch.tensor(float(init_sigma)))

        #precompute squared deltas in x and y for all cell pairs
        coords = torch.as_tensor(cell_coords, dtype=torch.float32)  #(C,2)
        dx = coords[:, 0].unsqueeze(0) - coords[:, 0].unsqueeze(1) #(C,C)
        dy = coords[:, 1].unsqueeze(0) - coords[:, 1].unsqueeze(1) #(C,C)
        DX2 = dx.pow(2)
        DY2 = dy.pow(2)
        self.register_buffer("DX2", DX2, persistent=False)
        self.register_buffer("DY2", DY2, persistent=False)

        self.max_lag = max_lag
        self._two_pi = 2.0 * torch.pi
        self._eps = 1e-12

        self.coalesce_duplicates = coalesce_duplicates

    #positive parameter views
    @property
    def C(self):
        return torch.exp(self.raw_C) + self._eps #here we use exp instead of softplus because we expect scale parameter to be big but still adjustable with autograd
    @property
    def beta(self):
        return F.softplus(self.raw_beta) + self._eps
    @property
    def sigma(self):
        return F.softplus(self.raw_sigma) + self._eps

    #bseline: exp(X beta)
    def _baseline(self, cov: torch.Tensor) -> torch.Tensor:
        if cov.dim() == 4 and cov.shape[0] == 1:
            cov = cov.squeeze(0)
        if cov.dim() != 3:
            raise ValueError(f"cov must be (T,C,p), got {tuple(cov.shape)}")
        T, C, p = cov.shape
        eta = self.linear(cov.view(-1, p)).view(T, C)
        #mild clamp to avoid overflow
        eta = torch.clamp(eta, -20.0, 20.0)
        return torch.exp(eta)

    #excitation component via diffusion kernel
    def _excitation(self, T: int, events_tc: torch.Tensor) -> torch.Tensor:
        if events_tc.dim() == 3 and events_tc.shape[0] == 1:
            events_tc = events_tc.squeeze(0)
        if events_tc.dim() != 2:
            raise ValueError(f"events_tc must be 2D, got {events_tc.dim()}D")
        Ccells = self.DX2.size(0)
        device = self.DX2.device
        lam_exc = torch.zeros(T, Ccells, device=device)

        if events_tc.numel() == 0:
            return lam_exc #if no events return 0s, just in case
        if events_tc.dtype != torch.long:
            events_tc = events_tc.long()
        times = events_tc[:, 0]
        cells = events_tc[:, 1]

        #keep only events in [0, T-1]
        mask = (times >= 0) & (times < T)
        times, cells = times[mask], cells[mask]

        #(optional) coalesce duplicates: same (t,c) but with count weight
        if self.coalesce_duplicates:
            tc = torch.stack([times, cells], dim=1)
            uniq, counts = torch.unique(tc, dim=0, return_counts=True)
            times = uniq[:, 0]
            cells = uniq[:, 1]
            weights = counts.to(torch.float32)
        else:
            weights = torch.ones_like(times, dtype=torch.float32)

        #sort by time so we can slice “past events” by prefix length
        order = torch.argsort(times)
        times, cells, weights = times[order], cells[order], weights[order]

        #prefix counts per day used for m(t) = #events with time < t during loop
        per_day = torch.bincount(times, minlength=T) #(T,)
        csum = torch.cumsum(per_day, dim=0) #(T,)

        #precompute static pieces
        sigma2 = self.sigma * self.sigma
        norm_xy = self._two_pi * sigma2

        for t in range(T):
            m = 0 if t == 0 else int(csum[t - 1].item())
            if m == 0:
                continue #skip 1st day since we have no history

            #optional max lag (days)
            if self.max_lag is not None:
                min_time = max(0, t - self.max_lag)
                start = torch.searchsorted(times[:m], torch.as_tensor(min_time, device=times.device))
            else:
                start = 0

            idx = slice(start, m)
            #events which are influencing the current time step t
            evt_cells = cells[idx]  #(m',) 
            evt_times = times[idx] #(m',)
            evt_w = weights[idx].to(torch.float32)

            #delt (strictly positive)
            dt = (t - evt_times).clamp_min(1).to(torch.float32) #(m',)

            #spatial part for all target cells vs each source cell
            #scaled_d2[c, j] = DX2[c, src_j]/sigmax^2 + DY2[c, src_j]/sigmay^2
            scaled_d2 = ((self.DX2[:, evt_cells] + self.DY2[:, evt_cells]) / sigma2) #(C, m')

            #time factor per event j
            factor_time = torch.exp(-self.beta * dt) * (self.C / (norm_xy * dt)) * evt_w #(m',)

            #exp(- scaled_d2 / (2 dt)) for each event j, (C, m')
            inv_dt = (1.0 / dt).unsqueeze(0) #(1, m')
            factor_space = torch.exp(-0.5 * scaled_d2 * inv_dt) #(C, m')

            lam_exc[t] = (factor_space * factor_time.unsqueeze(0)).sum(dim=1)

        return lam_exc

    def forward(self,
                cov: torch.Tensor, #(T,C,p)
                events_tc: torch.Tensor, #(N,2)
                return_parts: bool = False):
        if cov.shape[0] == 1:
            cov=cov.squeeze(0)
        if events_tc.shape[0] == 1:
            events_tc=events_tc.squeeze(0)
        T, C, _ = cov.shape
        lam_base = self._baseline(cov) #(T, C)
        lam_exc = self._excitation(T, events_tc) #(T, C)
        lam_total = lam_base + lam_exc
        if return_parts:
            return lam_total, {"baseline": lam_base, "excitation": lam_exc}
        return lam_total
    
    def loglik(self, cov: torch.Tensor, events: torch.Tensor):
        """
        cov: (T,C,p) tensor of covariates for any given year
        events: (N_y, 2) tensor of N_y events (all events that occurred in a given year) 
            with their time steps [,0] and grid cell ids [,1]
        returns the log-likelihood of the year.
        """
        lam = self.forward(cov, events) #(T,C)
        T_ids = events[:, 0]
        C_ids = events[:, 1]
        event_lams = lam[T_ids, C_ids].clamp_min(1e-12) #(N_y) containing lambdas of events, clamped before log so that we dont do log(0)
        logsum = torch.sum(torch.log(event_lams)) #logsum term of loglikelihood
        integral = torch.sum(lam) #integral term of loglikelihood
        return logsum - integral

class PoissonLinPSPSEPSSFlat(nn.Module):
    """
    Poisson linear + flat inhibition effects from PSPS/EPSS. Used for experimenting whether additive/multiplicative
    effects are more stable for PSPS/EPSS inhibition. Not using Hawkes kernel for testing because it is expensive.
    Input: cov of shape (T, C, p)
    Output: lam of shape (T, C)

    """
    def __init__(self, num_covariates: int, init_psps: float = 0., init_epss: float = 0., bias: bool = True):
        #init_psps and init_epss negative so that the softplus is small
        super().__init__()
        self.linear = nn.Linear(num_covariates, 1, bias=bias)

        #init bias large negative
        self.raw_psps = nn.Parameter(torch.tensor(float(init_psps)))
        self.raw_epss = nn.Parameter(torch.tensor(float(init_epss)))
        
    @property
    def psps(self):
        return torch.sigmoid(self.raw_psps)
    
    @property
    def epss(self):
        return torch.sigmoid(self.raw_epss)

    def _baseline(self, cov: torch.Tensor) -> torch.Tensor:
        """ 
        cov: (T, C, p); batching not implemented because it's somewhat of a pain when years have variable time steps
        returns (T,C) grid of lambda intensity values for each grid cell at each time step
        """
        dims = cov.dim()
        if dims == 4 and cov.shape[0] == 1:
            cov = cov.squeeze(0)
            dims = cov.dim()
        if dims == 3:
            T, C, p = cov.shape
            flat = cov.view(-1, p) #(T*C, p) for vectorized linear computation
            nu = self.linear(flat).view(T, C) #(T, C)
        else:
            raise ValueError(f"Expected 3D input, got {dims}D")

        lam = torch.exp(nu) #log link
        return lam

    def _inhibition(self, inhib: torch.Tensor) -> torch.Tensor:
        if inhib.dim() == 4 and inhib.shape[0] == 1:
            inhib = inhib.squeeze(0)
        if inhib.dim() != 3:
            raise ValueError(f"inhib must be 3D, got {inhib.dim()}D")

        #masks
        mask_psps = inhib[..., 0]  #(T, C)
        mask_epss = inhib[..., 1]  #(T, C)

        #apply masks
        eta_psps = self.psps * mask_psps
        eta_epss = self.epss * mask_epss
        inhib_psps = torch.where(mask_psps.bool(), eta_psps, torch.ones_like(eta_psps))
        inhib_epss = torch.where(mask_epss.bool(), eta_epss, torch.ones_like(eta_epss))
        return inhib_psps * inhib_epss

    def forward(self,
                cov: torch.Tensor, #(T,C,p)
                inhib: torch.Tensor, #(T,C,2)
                events_tc: torch.Tensor, #(N,2)
                return_parts: bool = False):
        if cov.shape[0] == 1:
            cov=cov.squeeze(0)
        if inhib.shape[0] == 1:
            inhib=inhib.squeeze(0)
        if events_tc.shape[0] == 1:
            events_tc=events_tc.squeeze(0)
        T, C, _ = cov.shape
        lam_base = self._baseline(cov) #(T, C)
        lam_inhib = self._inhibition(inhib) #(T, C)
        if return_parts:
            lam_total = lam_base * lam_inhib
            #We need to recover exactly how much lam_inhib is affecting lam_total, i.e. how much lam_base decreases when multiplied by lam_inhib
            lam_decreased = lam_base * -(1 - lam_inhib)
            return lam_total, {"baseline": lam_base, 'inhib': lam_decreased}
        return lam_base * lam_inhib
    
    def loglik(self, cov: torch.Tensor, inhib: torch.Tensor, events: torch.Tensor):
        """
        cov: (T,C,p) tensor of covariates for any given year
        inhib: (T,C,2) tensor of psps and epss inhibitory event indicators
        events: (N_y, 2) tensor of N_y events (all events that occurred in a given year) 
            with their time steps [,0] and grid cell ids [,1]
        returns the log-likelihood of the year.
        """
        lam = self.forward(cov, inhib, events) #(T,C)
        T_ids = events[:, 0]
        C_ids = events[:, 1]
        event_lams = lam[T_ids, C_ids].clamp_min(1e-12) #(N_y) containing lambdas of events, clamped before log so that we dont do log(0)
        logsum = torch.sum(torch.log(event_lams)) #logsum term of loglikelihood
        integral = torch.sum(lam) #integral term of loglikelihood

        #sum of abs parameters in PSPS/EPSS for L1 regularization
        #psps_parmsum = sum(p.abs().sum() for p in self.psps_linear.parameters()) 
        #epss_parmsum = sum(p.abs().sum() for p in self.epss_linear.parameters())

        #penalize weights in epss/psps:
        #return logsum - integral - self.L1_weight * (psps_parmsum + epss_parmsum)
        return logsum - integral

class PoissonLinPSPSEPSSLin(nn.Module):
    """
    Poisson linear + linear inhibition effects from PSPS/EPSS. Used for experimenting whether additive/multiplicative
    effects are more stable for PSPS/EPSS inhibition. Not using Hawkes kernel for testing because it is expensive.
    Input: cov of shape (T, C, p)
    Output: lam of shape (T, C)

    """
    def __init__(self, num_covariates: int, inhib_feats, inhib_init_bias: float = -1., bias: bool = True):
        #inhib_init_bias should be large and negative so that the softplus of the effects is small
        super().__init__()
        self.linear = nn.Linear(num_covariates, 1, bias=bias)
        self.inhib_feats = inhib_feats
        self.psps_linear = nn.Linear(len(self.inhib_feats), 1, bias=True)
        self.epss_linear = nn.Linear(len(self.inhib_feats), 1, bias=True)

        #init bias large negative
        with torch.no_grad():
            self.psps_linear.bias.fill_(inhib_init_bias) 
            self.epss_linear.bias.fill_(inhib_init_bias) 
        

    def _baseline(self, cov: torch.Tensor) -> torch.Tensor:
        """ 
        cov: (T, C, p); batching not implemented because it's somewhat of a pain when years have variable time steps
        returns (T,C) grid of lambda intensity values for each grid cell at each time step
        """
        dims = cov.dim()
        if dims == 4 and cov.shape[0] == 1:
            cov = cov.squeeze(0)
            dims = cov.dim()
        if dims == 3:
            T, C, p = cov.shape
            flat = cov.view(-1, p) #(T*C, p) for vectorized linear computation
            nu = self.linear(flat).view(T, C) #(T, C)
        else:
            raise ValueError(f"Expected 3D input, got {dims}D")

        lam = torch.exp(nu) #log link
        return lam

    def _inhibition(self, inhib: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        if inhib.dim() == 4 and inhib.shape[0] == 1:
            inhib = inhib.squeeze(0)
        if inhib.dim() != 3:
            raise ValueError(f"inhib must be 3D, got {inhib.dim()}D")

        if cov.dim() == 4 and cov.shape[0] == 1:
            cov = cov.squeeze(0)
        if cov.dim() != 3:
            raise ValueError(f"cov must be (T,C,p), got {tuple(cov.shape)}")
        
        T, C, p = cov.shape
        #forward through linears
        cov_inhib = cov[..., self.inhib_feats]
        eta_psps = -(F.softplus(self.psps_linear(cov_inhib.view(-1, len(self.inhib_feats))).view(T, C)))
        eta_epss = -(F.softplus(self.epss_linear(cov_inhib.view(-1, len(self.inhib_feats))).view(T, C)))

        #masks
        mask_psps = inhib[..., 0]  #(T, C)
        mask_epss = inhib[..., 1]  #(T, C)

        #apply masks
        inhib_psps = eta_psps * mask_psps
        inhib_epss = eta_epss * mask_epss
        return inhib_psps + inhib_epss

    def forward(self,
                cov: torch.Tensor, #(T,C,p)
                inhib: torch.Tensor, #(T,C,2)
                events_tc: torch.Tensor, #(N,2)
                return_parts: bool = False):
        if cov.shape[0] == 1:
            cov=cov.squeeze(0)
        if inhib.shape[0] == 1:
            inhib=inhib.squeeze(0)
        if events_tc.shape[0] == 1:
            events_tc=events_tc.squeeze(0)
        T, C, _ = cov.shape
        lam_base = self._baseline(cov) #(T, C)
        lam_inhib = self._inhibition(inhib, cov) #(T, C)
        if return_parts:
            lam_total = lam_base + lam_inhib
            #We need to recover exactly how much lam_inhib is affecting lam_total. We do not want extra magnitude that sends lam below 0.
            negative_mask = (lam_total < 0).float()
            lam_negative = torch.mul(lam_total, negative_mask) #only keep negative values
            lam_inhib_fixed = lam_inhib + (-1 * lam_negative) #only retains the actual effects inhib had on lambda
            return F.relu(lam_total), {"baseline": lam_base, 'inhib': lam_inhib_fixed}
        lam_total = F.relu(lam_base + lam_inhib)
        return lam_total
    
    def loglik(self, cov: torch.Tensor, inhib: torch.Tensor, events: torch.Tensor):
        """
        cov: (T,C,p) tensor of covariates for any given year
        inhib: (T,C,2) tensor of psps and epss inhibitory event indicators
        events: (N_y, 2) tensor of N_y events (all events that occurred in a given year) 
            with their time steps [,0] and grid cell ids [,1]
        returns the log-likelihood of the year.
        """
        lam = self.forward(cov, inhib, events) #(T,C)
        T_ids = events[:, 0]
        C_ids = events[:, 1]
        event_lams = lam[T_ids, C_ids].clamp_min(1e-12) #(N_y) containing lambdas of events, clamped before log so that we dont do log(0)
        logsum = torch.sum(torch.log(event_lams)) #logsum term of loglikelihood
        integral = torch.sum(lam) #integral term of loglikelihood

        #sum of abs parameters in PSPS/EPSS for L1 regularization
        #psps_parmsum = sum(p.abs().sum() for p in self.psps_linear.parameters()) 
        #epss_parmsum = sum(p.abs().sum() for p in self.epss_linear.parameters())

        #penalize weights in epss/psps:
        #return logsum - integral - self.L1_weight * (psps_parmsum + epss_parmsum)
        return logsum - integral

class PoissonLinPSPSEPSSLinMult(nn.Module):
    """
    Poisson linear + linear multiplicative inhibition effects from PSPS/EPSS. Used for experimenting whether additive/multiplicative
    effects are more stable for PSPS/EPSS inhibition. Not using Hawkes kernel for testing because it is expensive.
    Here the epss/psps linear outputs are shoved into a sigmoid and then multiplied by the entire intensity.
    Input: cov of shape (T, C, p)
    Output: lam of shape (T, C)

    """
    def __init__(self, num_covariates: int, inhib_feats = None, bias: bool = True):
        #L1 weight: for L1 regularization. Note that the scale of the weights is fairly small compared to our likelihood
        #so to penalize properly this needs to be tuned. 1 by default, maybe should be increased.
        super().__init__()
        self.linear = nn.Linear(num_covariates, 1, bias=bias)
        if inhib_feats is not None:
            self.inhib_feats = inhib_feats
        else:
            self.inhib_feats = list(range(0,num_covariates))
        self.psps_linear = nn.Linear(len(self.inhib_feats), 1, bias=True)
        self.epss_linear = nn.Linear(len(self.inhib_feats), 1, bias=True)
        

    def _baseline(self, cov: torch.Tensor) -> torch.Tensor:
        """ 
        cov: (T, C, p); batching not implemented because it's somewhat of a pain when years have variable time steps
        returns (T,C) grid of lambda intensity values for each grid cell at each time step
        """
        dims = cov.dim()
        if dims == 4 and cov.shape[0] == 1:
            cov = cov.squeeze(0)
            dims = cov.dim()
        if dims == 3:
            T, C, p = cov.shape
            flat = cov.view(-1, p) #(T*C, p) for vectorized linear computation
            nu = self.linear(flat).view(T, C) #(T, C)
        else:
            raise ValueError(f"Expected 3D input, got {dims}D")

        lam = torch.exp(nu) #log link
        return lam

    def _inhibition(self, inhib: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        if inhib.dim() == 4 and inhib.shape[0] == 1:
            inhib = inhib.squeeze(0)
        if inhib.dim() != 3:
            raise ValueError(f"inhib must be 3D, got {inhib.dim()}D")

        if cov.dim() == 4 and cov.shape[0] == 1:
            cov = cov.squeeze(0)
        if cov.dim() != 3:
            raise ValueError(f"cov must be (T,C,p), got {tuple(cov.shape)}")
        
        T, C, p = cov.shape
        #forward through linears
        cov_inhib = cov[..., self.inhib_feats]
        eta_psps = torch.sigmoid(self.psps_linear(cov_inhib.view(-1, len(self.inhib_feats))).view(T, C))
        eta_epss = torch.sigmoid(self.epss_linear(cov_inhib.view(-1, len(self.inhib_feats))).view(T, C))

        #masks
        mask_psps = inhib[..., 0]  #(T, C)
        mask_epss = inhib[..., 1]  #(T, C)

        #apply masks
        inhib_psps = torch.where(mask_psps.bool(), eta_psps, torch.ones_like(eta_psps))
        inhib_epss = torch.where(mask_epss.bool(), eta_epss, torch.ones_like(eta_epss))
        
        return inhib_psps * inhib_epss

    def forward(self,
                cov: torch.Tensor, #(T,C,p)
                inhib: torch.Tensor, #(T,C,2)
                events_tc: torch.Tensor, #(N,2)
                return_parts: bool = False):
        if cov.shape[0] == 1:
            cov=cov.squeeze(0)
        if inhib.shape[0] == 1:
            inhib=inhib.squeeze(0)
        if events_tc.shape[0] == 1:
            events_tc=events_tc.squeeze(0)
        T, C, _ = cov.shape
        lam_base = self._baseline(cov) #(T, C)
        lam_inhib = self._inhibition(inhib, cov) #(T, C)
        if return_parts:
            lam_total = lam_base * lam_inhib
            #We need to recover exactly how much lam_inhib is affecting lam_total, i.e. how much lam_base decreases when multiplied by lam_inhib
            lam_decreased = lam_base * -(1 - lam_inhib)
            return lam_total, {"baseline": lam_base, 'inhib': lam_decreased}
        return lam_base * lam_inhib
    
    def loglik(self, cov: torch.Tensor, inhib: torch.Tensor, events: torch.Tensor):
        """
        cov: (T,C,p) tensor of covariates for any given year
        inhib: (T,C,2) tensor of psps and epss inhibitory event indicators
        events: (N_y, 2) tensor of N_y events (all events that occurred in a given year) 
            with their time steps [,0] and grid cell ids [,1]
        returns the log-likelihood of the year.
        """
        lam = self.forward(cov, inhib, events) #(T,C)
        T_ids = events[:, 0]
        C_ids = events[:, 1]
        event_lams = lam[T_ids, C_ids].clamp_min(1e-12) #(N_y) containing lambdas of events, clamped before log so that we dont do log(0)
        logsum = torch.sum(torch.log(event_lams)) #logsum term of loglikelihood
        integral = torch.sum(lam) #integral term of loglikelihood

        return logsum - integral

class HawkesDiffusionLinbasePSPSEPSSLin(nn.Module):
    """
    Spatiotemporal Hawkes with diffusion-type kernel (Musmeci & Vere-Jones, 1992),
    + linear (log-link) baseline, + additive intensity inhibitions from PSPS/EPSS events

    cov: (T, C, p) covariates (float)
    inhib: (T,C,2) bool indicators for where PSPS or EPSS inhibitory events are active/inactive
    events_tc: (N,2) long/float tensor with columns:
                [time_index, cell_index]
                Times are 0..T-1 (integers). Only events with time < t affect time t.
    return_parts: if True, also returns dict with 'baseline', 'excitation', and 'inhibition'

    Notes:
    cell_coords should be in a metric CRS ideally (e.g. 3310 for CA)
    kernel parameters C, beta, sigma_x, sigma_y are positive via softplus.
    can add a 'max_lag' cutoff if needed.
    """
    def __init__(self,
                 num_covariates: int,
                 cell_coords,
                 bias: bool = True,
                 init_C: float = 8., #Scale excitation intensity; init big so that gradient descent doesn't ignore it
                 init_beta: float = 0.5, #time damping
                 init_sigma: float = 5., #kilometer scale
                 max_lag: int | None = None, #optional time cutoff (in time step units), could be useful for 6hr resolution?
                 coalesce_duplicates: bool = True #combine events in the same spacetime cell using weighting, making things faster
                 ):
        super().__init__()

        #baseline (Poisson GLM with log-link)
        self.linear = nn.Linear(num_covariates, 1, bias=bias)

        self.psps_linear = nn.Linear(num_covariates, 1, bias=bias)
        self.epss_linear = nn.Linear(num_covariates, 1, bias=bias)

        #raw kernel parameters initialization (before softplus)
        self.raw_C = nn.Parameter(torch.tensor(float(init_C)))
        self.raw_beta = nn.Parameter(torch.tensor(float(init_beta)))
        self.raw_sigma = nn.Parameter(torch.tensor(float(init_sigma)))

        #precompute squared deltas in x and y for all cell pairs
        coords = torch.as_tensor(cell_coords, dtype=torch.float32)  #(C,2)
        dx = coords[:, 0].unsqueeze(0) - coords[:, 0].unsqueeze(1) #(C,C)
        dy = coords[:, 1].unsqueeze(0) - coords[:, 1].unsqueeze(1) #(C,C)
        DX2 = dx.pow(2)
        DY2 = dy.pow(2)
        self.register_buffer("DX2", DX2, persistent=False)
        self.register_buffer("DY2", DY2, persistent=False)

        self.max_lag = max_lag
        self._two_pi = 2.0 * torch.pi
        self._eps = 1e-12

        self.coalesce_duplicates = coalesce_duplicates

    #positive parameter views
    @property
    def C(self):
        return torch.exp(self.raw_C) + self._eps
    @property
    def beta(self):
        return F.softplus(self.raw_beta) + self._eps
    @property
    def sigma(self):
        return F.softplus(self.raw_sigma) + self._eps

    #inhib params flipped negative
    @property
    def psps(self):
        return -F.relu(self.raw_psps) - self._eps

    @property
    def epss(self):
        return -F.relu(self.raw_epss) - self._eps

    #bseline: exp(X beta)
    def _baseline(self, cov: torch.Tensor) -> torch.Tensor:
        if cov.dim() == 4 and cov.shape[0] == 1:
            cov = cov.squeeze(0)
        if cov.dim() != 3:
            raise ValueError(f"cov must be (T,C,p), got {tuple(cov.shape)}")
        T, C, p = cov.shape
        eta = self.linear(cov.view(-1, p)).view(T, C)
        #mild clamp to avoid overflow
        eta = torch.clamp(eta, -20.0, 20.0)
        return torch.exp(eta)

    #excitation component via diffusion kernel
    def _excitation(self, T: int, events_tc: torch.Tensor) -> torch.Tensor:
        if events_tc.dim() == 3 and events_tc.shape[0] == 1:
            events_tc = events_tc.squeeze(0)
        if events_tc.dim() != 2:
            raise ValueError(f"events_tc must be 2D, got {events_tc.dim()}D")
        Ccells = self.DX2.size(0)
        device = self.DX2.device
        lam_exc = torch.zeros(T, Ccells, device=device)

        if events_tc.numel() == 0:
            return lam_exc #if no events return 0s, just in case
        if events_tc.dtype != torch.long:
            events_tc = events_tc.long()
        times = events_tc[:, 0]
        cells = events_tc[:, 1]

        #keep only events in [0, T-1]
        mask = (times >= 0) & (times < T)
        times, cells = times[mask], cells[mask]

        #(optional) coalesce duplicates: same (t,c) but with count weight
        if self.coalesce_duplicates:
            tc = torch.stack([times, cells], dim=1)
            uniq, counts = torch.unique(tc, dim=0, return_counts=True)
            times = uniq[:, 0]
            cells = uniq[:, 1]
            weights = counts.to(torch.float32)
        else:
            weights = torch.ones_like(times, dtype=torch.float32)

        #sort by time so we can slice “past events” by prefix length
        order = torch.argsort(times)
        times, cells, weights = times[order], cells[order], weights[order]

        #prefix counts per day used for m(t) = #events with time < t during loop
        per_day = torch.bincount(times, minlength=T) #(T,)
        csum = torch.cumsum(per_day, dim=0) #(T,)

        #precompute static pieces
        sigma2 = self.sigma * self.sigma
        norm_xy = self._two_pi * sigma2

        for t in range(T):
            m = 0 if t == 0 else int(csum[t - 1].item())
            if m == 0:
                continue #skip 1st day since we have no history

            #optional max lag (days)
            if self.max_lag is not None:
                min_time = max(0, t - self.max_lag)
                start = torch.searchsorted(times[:m], torch.as_tensor(min_time, device=times.device))
            else:
                start = 0

            idx = slice(start, m)
            #events which are influencing the current time step t
            evt_cells = cells[idx]  #(m',) 
            evt_times = times[idx] #(m',)
            evt_w = weights[idx].to(torch.float32)

            #delt (strictly positive)
            dt = (t - evt_times).clamp_min(1).to(torch.float32) #(m',)

            #spatial part for all target cells vs each source cell
            scaled_d2 = ((self.DX2[:, evt_cells] + self.DY2[:, evt_cells]) / sigma2) #(C, m')

            #time factor per event j
            factor_time = torch.exp(-self.beta * dt) * (self.C / (norm_xy * dt)) * evt_w #(m',)

            #exp(- scaled_d2 / (2 dt)) for each event j, (C, m')
            inv_dt = (1.0 / dt).unsqueeze(0) #(1, m')
            factor_space = torch.exp(-0.5 * scaled_d2 * inv_dt) #(C, m')

            lam_exc[t] = (factor_space * factor_time.unsqueeze(0)).sum(dim=1)

        return lam_exc
    
    def _inhibition(self, inhib: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        if inhib.dim() == 4 and inhib.shape[0] == 1:
            inhib = inhib.squeeze(0)
        if inhib.dim() != 3:
            raise ValueError(f"inhib must be 3D, got {inhib.dim()}D")

        if cov.dim() == 4 and cov.shape[0] == 1:
            cov = cov.squeeze(0)
        if cov.dim() != 3:
            raise ValueError(f"cov must be (T,C,p), got {tuple(cov.shape)}")
        
        T, C, p = cov.shape
        #forward through linears
        eta_psps = self.psps_linear(cov.view(-1, p)).view(T, C)
        eta_epss = self.epss_linear(cov.view(-1, p)).view(T, C)

        #masks
        mask_psps = inhib[..., 0]  #(T, C)
        mask_epss = inhib[..., 1]  #(T, C)

        #apply masks
        inhib_psps = eta_psps * mask_psps
        inhib_epss = eta_epss * mask_epss
        return inhib_psps + inhib_epss

    def forward(self,
                cov: torch.Tensor, #(T,C,p)
                inhib: torch.Tensor, #(T,C,2)
                events_tc: torch.Tensor, #(N,2)
                return_parts: bool = False):
        if cov.shape[0] == 1:
            cov=cov.squeeze(0)
        if inhib.shape[0] == 1:
            inhib=inhib.squeeze(0)
        if events_tc.shape[0] == 1:
            events_tc=events_tc.squeeze(0)
        T, C, _ = cov.shape
        lam_base = self._baseline(cov) #(T, C)
        lam_exc = self._excitation(T, events_tc) #(T, C)
        lam_inhib = self._inhibition(inhib, cov)
        if return_parts:
            lam_total = lam_base + lam_exc + lam_inhib
            #We need to recover exactly how much lam_inhib is affecting lam_total. We do not want extra magnitude that sends lam below 0.
            negative_mask = (lam_total < 0).float()
            lam_negative = torch.mul(lam_total, negative_mask) #only keep negative values
            lam_inhib_fixed = lam_inhib + (-1 * lam_negative) #only retains the actual effects inhib had on lambda
            return F.relu(lam_total), {"baseline": lam_base, "excitation": lam_exc, 'inhib': lam_inhib_fixed}
        lam_total = F.relu(lam_base + lam_exc + lam_inhib)
        return lam_total
    
    def loglik(self, cov: torch.Tensor, inhib: torch.Tensor, events: torch.Tensor):
        """
        cov: (T,C,p) tensor of covariates for any given year
        inhib: (T,C,2) tensor of psps and epss inhibitory event indicators
        events: (N_y, 2) tensor of N_y events (all events that occurred in a given year) 
            with their time steps [,0] and grid cell ids [,1]
        returns the log-likelihood of the year.
        """
        lam = self.forward(cov, inhib, events) #(T,C)
        T_ids = events[:, 0]
        C_ids = events[:, 1]
        event_lams = lam[T_ids, C_ids].clamp_min(1e-12) #(N_y) containing lambdas of events, clamped before log so that we dont do log(0)
        logsum = torch.sum(torch.log(event_lams)) #logsum term of loglikelihood
        integral = torch.sum(lam) #integral term of loglikelihood

        #sum of abs parameters in PSPS/EPSS for L1 regularization
        psps_parmsum = sum(p.abs().sum() for p in self.psps_linear.parameters()) 
        epss_parmsum = sum(p.abs().sum() for p in self.epss_linear.parameters())

        #penalize weights in epss/psps:
        return logsum - integral - psps_parmsum - epss_parmsum 

class HawkesDiffusionLinbasePSPSEPSSLinMult(nn.Module):
    """
    Spatiotemporal Hawkes with diffusion-type kernel (Musmeci & Vere-Jones, 1992),
    + linear (log-link) baseline, + additive intensity inhibitions from PSPS/EPSS events

    cov: (T, C, p) covariates (float)
    inhib: (T,C,2) bool indicators for where PSPS or EPSS inhibitory events are active/inactive
    events_tc: (N,2) long/float tensor with columns:
                [time_index, cell_index]
                Times are 0..T-1 (integers). Only events with time < t affect time t.
    return_parts: if True, also returns dict with 'baseline', 'excitation', and 'inhibition'

    Notes:
    cell_coords should be in a metric CRS ideally (e.g. 3310 for CA)
    kernel parameters C, beta, sigma_x, sigma_y are positive via softplus.
    can add a 'max_lag' cutoff if needed.
    """
    def __init__(self,
                 num_covariates: int,
                 cell_coords,
                 inhib_feats = None,
                 bias: bool = True,
                 init_C: float = 8., #Scale excitation intensity; init big so that gradient descent doesn't ignore it
                 init_beta: float = 0.5, #time damping
                 init_sigma: float = 5., #kilometer scale
                 max_lag: int | None = None, #optional time cutoff (in time step units), could be useful for 6hr resolution?
                 coalesce_duplicates: bool = True #combine events in the same spacetime cell using weighting, making things faster
                 ):
        super().__init__()

        #baseline (Poisson GLM with log-link)
        self.linear = nn.Linear(num_covariates, 1, bias=bias)

        #raw kernel parameters initialization (before softplus)
        self.raw_C = nn.Parameter(torch.tensor(float(init_C)))
        self.raw_beta = nn.Parameter(torch.tensor(float(init_beta)))
        self.raw_sigma = nn.Parameter(torch.tensor(float(init_sigma)))

        #precompute squared deltas in x and y for all cell pairs
        coords = torch.as_tensor(cell_coords, dtype=torch.float32)  #(C,2)
        dx = coords[:, 0].unsqueeze(0) - coords[:, 0].unsqueeze(1) #(C,C)
        dy = coords[:, 1].unsqueeze(0) - coords[:, 1].unsqueeze(1) #(C,C)
        DX2 = dx.pow(2)
        DY2 = dy.pow(2)
        self.register_buffer("DX2", DX2, persistent=False)
        self.register_buffer("DY2", DY2, persistent=False)

        self.max_lag = max_lag
        self._two_pi = 2.0 * torch.pi
        self._eps = 1e-12

        self.coalesce_duplicates = coalesce_duplicates

        if inhib_feats is not None:
            self.inhib_feats = inhib_feats
        else:
            self.inhib_feats = list(range(0,num_covariates))
        self.psps_linear = nn.Linear(len(self.inhib_feats), 1, bias=True)
        self.epss_linear = nn.Linear(len(self.inhib_feats), 1, bias=True)

    #positive parameter views
    @property
    def C(self):
        return torch.exp(self.raw_C) + self._eps
    @property
    def beta(self):
        return F.softplus(self.raw_beta) + self._eps
    @property
    def sigma(self):
        return F.softplus(self.raw_sigma) + self._eps

    #inhib params flipped negative
    @property
    def psps(self):
        return -F.relu(self.raw_psps) - self._eps

    @property
    def epss(self):
        return -F.relu(self.raw_epss) - self._eps

    #bseline: exp(X beta)
    def _baseline(self, cov: torch.Tensor) -> torch.Tensor:
        if cov.dim() == 4 and cov.shape[0] == 1:
            cov = cov.squeeze(0)
        if cov.dim() != 3:
            raise ValueError(f"cov must be (T,C,p), got {tuple(cov.shape)}")
        T, C, p = cov.shape
        eta = self.linear(cov.view(-1, p)).view(T, C)
        #mild clamp to avoid overflow
        eta = torch.clamp(eta, -20.0, 20.0)
        return torch.exp(eta)

    #excitation component via diffusion kernel
    def _excitation(self, T: int, events_tc: torch.Tensor) -> torch.Tensor:
        if events_tc.dim() == 3 and events_tc.shape[0] == 1:
            events_tc = events_tc.squeeze(0)
        if events_tc.dim() != 2:
            raise ValueError(f"events_tc must be 2D, got {events_tc.dim()}D")
        Ccells = self.DX2.size(0)
        device = self.DX2.device
        lam_exc = torch.zeros(T, Ccells, device=device)

        if events_tc.numel() == 0:
            return lam_exc #if no events return 0s, just in case
        if events_tc.dtype != torch.long:
            events_tc = events_tc.long()
        times = events_tc[:, 0]
        cells = events_tc[:, 1]

        #keep only events in [0, T-1]
        mask = (times >= 0) & (times < T)
        times, cells = times[mask], cells[mask]

        #(optional) coalesce duplicates: same (t,c) but with count weight
        if self.coalesce_duplicates:
            tc = torch.stack([times, cells], dim=1)
            uniq, counts = torch.unique(tc, dim=0, return_counts=True)
            times = uniq[:, 0]
            cells = uniq[:, 1]
            weights = counts.to(torch.float32)
        else:
            weights = torch.ones_like(times, dtype=torch.float32)

        #sort by time so we can slice “past events” by prefix length
        order = torch.argsort(times)
        times, cells, weights = times[order], cells[order], weights[order]

        #prefix counts per day used for m(t) = #events with time < t during loop
        per_day = torch.bincount(times, minlength=T) #(T,)
        csum = torch.cumsum(per_day, dim=0) #(T,)

        #precompute static pieces
        sigma2 = self.sigma * self.sigma
        norm_xy = self._two_pi * sigma2

        for t in range(T):
            m = 0 if t == 0 else int(csum[t - 1].item())
            if m == 0:
                continue #skip 1st day since we have no history

            #optional max lag (days)
            if self.max_lag is not None:
                min_time = max(0, t - self.max_lag)
                start = torch.searchsorted(times[:m], torch.as_tensor(min_time, device=times.device))
            else:
                start = 0

            idx = slice(start, m)
            #events which are influencing the current time step t
            evt_cells = cells[idx]  #(m',) 
            evt_times = times[idx] #(m',)
            evt_w = weights[idx].to(torch.float32)

            #delt (strictly positive)
            dt = (t - evt_times).clamp_min(1).to(torch.float32) #(m',)

            #spatial part for all target cells vs each source cell
            scaled_d2 = ((self.DX2[:, evt_cells] + self.DY2[:, evt_cells]) / sigma2) #(C, m')

            #time factor per event j
            factor_time = torch.exp(-self.beta * dt) * (self.C / (norm_xy * dt)) * evt_w #(m',)

            #exp(- scaled_d2 / (2 dt)) for each event j, (C, m')
            inv_dt = (1.0 / dt).unsqueeze(0) #(1, m')
            factor_space = torch.exp(-0.5 * scaled_d2 * inv_dt) #(C, m')

            lam_exc[t] = (factor_space * factor_time.unsqueeze(0)).sum(dim=1)

        return lam_exc
    
    def _inhibition(self, inhib: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        if inhib.dim() == 4 and inhib.shape[0] == 1:
            inhib = inhib.squeeze(0)
        if inhib.dim() != 3:
            raise ValueError(f"inhib must be 3D, got {inhib.dim()}D")

        if cov.dim() == 4 and cov.shape[0] == 1:
            cov = cov.squeeze(0)
        if cov.dim() != 3:
            raise ValueError(f"cov must be (T,C,p), got {tuple(cov.shape)}")
        
        T, C, p = cov.shape
        #forward through linears
        cov_inhib = cov[..., self.inhib_feats]
        eta_psps = torch.sigmoid(self.psps_linear(cov_inhib.view(-1, len(self.inhib_feats))).view(T, C))
        eta_epss = torch.sigmoid(self.epss_linear(cov_inhib.view(-1, len(self.inhib_feats))).view(T, C))

        #masks
        mask_psps = inhib[..., 0]  #(T, C)
        mask_epss = inhib[..., 1]  #(T, C)

        #apply masks
        inhib_psps = torch.where(mask_psps.bool(), eta_psps, torch.ones_like(eta_psps))
        inhib_epss = torch.where(mask_epss.bool(), eta_epss, torch.ones_like(eta_epss))
        
        return inhib_psps * inhib_epss

    def forward(self,
                cov: torch.Tensor, #(T,C,p)
                inhib: torch.Tensor, #(T,C,2)
                events_tc: torch.Tensor, #(N,2)
                return_parts: bool = False):
        if cov.shape[0] == 1:
            cov=cov.squeeze(0)
        if inhib.shape[0] == 1:
            inhib=inhib.squeeze(0)
        if events_tc.shape[0] == 1:
            events_tc=events_tc.squeeze(0)
        T, C, _ = cov.shape
        lam_base = self._baseline(cov) #(T, C)
        lam_exc = self._excitation(T, events_tc) #(T, C)
        lam_inhib = self._inhibition(inhib, cov)
        if return_parts:
            lam_total = (lam_base + lam_exc) * lam_inhib
            #We need to recover exactly how much lam_inhib is reducing lam_total. We do not want extra magnitude that sends lam below 0.
            lam_decreased = lam_base * -(1 - lam_inhib)
            return F.relu(lam_total), {"baseline": lam_base, "excitation": lam_exc, 'inhib': lam_decreased}
        return (lam_base + lam_exc) * lam_inhib

    def loglik(self, cov: torch.Tensor, inhib: torch.Tensor, events: torch.Tensor):
        """
        cov: (T,C,p) tensor of covariates for any given year
        inhib: (T,C,2) tensor of psps and epss inhibitory event indicators
        events: (N_y, 2) tensor of N_y events (all events that occurred in a given year) 
            with their time steps [,0] and grid cell ids [,1]
        returns the log-likelihood of the year.
        """
        lam = self.forward(cov, inhib, events) #(T,C)
        T_ids = events[:, 0]
        C_ids = events[:, 1]
        event_lams = lam[T_ids, C_ids].clamp_min(1e-12) #(N_y) containing lambdas of events, clamped before log so that we dont do log(0)
        logsum = torch.sum(torch.log(event_lams)) #logsum term of loglikelihood
        integral = torch.sum(lam) #integral term of loglikelihood

        return logsum - integral

class HawkesDiffusionFlatbase(nn.Module):
    """
    Spatiotemporal Hawkes with diffusion-type kernel (Musmeci & Vere-Jones, 1992)
    + flat baseline.

    cov: (T, C, p) covariates (float)
    events_tc: (N,2) long/float tensor with columns:
                [time_index, cell_index]
                Times are 0..T-1 (integers). Only events with time < t affect time t.
    return_parts: if True, also returns dict with 'baseline' and 'excitation'

    Notes:
    cell_coords should be in a metric CRS ideally (e.g. 3310 for CA)
    kernel parameters C, beta, sigma_x, sigma_y are positive via softplus.
    can add a `max_lag` cutoff if needed.
    """
    def __init__(self,
                 cell_coords,
                 init_lam: float = 0.5,
                 init_C: float = 8., #Scale excitation intensity; init big so that gradient descent doesn't ignore it
                 init_beta: float = 0.5, #time damping
                 init_sigma: float = 5., #kilometer scale
                 max_lag: int | None = None, #optional time cutoff (in time step units), could be useful for 6hr resolution?
                 coalesce_duplicates: bool = True #combine events in the same spacetime cell using weighting, making things faster
                 ):
        super().__init__()

        #baseline (Poisson GLM with log-link)
        self.raw_lam = nn.Parameter(torch.tensor(float(init_lam)))

        #raw kernel parameters initialization (before softplus)
        self.raw_C = nn.Parameter(torch.tensor(float(init_C)))
        self.raw_beta = nn.Parameter(torch.tensor(float(init_beta)))
        self.raw_sigma = nn.Parameter(torch.tensor(float(init_sigma)))

        #precompute squared deltas in x and y for all cell pairs
        coords = torch.as_tensor(cell_coords, dtype=torch.float32)  #(C,2)
        dx = coords[:, 0].unsqueeze(0) - coords[:, 0].unsqueeze(1) #(C,C)
        dy = coords[:, 1].unsqueeze(0) - coords[:, 1].unsqueeze(1) #(C,C)
        DX2 = dx.pow(2)
        DY2 = dy.pow(2)
        self.register_buffer("DX2", DX2, persistent=False)
        self.register_buffer("DY2", DY2, persistent=False)

        self.max_lag = max_lag
        self._two_pi = 2.0 * torch.pi
        self._eps = 1e-12

        self.coalesce_duplicates = coalesce_duplicates

    #positive parameter views
    @property
    def base_lam(self):
        return F.softplus(self.raw_lam) + self._eps
    @property
    def C(self):
        return torch.exp(self.raw_C) + self._eps #here we use exp instead of softplus because we expect scale parameter to be big but still adjustable with autograd
    @property
    def beta(self):
        return F.softplus(self.raw_beta) + self._eps
    @property
    def sigma(self):
        return F.softplus(self.raw_sigma) + self._eps

    #bseline: base_lam
    def _baseline(self, cov: torch.Tensor) -> torch.Tensor:
        if cov.dim() == 4 and cov.shape[0] == 1:
            cov = cov.squeeze(0)
        if cov.dim() != 3:
            raise ValueError(f"cov must be (T,C,p), got {tuple(cov.shape)}")
        T, C, p = cov.shape
        return self.base_lam.expand((T,C)).to(self.DX2.device)

    #excitation component via diffusion kernel
    def _excitation(self, T: int, events_tc: torch.Tensor) -> torch.Tensor:
        if events_tc.dim() == 3 and events_tc.shape[0] == 1:
            events_tc = events_tc.squeeze(0)
        if events_tc.dim() != 2:
            raise ValueError(f"events_tc must be 2D, got {events_tc.dim()}D")
        Ccells = self.DX2.size(0)
        device = self.DX2.device
        lam_exc = torch.zeros(T, Ccells, device=device)

        if events_tc.numel() == 0:
            return lam_exc #if no events return 0s, just in case
        if events_tc.dtype != torch.long:
            events_tc = events_tc.long()
        times = events_tc[:, 0]
        cells = events_tc[:, 1]

        #keep only events in [0, T-1]
        mask = (times >= 0) & (times < T)
        times, cells = times[mask], cells[mask]

        #(optional) coalesce duplicates: same (t,c) but with count weight
        if self.coalesce_duplicates:
            tc = torch.stack([times, cells], dim=1)
            uniq, counts = torch.unique(tc, dim=0, return_counts=True)
            times = uniq[:, 0]
            cells = uniq[:, 1]
            weights = counts.to(torch.float32)
        else:
            weights = torch.ones_like(times, dtype=torch.float32)

        #sort by time so we can slice “past events” by prefix length
        order = torch.argsort(times)
        times, cells, weights = times[order], cells[order], weights[order]

        #prefix counts per day used for m(t) = #events with time < t during loop
        per_day = torch.bincount(times, minlength=T) #(T,)
        csum = torch.cumsum(per_day, dim=0) #(T,)

        #precompute static pieces
        sigma2 = self.sigma * self.sigma
        norm_xy = self._two_pi * sigma2

        for t in range(T):
            m = 0 if t == 0 else int(csum[t - 1].item())
            if m == 0:
                continue #skip 1st day since we have no history

            #optional max lag (days)
            if self.max_lag is not None:
                min_time = max(0, t - self.max_lag)
                start = torch.searchsorted(times[:m], torch.as_tensor(min_time, device=times.device))
            else:
                start = 0

            idx = slice(start, m)
            #events which are influencing the current time step t
            evt_cells = cells[idx]  #(m',) 
            evt_times = times[idx] #(m',)
            evt_w = weights[idx].to(torch.float32)

            #delt (strictly positive)
            dt = (t - evt_times).clamp_min(1).to(torch.float32) #(m',)

            #spatial part for all target cells vs each source cell
            #scaled_d2[c, j] = DX2[c, src_j]/sigmax^2 + DY2[c, src_j]/sigmay^2
            scaled_d2 = ((self.DX2[:, evt_cells] + self.DY2[:, evt_cells]) / sigma2) #(C, m')

            #time factor per event j
            factor_time = torch.exp(-self.beta * dt) * (self.C / (norm_xy * dt)) * evt_w #(m',)

            #exp(- scaled_d2 / (2 dt)) for each event j, (C, m')
            inv_dt = (1.0 / dt).unsqueeze(0) #(1, m')
            factor_space = torch.exp(-0.5 * scaled_d2 * inv_dt) #(C, m')

            lam_exc[t] = (factor_space * factor_time.unsqueeze(0)).sum(dim=1)

        return lam_exc

    def forward(self,
                cov: torch.Tensor, #(T,C,p)
                events_tc: torch.Tensor, #(N,2)
                return_parts: bool = False):
        T, C, _ = cov.shape
        lam_base = self._baseline(cov) #(T, C)
        lam_exc = self._excitation(T, events_tc) #(T, C)
        lam_total = lam_base + lam_exc
        if return_parts:
            return lam_total, {"baseline": lam_base, "excitation": lam_exc}
        return lam_total
    
    def loglik(self, cov: torch.Tensor, events: torch.Tensor):
        """
        cov: (T,C,p) tensor of covariates for any given year
        events: (N_y, 2) tensor of N_y events (all events that occurred in a given year) 
            with their time steps [,0] and grid cell ids [,1]
        returns the log-likelihood of the year.
        """
        lam = self.forward(cov, events) #(T,C)
        T_ids = events[:, 0]
        C_ids = events[:, 1]
        event_lams = lam[T_ids, C_ids].clamp_min(1e-12) #(N_y) containing lambdas of events, clamped before log so that we dont do log(0)
        logsum = torch.sum(torch.log(event_lams)) #logsum term of loglikelihood
        integral = torch.sum(lam) #integral term of loglikelihood
        return logsum - integral


