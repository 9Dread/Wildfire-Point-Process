

import torch
import torch.nn as nn
import torch.nn.functional as F
from Functions import WildfireDataset
from torch.utils.data import DataLoader
import pandas as pd
from copy import deepcopy

def train_model(model, optimizer, loader, max_epochs, device, delta_loss = 0.01, improve_epochs = 10, print_iter = 10):
    """
    Runs a training loop with early stopping.
    model: one of the model classes initialized; must have loglik function
    optimizer: an initialized optimizer. 
    loader: a data loader with the training set.
    max_epochs: the maximum number of epochs to run.
    device: the device to train on.
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
        for cov, events in loader:
            #move to device, squeeze batch‐dim
            cov = cov.squeeze(0).to(device)
            events = events.squeeze(0).to(device)
            
            #negative log likelihood
            nll = -model.loglik(cov, events)

            #backward & step
            optimizer.zero_grad()
            nll.backward()
            optimizer.step()

            total_nll += nll.item()
        avg_nll = total_nll / len(loader)

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

        if (epoch - 1) % print_iter == 0:
            print(f"Epoch {epoch:02d} — Avg NLL: {avg_nll:.4f}")

    print(f"{max_epochs} epochs completed with most recent average negative log likelihood {avg_nll}.")
    print("This makes the total log likelihood over the training set: " + str(-(avg_nll * len(loader))))
    return -(avg_nll * len(loader))

def cross_validation(model, lr, covars, events, device, delta_loss = 0.01, print_iter = 10, save_results=True, save_path = None, train_on_all=True):
    """
    cross validation routine. the goal is to estimate how well a model class actually describes the data. to do this,
    given lists of data for n years, for each year we fit a model on data from all other years and then compute the
    log likelihood of the model for the left out year. finally, we can estimate how well a model class describes
    the data by taking the mean of these out-of-sample likelihoods.

    model: one of the model classes initialized
    lr: the learning rate passed to the Adam optimizer
    covars: a list of tensors of shape (T_y, C, p); spacetime grid of covariates
    events: a list of tensors of shape (N_y, 2) containing the time and space indices of each event occurrence
    device: a pytorch device
    delta_loss: passed to train_model
    print_iter: passed to train_model
    save_results: whether or not to save the results of the analysis. If true, will save to csv in save_path or append
        to it if it already exists.
    save_path: the path to save the csv of the results of the cross validation to. must be provided if save_results is true.
    train_on_all: whether or not to run training on the *entire* dataset after cross validation is complete. useful
        if you want to additionally save the training likelihood of the model class, not just the validation likelihood. 
    returns a list of the recorded out-of-sample log likelihoods of each year.
    """
    if save_results:
        assert save_path is not None, "If save_results=True, make sure save_path provides a csv save path! This can be an existing csv to append to or a path to create one."
    oos_list = []
    for i in range(0, len(covars)):
        #make list of data excluding the current year
        covars_list = [covars[j] for j in range(0, len(covars)) if j != i]
        events_list = [events[j] for j in range(0, len(covars)) if j != i]
        #make data loader
        dataset = WildfireDataset(covars_list, events_list)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        model_copy = deepcopy(model).to(device)
        optimizer = torch.optim.Adam(model_copy.parameters(), lr=lr)
        #train loop
        #just keep running until training converges:
        training_ll = train_model(model_copy, optimizer, loader, 1000000, device, delta_loss, print_iter) #returns total loglik over training set after training complete

        #evaluate loglik on unseen data:
        loglik = model_copy.loglik(covars[i], events[i])
        oos_list.append([training_ll, loglik])
    
    #TODO: SAVING AND TRAIN_ON_ALL
    return(oos_list)


class PoissonLinearIntensity(nn.Module):
    """
    Poisson-GLM intensity model with log-link (the canonical link function)

    Input: cov of shape (T, C, p)
    Output: lam of shape (T, C)

    """
    def __init__(self, num_covariates: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(num_covariates, 1, bias=bias)

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
        T_ids = events[:, 0]
        C_ids = events[:, 1]
        event_lams = lam[T_ids, C_ids] #(N_y) containing lambdas of events
        logsum = torch.sum(torch.log(event_lams)) #logsum term of loglikelihood
        integral = torch.sum(lam) #integral term of loglikelihood
        return(logsum - integral)

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
        return(logsum - integral)

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
                 max_lag: int | None = None,   #optional time cutoff (days)
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

        # (optional) coalesce duplicates: same (t,c) but with count weight
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
        per_day = torch.bincount(times, minlength=T)      # (T,)
        csum = torch.cumsum(per_day, dim=0)               # (T,)

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
        return(logsum - integral)
