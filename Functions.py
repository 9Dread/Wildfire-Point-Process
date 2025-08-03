# key Python dependencies
import torch
import arrow
import itertools
import numpy as np
import torch.optim as optim
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

import xarray as xr
import pandas as pd
import geopandas as gpd
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

class StdDiffusionKernel:
    def __init__(self, C=1.0, beta=1.0, sigma_x=1.0, sigma_y=1.0):
        self.C = C
        self.beta = beta
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.norm_factor = 1.0 / (2 * math.pi * sigma_x * sigma_y)

    def __call__(self, delta_s, delta_t):
        """
        delta_s: (..., 2), delta_t: (...,)
        """
        eps = 1e-6  # for numerical stability
        delta_t = delta_t.clamp(min=eps)  # ensure t > 0

        dx = delta_s[..., 0]
        dy = delta_s[..., 1]

        exponent = -0.5 * (
            (dx**2 / (self.sigma_x**2)) +
            (dy**2 / (self.sigma_y**2))
        ) / delta_t

        temporal_decay = torch.exp(-self.beta * delta_t)
        spatial_kernel = torch.exp(exponent)

        result = temporal_decay * spatial_kernel * self.C * self.norm_factor / delta_t
        return result

    def integrate(self, t_j, T):
        """
        Computes ∫_{t_j}^T ∫_{R^2} ν(x, t | x_j, t_j) dx dt
        Assumes spatial domain is all of R^2 (Gaussian normalization = 1)
        """
        eps = 1e-6
        t_diff = torch.clamp(T - t_j, min=eps)
        if self.beta == 0:
            return self.C * torch.log(t_diff)
        else:
            return self.C / self.beta * (1 - torch.exp(-self.beta * t_diff))



class BasePointProcess(torch.nn.Module):
    """
    Point Process Base Class
    """
    @abstractmethod
    def __init__(self, T, S, data_dim, grid_cells, int_config = None):
        """
        Args:
        - T:             time horizon. e.g. (0, 1)
        - S:             bounded space for marks. e.g. a two dimensional box region [(0, 1), (0, 1)]
        - grid_cells:    [num_cells, 2] Tensor of centroid coordinates
        - data_dim:      dimension of input data
        - int_config: numerical integral configuration (mc or grid; grid works well on low-dimensional mark spaces). Should receive tuple of ('mc', nsamples) or ('grid', int_res), where nsamples is the number of monte carlo samples and int_res is the resolution of the grid integral approximator.
        """
        super(BasePointProcess, self).__init__()
        #configuration
        self.data_dim      = data_dim
        self.T             = T #time horizon. e.g. (0, 1)
        self.S             = S #bounded space for marks e.g. a two dimensional box region [(0, 1), (0, 1)]
        self.int_config = int_config
        self.grid_cells = grid_cells
        assert len(S) + 3 == self.data_dim, "Invalid dimension"
        assert (int_config[0] == "mc") | (int_config[0] == "grid"), "Invalid integral approximator (choose 'mc' or 'grid')" #make this better check, im lazy

        #numerical likelihood integral preparation
        if int_config[0] == "grid":
            self.int_res  = int_config[1] #resolution of the integral. each non-spatial mark is gridded to this resolution to estimate the integral of the intensity. 
            self.tt       = torch.FloatTensor(np.linspace(self.T[0], self.T[1], int_config[1]))  #[ int_res ], evenly-spaced time grid of int_res points in the time horizon
            self.ss       = [ np.linspace(S_k[0], S_k[1], int_config[1]) for S_k in self.S ]     #[ data_dim - 1, in_res ], evenly spaced mark grid
            #spatio-temporal coordinates that need to be evaluated
            self.t_coords = torch.ones((int_config[1] ** (data_dim - 1), 1))                     #[ int_res^(data_dim - 1), 1 ]
            self.s_coords = torch.FloatTensor(np.array(list(itertools.product(*self.ss)))) #[ int_res^(data_dim - 1), data_dim - 1 ]
            #unit volume
            self.unit_vol = np.prod([ S_k[1] - S_k[0] for S_k in self.S ] + [ self.T[1] - self.T[0] ] + [ self.grid_cells.shape[0] ]) / (self.int_res) ** self.data_dim
        if int_config[0] == "mc":
            self.nsamples  = int_config[1] #number of monte carlo samples to draw 

    def grid_integral(self, X):
        """
        return conditional intensity evaluation at grid points, the numerical 
        integral can be further calculated by summing up these evaluations and 
        scaling by the unit volume.
        """
        batch_size, seq_len, _ = X.shape
        integral = []
        for t in self.tt:
            # all possible points at time t (x_t) 
            t_coord = self.t_coords * t
            xt      = torch.cat([t_coord, self.s_coords], 1) # [ int_res^(data_dim - 1), data_dim ] 
            xt      = xt\
                .unsqueeze_(0)\
                .repeat(batch_size, 1, 1)\
                .reshape(-1, self.data_dim)                  # [ batch_size * int_res^(data_dim - 1), data_dim ]
            # history points before time t (H_t)
            mask = ((X[:, :, 0].clone() <= t) * (X[:, :, 0].clone() > 0))\
                .unsqueeze_(-1)\
                .repeat(1, 1, self.data_dim)                 # [ batch_size, seq_len, data_dim ]
            ht   = X * mask                                  # [ batch_size, seq_len, data_dim ]
            ht   = ht\
                .unsqueeze_(1)\
                .repeat(1, self.int_res ** (self.data_dim - 1), 1, 1)\
                .reshape(-1, seq_len, self.data_dim)         # [ batch_size * int_res^(data_dim - 1), seq_len, data_dim ]
            # lambda and integral 
            lams = torch.nn.functional.softplus(self.cond_lambda(xt, ht))\
                .reshape(batch_size, -1)                     # [ batch_size, int_res^(data_dim - 1) ]
            integral.append(lams)                            
        # NOTE: second dimension is time, third dimension is mark space
        integral = torch.stack(integral, 1)                  # [ batch_size, int_res, int_res^(data_dim - 1) ]
        return integral

    def monte_carlo_integral(self, X):
        """
        Monte Carlo integral of the intensity

        Args:
        - X: [batch_size, seq_len, data_dim]
        - nsamples: number of Monte Carlo samples

        Returns:
        - integral: [batch_size] integral estimate per sequence
        """
        batch_size, seq_len, data_dim = X.shape
        spatial_dim = 2
        mark_dim = data_dim - 1 - spatial_dim  # exclude time and spatial centroid

        # 1. Sample time uniformly
        t_samples = torch.FloatTensor(batch_size, self.nsamples).uniform_(self.T[0], self.T[1])  # [B, N]

        # 2. Sample discrete spatial cell indices and map to centroids
        num_cells = self.grid_cells.shape[0]
        cell_indices = torch.randint(0, num_cells, (batch_size, self.nsamples))                # [B, N]
        spatial_samples = self.grid_cells[cell_indices]                                   # [B, N, 2]

        # 3. Sample continuous marks uniformly over their bounding boxes
        mark_samples = []
        for lower, upper in self.S:
            m = torch.FloatTensor(batch_size, self.nsamples).uniform_(lower, upper)           # [B, N]
            mark_samples.append(m)
        mark_samples = torch.stack(mark_samples, dim=-1)                                  # [B, N, mark_dim]

        # 4. Construct x_samples = [t, spatial_centroid, mark]
        x_samples = torch.cat([
        t_samples.unsqueeze(-1),      # [B, N, 1]
        spatial_samples,              # [B, N, 2]
        mark_samples                  # [B, N, mark_dim]
        ], dim=-1)                                                             # [B, N, data_dim]
        x_samples = x_samples.view(-1, data_dim)                              # [B*N, data_dim]

        # 5. Build H_t for each sample: events with time < t_sample
        X_expanded = X.unsqueeze(2).repeat(1, 1, self.nsamples, 1)                 # [B, seq, N, data_dim]
        t_mask = (X_expanded[:, :, :, 0] < t_samples.unsqueeze(1)).float()   # [B, seq, N]
        mask = t_mask.unsqueeze(-1)                                          # [B, seq, N, 1]
        H_t = X_expanded * mask                                              # zero-out future events
        H_t = H_t.permute(0, 2, 1, 3).reshape(-1, seq_len, data_dim)         # [B*N, seq_len, data_dim]

        # 6. Evaluate conditional intensity
        lams = torch.nn.functional.softplus(self.cond_lambda(x_samples, H_t))  # [B*N]
        lams = lams.view(batch_size, self.nsamples)                                 # [B, N]

        # 7. Compute total volume of domain: |T| × |G| × vol(S)
        T_range = self.T[1] - self.T[0]
        G_size = num_cells
        S_volume = torch.tensor([upper - lower for lower, upper in self.S]).prod()
        volume = T_range * G_size * S_volume

        # 8. Monte Carlo estimate
        integral = volume * lams.mean(dim=1)  # [batch_size]
        return integral

    def cond_lambda(self, xi, hti):
        """
        return conditional intensity given x
        Args:
        - xi:   current i-th point       [ batch_size, data_dim ]
        - hti:  history points before ti [ batch_size, seq_len, data_dim ]
        Return:
        - lami: i-th lambda              [ batch_size ]
        """
        # if length of the history is zero
        if hti.size()[0] == 0:
            return self.mu()
        # otherwise treat zero in the time (the first) dimension as invalid points
        batch_size, seq_len, _ = hti.shape
        mask = hti[:, :, 0].clone() > 0                                          # [ batch_size, seq_len ]
        xi   = xi.unsqueeze_(1).repeat(1, seq_len, 1).reshape(-1, self.data_dim) # [ batch_size * seq_len, data_dim ]
        hti  = hti.reshape(-1, self.data_dim)                                    # [ batch_size * seq_len, data_dim ]
        K    = self.kernel(xi, hti).reshape(batch_size, seq_len)                 # [ batch_size, seq_len ]
        K    = K * mask                                                          # [ batch_size, seq_len ]
        lami = K.sum(1) + self.mu()                                              # [ batch_size ]
        return lami

    def log_likelihood(self, X):
        """
        return log-likelihood given sequence X
        Args:
        - X:      input points sequence [ batch_size, seq_len, data_dim ]
        Return:
        - lams:   sequence of lambda    [ batch_size, seq_len ]
        - loglik: log-likelihood        [ batch_size ]
        """
        batch_size, seq_len, _ = X.shape
        lams     = [
            torch.nn.functional.softplus(self.cond_lambda(
                X[:, i, :].clone(), 
                X[:, :i, :].clone())) + 1e-5
            for i in range(seq_len) ]
        lams     = torch.stack(lams, dim=1)                                   # [ batch_size, seq_len ]
        # log-likelihood
        mask     = X[:, :, 0] > 0                                             # [ batch_size, seq_len ]
        sumlog   = torch.log(lams) * mask                                     # [ batch_size, seq_len ]
        if self.int_config[0] == "grid":
            integral = self.grid_integral(X)                                 # [ batch_size, int_res, int_res^(data_dim - 1) ]
            loglik = sumlog.sum(1) - integral.sum(-1).sum(-1) * self.unit_vol # [ batch_size ]
        else: 
            integral = self.monte_carlo_integral(X)  # [batch_size]           
            loglik = sumlog.sum(1) - integral # [ batch_size ]
            print("Log-sum term:", sumlog.sum(1).mean().item())
            print("MC integral :", integral.mean().item())
        return lams, loglik

    @abstractmethod
    def mu(self):
        """
        return base intensity
        """
        raise NotImplementedError()

    @abstractmethod
    def forward(self, X):
        """
        custom forward function returning conditional intensities and corresponding log-likelihood
        """
        # return conditional intensities and corresponding log-likelihood
        return self.log_likelihood(X)

class DeepBasisPointProcess(BasePointProcess):
    """
    Point Process with Deep NN basis
    """
    def __init__(self, 
                 T, S, mu, 
                 n_basis, basis_dim, data_dim, 
                 grid_cells, int_config, 
                 init_gain=5e-1, init_bias=1e-3, init_std=1.,
                 nn_width=5):
        """
        Args:
        - T:             time horizon. e.g. (0, 1)
        - S:             bounded space for marks. e.g. a two dimensional box region [(0, 1), (0, 1)]
        - n_basis:       number of basis functions
        - basis_dim:     dimension of basis function
        - data_dim:      dimension of input data
        - numerical_int: numerical integral flag
        - int_res:       numerical integral resolution
        - nn_width:      the width of each layer in kernel basis NN
        """
        super(DeepBasisPointProcess, self).__init__(T, S, data_dim, grid_cells, int_config)
        # configuration
        self.n_basis   = n_basis
        self.basis_dim = basis_dim
        self._mu       = mu
        # deep nn basis kernel
        self.kernel    = DeepBasisKernel(n_basis, data_dim, basis_dim, 
                                         init_gain=init_gain, init_bias=init_bias, init_std=init_std,
                                         nn_width=nn_width)
    
    def mu(self):
        """
        return base intensity
        """
        return self._mu

    def forward(self, X, n_sampled_fouriers=200):
        """
        custom forward function returning conditional intensities and corresponding log-likelihood
        """
        # return conditional intensities and corresponding log-likelihood
        return self.log_likelihood(X)



class SpatioTemporalDeepBasisPointProcess(BasePointProcess):
    """
    Spatio-temporal Point Process with Deep NN basis
    """
    def __init__(self, 
                 T, S, 
                 n_basis, basis_dim, data_dim, grid_cells, int_config,
                 init_gain=5e-1, init_bias=1e-3, init_std=1.,
                    nn_width=5, beta=1):
        """
        See `DeepBasisPointProcess` for more details
        """
        super(SpatioTemporalDeepBasisPointProcess, self).__init__(T, S, data_dim, grid_cells = grid_cells, int_config = int_config)
        # configuration
        self.n_basis   = n_basis
        self.basis_dim = basis_dim
        # spatio-temporal deep nn basis kernel
        self.kernel    = SpatioTemporalDeepBasisKernel(n_basis, data_dim, basis_dim, 
                                                       init_gain=init_gain, init_bias=init_bias, init_std=init_std,
                                                       nn_width=nn_width, beta=beta)
    
    def mu(self):
        """
        return base intensity
        """
        return 1.

    def forward(self, X, n_sampled_fouriers=200):
        """
        custom forward function returning conditional intensities and corresponding log-likelihood
        """
        # return conditional intensities and corresponding log-likelihood
        return self.log_likelihood(X)

class DeepNetworkBasis(torch.nn.Module):
    """
    Deep Neural Network Basis Kernel

    This class directly models the kernel-induced feature mapping by a deep 
    neural network.
    """
    def __init__(self, data_dim, basis_dim, 
                 init_gain=5e-1, init_bias=1e-3, nn_width=5):
        """
        Args:
        - data_dim:  dimension of input data point
        - basis_dim: dimension of basis function
        - nn_width:  the width of each layer in NN
        """
        super(DeepNetworkBasis, self).__init__()
        # configurations
        self.data_dim  = data_dim
        self.basis_dim = basis_dim
        # init parameters for net
        self.init_gain   = init_gain
        self.init_bias   = init_bias
        # network for basis function
        self.net = torch.nn.Sequential(
            torch.nn.Linear(data_dim, nn_width),  # [ data_dim, n_hidden_nodes ]
            torch.nn.Softplus(), 
            torch.nn.Linear(nn_width, nn_width),  # [ n_hidden_nodes, n_hidden_nodes ]
            torch.nn.Softplus(), 
            torch.nn.Linear(nn_width, nn_width),  # [ n_hidden_nodes, n_hidden_nodes ]
            torch.nn.Softplus(), 
            torch.nn.Linear(nn_width, basis_dim), # [ n_hidden_nodes, basis_dim ]
            torch.nn.Sigmoid())
        self.net.apply(self.init_weights)

    def init_weights(self, m):
        """
        initialize weight matrices in network
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight, gain=self.init_gain)
            m.bias.data.fill_(self.init_bias)

    def forward(self, x):
        """
        customized forward function returning basis function evaluated at x
        with size [ batch_size, data_dim ]
        """
        return self.net(x) * 2 - 1         # [ batch_size, basis_dim ]



class DeepBasisKernel(torch.nn.Module):
    """
    Deep Basis Kernel
    """
    def __init__(self, 
                 n_basis, data_dim, basis_dim, 
                 init_gain=5e-1, init_bias=1e-3, init_std=1.,
                 nn_width=5):
        """
        Arg:
        - n_basis:   number of basis functions
        - data_dim:  dimension of input data point
        - basis_dim: dimension of basis function
        - nn_width:  the width of each layer in basis NN
        """
        super(DeepBasisKernel, self).__init__()
        # configurations
        self.n_basis   = n_basis
        self.data_dim  = data_dim
        self.basis_dim = basis_dim
        # set of basis functions and corresponding weights
        self.xbasiss   = torch.nn.ModuleList([])
        self.ybasiss   = torch.nn.ModuleList([])
        self.weights   = torch.nn.ParameterList([])
        for i in range(n_basis):
            self.xbasiss.append(DeepNetworkBasis(data_dim, basis_dim, 
                                                 init_gain=init_gain, init_bias=init_bias,
                                                 nn_width=nn_width))
            self.ybasiss.append(DeepNetworkBasis(data_dim, basis_dim, 
                                                 init_gain=init_gain, init_bias=init_bias,
                                                 nn_width=nn_width))
            self.weights.append(torch.nn.Parameter(torch.empty(1).normal_(mean=0,std=init_std), requires_grad=True))
            
    def forward(self, x, y):
        """
        customized forward function returning kernel evaluation at x and y with 
        size [ batch_size, batch_size ], where
        - x: the first input with size  [ batch_size, data_dim ]
        - y: the second input with size [ batch_size, data_dim ] 
        """
        K = []
        for weight, xbasis, ybasis in zip(self.weights, self.xbasiss, self.ybasiss):
            xbasis_func = xbasis(x)                                   # [ batch_size, basis_dim ]
            ybasis_func = ybasis(y)                                   # [ batch_size, basis_dim ]
            weight      = torch.nn.functional.softplus(weight)        # scalar
            ki          = (weight * xbasis_func * ybasis_func).sum(1) # [ batch_size ]
            K.append(ki)
        K = torch.stack(K, 1).sum(1)
        return K



class SpatioTemporalDeepBasisKernel(torch.nn.Module):
    """
    Spatio-temporal Deep Basis Kernel
    """
    def __init__(self, 
                 n_basis, data_dim, basis_dim, 
                 init_gain=5e-1, init_bias=1e-3, init_std=1.,
                 nn_width=5, beta=1):
        """
        Arg:
        - n_basis:   number of basis functions
        - data_dim:  dimension of input data point
        - basis_dim: dimension of basis function
        - nn_width:  the width of each layer in basis NN
        """
        super(SpatioTemporalDeepBasisKernel, self).__init__()
        # decoupled kernels
        self.spatialkernel  = DeepBasisKernel(n_basis, data_dim-1, basis_dim, 
                                              init_gain=init_gain, init_bias=init_bias, init_std=init_std,
                                              nn_width=nn_width)
        # self.temporalkernel = ExponentialDecayingKernel(beta)
        self.beta = beta

    def temporalkernel(self, x, y):
        return self.beta * torch.exp(- self.beta * (x - y))
      
    def forward(self, x, y):
        """
        customized forward function returning kernel evaluation at x and y with 
        size [ batch_size, batch_size ], where
        - x: the first input with size  [ batch_size, data_dim ]
        - y: the second input with size [ batch_size, data_dim ] 
        """
        xt, yt = x[:, 0].clone(), y[:, 0].clone()   # [ batch_size ]
        xs, ys = x[:, 1:].clone(), y[:, 1:].clone() # [ batch_size, data_dim - 1 ]
        tval   = self.temporalkernel(xt, yt)        # [ batch_size ]
        sval   = self.spatialkernel(xs, ys)         # [ batch_size ]
        return tval * sval                          # [ batch_size ]



#Now get empirical quantiles (1,99) for each covariate from the entire year's raster so that we can normalize them:
def get_cov_bounds(raster_nc, covariates):
    #Load raster
    ds = xr.open_dataset(raster_nc)

    pctiles = []
    for i in range(len(covariates)):
        var = covariates[i]
        values = ds[var].values.flatten()
        values = values[~np.isnan(values)]  #remove NaNs
        p1, p99 = np.percentile(values, [1, 99])
        pctiles.append((p1, p99))
    return pctiles
pctiles = get_cov_bounds(raster_nc, covs)
#Now apply these normalization bounds to the marks
for i in range(len(pctiles)):
    data[:,:,3+i] = (data[:,:,3+i] - pctiles[i][0]) / (pctiles[i][1]-pctiles[i][0])





seed = 3
rootpath = "C:\\Users\\dread\\Downloads\\wildfire"
# training configurations
batch_size = 4
num_epochs = 40
lr         = 1e-2
modelname  = "marktemporal-deepnnbasis_learned_model-%d" % seed





def train(model,
          train_loader,
          test_data,
          ts, T, S, ngrid, 
          modelname="pp", 
          num_epochs=10, 
          lr=1e-4, 
          print_iter=10):
    """training procedure"""
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for i in range(num_epochs):
        epoch_loss = 0
        for j, data in enumerate(train_loader):

            optimizer.zero_grad()
            X_batch   = data[0]
            _, loglik = model(X_batch)
            loss      = - loglik.mean()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss
            if j % print_iter == 0:
                print("[%s] Epoch : %d,\titer : %d,\tloss : %.5e" % (arrow.now(), i, j, loss / print_iter))
                torch.save(model.state_dict(), "%s/saved_models/%s.pth" % (rootpath, modelname))
        
        print("[%s] Epoch : %d,\tTotal loss : %.5e" % (arrow.now(), i, epoch_loss))

train(init_model, train_loader, ts=[1, 2, 3], T=T, S=S[0], ngrid=50, test_data=test_seq, 
        modelname=modelname, num_epochs=num_epochs, lr=lr, print_iter=1)
