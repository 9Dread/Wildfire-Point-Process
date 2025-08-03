import torch
import math 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class StdDiffusionKernel:
    def __init__(self, C=1.0, beta=1.0, sigma_x=1.0, sigma_y=1.0):
        self.C = C
        self.beta = beta
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.norm = 1.0 / (2 * math.pi * sigma_x * sigma_y)

    def __call__(self, delta_s, delta_t):
        """Evaluate kernel for many (x, t) - (x_j, t_j) pairs."""
        eps = 1e-6
        delta_t = delta_t.clamp(min=eps)

        dx = delta_s[..., 0]
        dy = delta_s[..., 1]

        space_exp = -0.5 * ((dx**2 / self.sigma_x**2) + (dy**2 / self.sigma_y**2)) / delta_t
        space_part = torch.exp(space_exp)
        time_part = torch.exp(-self.beta * delta_t)

        return self.C * self.norm * space_part * time_part / delta_t

class HawkesProcess(torch.nn.Module):
    def __init__(self, covariate_dim, kernel):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.zeros(covariate_dim))  # baseline weights
        self.kernel = kernel

    def baseline(self, z_xt):
        return torch.clamp(torch.matmul(z_xt, self.beta), min=1e-6)  # (B,)

    def log_intensity(self, x, t, past_x, past_t, covariates_xt):
        """
        x: (N, 2), t: (N,), covariates_xt: (N, D)
        past_x: (N, M, 2), past_t: (N, M)
        """
        mu = self.baseline(covariates_xt)  # (N,)
        x_diff = x[:, None, :] - past_x    # (N, M, 2)
        t_diff = t[:, None] - past_t       # (N, M)
        mask = (t_diff > 0).float()
        excitation = self.kernel(x_diff, t_diff) * mask  # (N, M)
        lambda_ = mu + excitation.sum(dim=1)             # (N,)
        return torch.log(lambda_ + 1e-6)                  # (N,)

    def integral_numerical(self, z_grid, x_grid, t_grid, event_x, event_t):
        """
        z_grid: (T, G, D), x_grid: (G, 2), t_grid: (T,)
        event_x: (N, 2), event_t: (N,)
        """
        T, G, D = z_grid.shape
        z_flat = z_grid.view(-1, D)         # (T*G, D)
        x_flat = x_grid.repeat(T, 1)        # (T*G, 2)
        t_flat = t_grid[:, None].repeat(1, G).reshape(-1)  # (T*G,)
        baseline_vals = self.baseline(z_flat)

        # Compute excitation sum over all events for each (x, t)
        delta_x = x_flat[None, :, :] - event_x[:, None, :]   # (N, T*G, 2)
        delta_t = t_flat[None, :] - event_t[:, None]         # (N, T*G)
        mask = (delta_t > 0).float()
        nu_vals = self.kernel(delta_x, delta_t) * mask       # (N, T*G)
        excitation_vals = nu_vals.sum(dim=0)                 # (T*G,)

        intensity_vals = baseline_vals + excitation_vals     # (T*G,)
        dxdy = 1.0 / G
        dt = (t_grid[1] - t_grid[0])
        return (intensity_vals.sum() * dxdy * dt)

def run_example():
    # Grid: 5x5 space, 10 time steps
    G = 25
    T = 10
    D = 2  # 2 covariates
    x_grid = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, 5), torch.linspace(0, 1, 5), indexing='ij'
    ), dim=-1).reshape(-1, 2)  # (G, 2)
    t_grid = torch.linspace(0, 1, T)  # (T,)

    z_grid = torch.randn(T, G, D) * 0.1 + 0.5  # small variation

    # Synthetic events: 10
    N = 10
    event_x = torch.rand(N, 2)
    event_t = torch.rand(N)
    event_t = event_t.sort()[0]
    z_events = torch.rand(N, D) * 0.1 + 0.5

    # Kernel and model
    kernel = StdDiffusionKernel(C=0.5, beta=2.0, sigma_x=0.1, sigma_y=0.1)
    model = HawkesProcess(D, kernel)

    # Forward pass
    past_x = event_x.unsqueeze(0).repeat(N, 1, 1)
    past_t = event_t.unsqueeze(0).repeat(N, 1)
    log_intensities = model.log_intensity(event_x, event_t, past_x, past_t, z_events)
    logsum = log_intensities.sum()

    integral = model.integral_numerical(z_grid, x_grid, t_grid, event_x, event_t)

    log_likelihood = logsum - integral
    print("Log-likelihood:", log_likelihood.item())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(100):
        optimizer.zero_grad()
        past_x = event_x.unsqueeze(0).repeat(N, 1, 1)
        past_t = event_t.unsqueeze(0).repeat(N, 1)
        log_intensities = model.log_intensity(event_x, event_t, past_x, past_t, z_events)
        logsum = log_intensities.sum()
        ll = logsum - model.integral_numerical(...)
        loss = -ll
        loss.backward()
        optimizer.step()
        if _ % 10 == 0:
            print("Log-likelihood:", f'{_}: {ll.item()}')
    

run_example()

