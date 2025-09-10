import numpy as np
import torch
import torch.nn
import matplotlib.pyplot as plt
import imageio
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogFormatterMathtext
from matplotlib import colors


#VISUALIZATION:


def log_norm_from_pos(arr, p_lo=5, p_hi=99.5, floor=1e-8):
    """
    Build a LogNorm from positive entries of arr.
    Uses percentiles to avoid outliers. Falls back to linear if no positives.

    Makes visualization of kernel intensity visible.
    """
    pos = arr[arr > 0]
    if pos.size == 0:
        #fall back: linear 0..1 to avoid crashes
        return None, Normalize(vmin=0.0, vmax=1.0, clip=True)
    vmin = max(np.percentile(pos, p_lo), floor)
    vmax = max(np.percentile(pos, p_hi), vmin * 1.001)
    return LogNorm(vmin=vmin, vmax=vmax, clip=True), None

def animate_intensity(lam, cbar_label, events, cell_coords, output_path, scale = 'lin', cmap = "OrRd", fps=5,
                      decay=0.8, figsize=(6,6), separate_base_ker=False, device="cpu"):
    """
    lam: float torch.tensor object of size (T,C) for the year containing the intensity in each grid cell at each time step.
        althought this is called lam, it is actually a general method which can take any part of the intensity,
        e.g. just the baseline or excitation part of a hawkes model. 
    cbar_label: str label for colorbar of plot. describes what kind of intensity is shown, e.g. lambda, baseline, excitation, inhibitory
    events: int torch.Tensor, shape (N_y, 2); time step and grid cell ids of all events for the year
    cell_coords: np.ndarray shape (C,2) of (x,y) centroids for each cell
    output_path: path to save gif to
    scale: scale of colorbar. defaults to linear 'lin'. one of 'lin', 'log'.
    cmap: a cmap for the colorbar.
    fps: fps of the gif
    decay: float in (0,1), controls per-frame decay of event dots
    figsize: size of the figure
    separate_base_ker: whether to make separate animations for the base intensity and the excitation kernel intensity
    """
    assert (scale == "lin") | (scale == "log"), "scale must be 'lin' or 'log'"
    dev = torch.device(device)
    if lam.ndim == 3 and lam.shape[0] == 1:
        lam_t = lam.squeeze(0)
    else: 
        lam_t = lam
    if lam_t.ndim != 2:
        raise ValueError(f"Expected intensity shape (T,C), got {lam_t.shape}")
    T, C = lam.shape

    #build an event mask the same shape (T, C)
    event_mask = torch.zeros((T, C), dtype=torch.bool, device=dev)
    if not (torch.is_tensor(events) and events.dtype == torch.long):
        events = torch.as_tensor(events, dtype=torch.long, device = dev)
    else:
        events = events.to(dev)
    if events.ndim == 3 and events.shape[0] == 1:
        events = events.squeeze(0)
    event_mask[events[:,0], events[:,1]] = True

    ev = event_mask.to(dev).float() #need float to do computations for viz
    if ev.ndim == 3:
        ev = ev.squeeze(0)

    lam_arr = lam_t.cpu().numpy()
    xs, ys = cell_coords[:,0], cell_coords[:,1]

    #setup figure
    event_disp = np.zeros(C, dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    if scale == 'log':
        norm_base_log, norm_base_lin = log_norm_from_pos(lam_arr)
        if norm_base_log is not None:
            sc_int = ax.scatter(xs, ys, c=lam_arr[0], norm=norm_base_log, s=20, cmap=cmap)
        else:
            sc_int = ax.scatter(xs, ys, c=lam_arr[0], s=20, cmap=cmap, norm=colors.Normalize(vmin=lam_arr.min(), vmax=lam_arr.max()))
    else:
        sc_int = ax.scatter(xs, ys, c=lam_arr[0], s=20, cmap=cmap, norm=colors.Normalize(vmin=lam_arr.min(), vmax=lam_arr.max()))

    if scale == 'lin':
        cbar = plt.colorbar(sc_int, ax=ax, label=cbar_label, format = ScalarFormatter(useMathText=True))
    else:
        cbar = plt.colorbar(sc_int, ax=ax, label=cbar_label, format = LogFormatterMathtext())
    sc_evt = ax.scatter(xs, ys, s=0, c="#78e8ff", alpha=0.0)
    ax.set_axis_off()
    #update function
    def update(t):
        nonlocal event_disp
        #update intensity colors
        arr = lam_arr[t] #1D of length C
        sc_int.set_array(arr)
        #update decaying event size + alpha
        event_disp = event_disp * decay + ev[t].cpu().numpy()
        event_disp = np.clip(event_disp, 0.0, 1.0)
        sc_evt.set_sizes(100 * event_disp)
        sc_evt.set_alpha(event_disp)
        ax.set_title(f"Time step {t}")
        return sc_int, sc_evt 
    #draw baseline
    frames = []
    for t in range(T):
        update(t) #redraw artists for frame t
        fig.canvas.draw() #render the canvas
        #grab the RGB buffer from the figure
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = buf.reshape(h, w, 3)
        frames.append(img)
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Saved GIF to {output_path}")
    plt.close('all')
