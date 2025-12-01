import argparse
import torch
from typing import Optional
from acds.archetypes import RandomizedOscillatorsNetwork
from acds.networks import ArchetipesNetwork, random_matrix, full_matrix, cycle_matrix
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from einops import rearrange


import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

def plot_multiple_fading_trajectories(
        trajectories,
        transient = 100,
        labels=None,
        alpha_start=0.1,
        alpha_end=1.0,
        line_alpha=0.5,
        line_width=1.5,
        ax=None):
    """
    Plot multiple trajectories with progressive transparency.

    Parameters
    ----------
    trajectories : list of (x, y)
        Each trajectory is a pair of equal-length arrays.
    transient : int
        Transient of the trajectories
    labels : list of str, optional
        Labels for each trajectory. If None, no labels are added.
    alpha_start, alpha_end : float
        Start and end alpha for fading scatter points.
    line_alpha : float
        Transparency of connecting line.
    line_width : float
        Width of connecting line.
    ax : matplotlib Axes, optional
        Axes to plot on. If None, a new one is created.

    Returns
    -------
    ax : matplotlib Axes
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Matplotlib default color cycle
    color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    # Auto-generate labels if user passes None
    if labels is None:
        labels = [None] * len(trajectories)

    for (x, y), label in zip(trajectories, labels):
        x = np.asarray(x[transient:])
        y = np.asarray(y[transient:])
        n = len(x)

        # Get next line color from Matplotlib automatic cycle
        base_color = next(color_cycle)

        # Fading alphas
        alphas = np.linspace(alpha_start, alpha_end, n)

        # Convert base_color (RGB) to per-point RGBA list
        rgb = mcolors.to_rgb(base_color)
        colors = [(rgb[0], rgb[1], rgb[2], a) for a in alphas]

        # Plot line with label
        ax.plot(x, y, color=base_color, alpha=line_alpha, linewidth=line_width, label=label)

        # Scatter with varying transparency
        ax.scatter(x, y, c=colors)

    if any(label is not None for label in labels):
        ax.legend()

    return ax


def get_trajectory(x=None, seq_len=None, in_dim:Optional[int]=None, **network_kwargs):
    # validate inputs
    try:
        n_hid = network_kwargs["n_hid"]
        dt = network_kwargs["dt"]
        gamma = network_kwargs["gamma"]
        epsilon = network_kwargs["epsilon"]
        diffusive_gamma = network_kwargs["diffusive_gamma"]
        rho = network_kwargs["rho"]
        input_scaling = network_kwargs["input_scaling"]
    except Exception as e:
        raise e
    
    # create the input sequence if needed
    if x is None:
        assert seq_len is not None and in_dim is not None, "Either provide an input sequence x of shape (seq_len, in_dim), or values for seq_len and in_dim "
        n_inp = in_dim
        x = torch.empty((seq_len, in_dim)).normal_() # white noise
    else:
        n_inp = x.shape[1]
    # initialization
    n_modules = network_kwargs.get("n_modules", 1)
    modules = []
    for _ in range(n_modules):
        modules.append(RandomizedOscillatorsNetwork(
            n_inp,
            n_hid,
            dt,
            gamma,
            epsilon,
            diffusive_gamma,
            rho,
            input_scaling
        ))
    cm_dict ={"random": random_matrix, "full": full_matrix, "cycle": cycle_matrix}
    connection_matrix = cm_dict[network_kwargs.get("connection_matrix", "cycle")]
    network = ArchetipesNetwork(modules, connection_matrix(n_modules))
    initial_states = torch.empty(n_modules, 2, n_hid).normal_()# should be maybe changed to be random ?
    with torch.no_grad():
        states, outs = network.forward(x, initial_states)
    return torch.stack([s[:,0] for s in states]), torch.stack([s[:, 1] for s in states])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None
    )
    parser.add_argument(
        "--in_dim",
        type=int,
        default=None
    )

    group = parser.add_argument_group("network_args")
    group.add_argument(
        "--n_modules",
        type=int,
        default=1,
    )
    group.add_argument(
        "--n_hid",
        type=int,
        default=2,
    )
    group.add_argument(
        "--dt",
        type=float,
        default=0.1,
    )
    group.add_argument(
        "--gamma",
        type=float,
        default=1.,
    )
    group.add_argument(
        "--epsilon",
        type=float,
        default=1.,
    )
    group.add_argument(
        "--diffusive_gamma",
        type=float,
        default=1.,
    )
    group.add_argument(
        "--rho",
        type=float,
        default=0.9,
    )
    group.add_argument(
        "--input_scaling",
        type=float,
        default=1.,
    )

    args = parser.parse_args()

    network_kwargs = {
        action.dest: getattr(args, action.dest)
        for action in group._group_actions
    }

    print(network_kwargs)
    traj = get_trajectory(
        seq_len=args.seq_len,
        in_dim = args.in_dim,
        **network_kwargs,
        x = torch.zeros((10000, 1))
    )
    print(traj[0].shape, traj[1].shape)
    
    h_trajectories = rearrange(traj[0], "seq nmod hdim -> nmod hdim seq")

    plot_multiple_fading_trajectories(h_trajectories, transient=1000)

plt.show()