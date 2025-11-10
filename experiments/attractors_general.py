import argparse
import torch
import random
import numpy as np
from tqdm import tqdm
import os
from experiments.attractors_single import pca
from acds.archetypes.ron import RandomizedOscillatorsNetwork
from experiments.archetipes_network import ArchetipesNetwork
from experiments.connection_matrices import cycle_matrix
from collections import defaultdict


def plot_combined_pca(pca_results, out_dir, labels=None):
    """
    Plot multiple PCA results in a single scatter plot with different markers.

    Args:
        pca_results: list of np.ndarray, each of shape (N, 2) or (N, 3)
        out_path: path to save the figure (e.g., 'combined_pca.png')
        labels: optional list of labels for the legend
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    if not pca_results:
        raise ValueError("pca_results list is empty")

    dim = pca_results[0].shape[1]
    assert dim in (2, 3), "PCA results must be 2D or 3D"

    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>']
    colors = [plt.get_cmap('tab10')(i) for i in range(10)]

    fig = plt.figure(figsize=(8, 6))
    if dim == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    for i, data in enumerate(pca_results):
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        label = labels[i] if labels and i < len(labels) else f"Set {i+1}"
        if dim == 3:
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker=marker, color=color, label=label, alpha=0.7)
        else:
            ax.scatter(data[:, 0], data[:, 1], marker=marker, color=color, label=label, alpha=0.7)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if dim == 3:
        if hasattr(ax, "set_zlabel"):
            ax.set_zlabel("PC3")  # type: ignore
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pca_combined.png"))
    plt.close()


def main(args):
    # Prepare output directory
    out_dir = os.path.join(
        "results/results_collective",
        f"mod{args.n_modules}_rho_{args.rho}_nhid_{args.n_hid}_timesteps_{args.timesteps}_inpscaling_{args.inp_scaling}{args.suffix}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # Fix all random seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # define a list of models
    models = []
    for _ in range(args.n_modules):
        ron = RandomizedOscillatorsNetwork(
            n_inp=args.n_hid,
            n_hid=args.n_hid,
            dt=args.dt,
            rho=args.rho,
            gamma=args.gamma,
            epsilon=args.epsilon,
            device=args.device,
            input_scaling=args.inp_scaling
        )
        ron.bias = torch.nn.Parameter(torch.zeros(args.n_hid).to(args.device), requires_grad=False)
        models.append(ron)
    connection_matrix = cycle_matrix(args.n_modules) # make a param for connection type?
    network = ArchetipesNetwork(models, connection_matrix)
    x = torch.randn((args.timesteps, args.n_modules, args.n_hid)) # same input for each initialization
    initial_states = torch.rand(args.n_init_states, args.n_modules, 2, args.n_hid) * 2 - 1
    print(x.shape)

    def retrieve_states(x, initial_states):
        all_states, input_signals = network.forward(x, initial_states)
        all_states = torch.stack(all_states).permute(1, 0, 2, 3) # (n_modules, seq_len, 2, hdim)
        all_states = all_states[:, :, 0] # take just the first (hy)
        return all_states, input_signals
    
    all_states, input_signals = torch.vmap(retrieve_states, in_dims=(None, 0)
                                           )(x, initial_states)
    
    input_signals = torch.stack(input_signals).permute(2, 1, 0, 3).detach().numpy()
    input_signals_dict = {i: tensor for i, tensor in enumerate(input_signals)} # reshape and put input_signals in a dictionary indexed by modules
    torch.save(input_signals_dict, os.path.join(out_dir, f"input_signals.pt"))
    all_states =  all_states.permute(1, 0, 2, 3).detach().numpy()
    for i in range(args.n_modules):
        np.save(os.path.join(out_dir, f"all_states{i}.npy"), all_states[i])

    pca_results = []
    for i in range(args.n_modules):
        pca_result = pca(np.concatenate(all_states[i], axis=0), args.pca_dim, out_dir, suffix_file=f"_{i}")
        pca_results.append(pca_result)
    plot_combined_pca(pca_results, out_dir, labels=[f"{i}" for i in range(args.n_modules)])

    for i, ron in enumerate(models):
        print(i)
        np.savetxt(os.path.join(out_dir, f"W_{i}.csv"), ron.h2h.detach().cpu().numpy(), delimiter=',', fmt="%.6f")
        np.savetxt(os.path.join(out_dir, f"V_{i}.csv"), ron.x2h.detach().cpu().numpy(), delimiter=',', fmt="%.6f")
        np.savetxt(os.path.join(out_dir, f"b_{i}.csv"), ron.bias.detach().cpu().numpy(), delimiter=',', fmt="%.6f")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run attractors general experiment.")
    
    parser.add_argument('--n_modules', type=int, required=True, help='Number of modules')
    parser.add_argument('--rho', type=float, required=True, help='Parameter rho')
    parser.add_argument('--n_hid', type=int, required=True, help='Number of hidden units')
    parser.add_argument('--timesteps', type=int, required=True, help='Number of timesteps')
    parser.add_argument('--inp_scaling', type=float, default=1.0, help='Input scaling factor')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dt', type=float, default=1.0, help='Time step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='Parameter gamma')
    parser.add_argument('--epsilon', type=float, default=0.01, help='Parameter epsilon')
    parser.add_argument('--device', type=str, default='cpu', help='Device (e.g., cpu, cuda)')
    parser.add_argument('--pca_dim', type=int, default=2, help='PCA dimensionality')
    parser.add_argument('--washout', type=int, default=0, help='Number of washout timesteps')
    parser.add_argument("--n_init_states", type=int, default=1000, help="Number of initial states to generate")

    args = parser.parse_args()
    main(args)