import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple

from tqdm import tqdm
from acds.archetypes import InterconnectionRON
from acds.networks import ArchetipesNetwork, random_matrix, full_matrix, cycle_matrix, deep_reservoir, star_matrix, local_connections
from att_dim_experiments.utils import save_results, load_results
from einops import rearrange
import pickle
from pathlib import Path
from itertools import product
import traceback

def get_trajectory(network, x=None, seq_len=None, in_dim: Optional[int] = None, **network_kwargs):
    """Generate trajectory from network with given hyperparameters."""
    # validate inputs
    try:
        n_hid = network_kwargs["n_hid"]
        dt = network_kwargs["dt"]
        gamma = network_kwargs["gamma"]
        epsilon = network_kwargs["epsilon"]
        diffusive_gamma = network_kwargs["diffusive_gamma"]
        rho = network_kwargs["rho"]
        input_scaling = network_kwargs["input_scaling"]
        rho_m = network_kwargs.get("rho_m", 1.0)
    except Exception as e:
        raise e

    # create the input sequence if needed
    if x is None:
        assert seq_len is not None and in_dim is not None, \
            "Either provide an input sequence x of shape (seq_len, in_dim), or values for seq_len and in_dim"
        n_inp = in_dim
        x = torch.empty((seq_len, in_dim)).normal_()  # white noise
    else:
        n_inp = x.shape[1]

    # initialization
    n_modules = network_kwargs.get("n_modules", 1)

    initial_states = torch.empty(n_modules, 2, n_hid).normal_()

    with torch.no_grad():
        states, outs = network.forward(x, initial_states)

    return torch.stack([s[:, 0] for s in states]), torch.stack([s[:, 1] for s in states])

def get_model(network_kwargs, n_inp, rho_m):
    modules = []
    for _ in range(network_kwargs['n_modules']):
        modules.append(InterconnectionRON(
            n_inp=n_inp,
            n_hid=network_kwargs.get('n_hid'),
            dt=network_kwargs.get('dt'),
            gamma=network_kwargs.get('gamma'),
            epsilon=network_kwargs.get('epsilon'),
            diffusive_gamma=network_kwargs.get('diffusive_gamma'),
            rho=network_kwargs.get('rho'),
            input_scaling=network_kwargs.get('input_scaling'),
        ))

    cm_dict = {"random": random_matrix, 
               "full": full_matrix, 
               "cycle": cycle_matrix,
               "deep": deep_reservoir,
                "line": deep_reservoir,
                "star": star_matrix,
                "local": local_connections,
                "bidirectional": local_connections 
                }
    cm_type = network_kwargs.get("connection_matrix", "cycle")
    connection_matrix = cm_dict[cm_type]
    # input connection mask
    input_mask = torch.ones((network_kwargs['n_modules'],))
    if cm_type == "bidirectional" or cm_type == "deep":
        input_mask[1:] = 0 # only first module gets input

    if network_kwargs.get("connection_matrix", "cycle") == "random":
        p = network_kwargs.get("p", 0.2)
        connection_matrix = lambda n: random_matrix(n, p)
    network = ArchetipesNetwork(modules, connection_matrix(network_kwargs['n_modules']), rho_m = rho_m)
    return network


def collect_trajectories_grid(
    seq_len: int,
    in_dim: int,
    n_modules_list: List[int],
    n_hid_list: List[int],
    rho_list: List[float],
    input_scaling_list: List[float],
    mixing_scaling_list: List[float],
    dt_list: List[float],
    gamma_list: List[Tuple[float, float]],
    epsilon_list: List[Tuple[float, float]],
    base_network_kwargs: Dict,
    n_seeds: int = 1,
    use_same_input: bool = False,
) -> Dict:
    """
    Collect trajectories for a grid of hyperparameters.
    
    Parameters
    ----------
    seq_len : int
        Length of input sequence
    in_dim : int
        Input dimensionality
    n_modules_list : List[int]
        List of n_modules values to try
    n_hid_list : List[int]
        List of hidden dimensions to try
    rho_list : List[float]
        List of rho values to try
    input_scaling_list : List[float]
        List of input scaling values to try
    base_network_kwargs : Dict
        Base network kwargs (dt, gamma, epsilon, diffusive_gamma, connection_matrix)
    n_seeds : int
        Number of random seeds per configuration
    use_same_input : bool
        If True, use the same input sequence for all configurations
        
    Returns
    -------
    results : Dict
        Dictionary containing trajectories and metadata
    """
    results = {
        'trajectories': [],
        'hyperparameters': [],
        'metadata': {
            'seq_len': seq_len,
            'in_dim': in_dim,
            'base_network_kwargs': base_network_kwargs,
            'n_seeds': n_seeds
        }
    }
    
    # Generate input sequence once if needed
    shared_input = None

    if use_same_input:
        if args.input_type == 'white_noise':
            shared_input = torch.empty((seq_len, in_dim)).normal_()
        elif args.input_type == 'sine_wave':
            t = torch.linspace(0, seq_len * base_network_kwargs.get('dt', 1.0), seq_len)
            frequency = 0.1  # example frequency
            shared_input = torch.sin(2 * np.pi * frequency * t).unsqueeze(1).repeat(1, in_dim)
        else: # constant input
            try: 
                breakpoint()
                value = float(args.input_type)
                shared_input = torch.full((seq_len, in_dim), value)
            except:
                raise ValueError("Invalid input_type specified.")
    
    # Grid search over hyperparameters
    total_configs = (len(n_modules_list) * 
                    len(n_hid_list) * 
                    len(rho_list) * 
                    len(input_scaling_list) * 
                    len(mixing_scaling_list) *
                    len(dt_list) *
                    len(gamma_list) *
                    len(epsilon_list)
        )
    config_idx = 0
    hyperparameter_product = product(
        n_modules_list,
        n_hid_list,
        rho_list,
        input_scaling_list,
        mixing_scaling_list,
        dt_list,
        gamma_list,
        epsilon_list
    )
    for (n_modules, 
        n_hid, 
        rho, 
        input_scaling, 
        mixing_scaling,
        dt,
        gamma,
        epsilon
    ) in hyperparameter_product:
        # Prepare network kwargs
        network_kwargs = {
            **base_network_kwargs,
            'n_modules': n_modules,
            'n_hid': n_hid,
            'rho': rho,
            'rho_m': mixing_scaling,
            'input_scaling': input_scaling,
            'dt': dt,
            'gamma': gamma,
            'epsilon': epsilon
        }
        network = get_model(
            network_kwargs,
            n_inp=in_dim,
            rho_m = mixing_scaling
        )
        print(f"Collecting trajectories for config {config_idx + 1}/{total_configs}: "
              f"n_modules={n_modules}, n_hid={n_hid}, rho={rho}, "
                f"input_scaling={input_scaling}, mixing_scaling={mixing_scaling}")
        print(f"Trying {n_seeds} seeds...")
        config_idx += 1
        for seed in tqdm(range(n_seeds)):
            # Set seed for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)
            # Generate trajectory
            traj_h, traj_c = get_trajectory(
                network=network,
                x=shared_input,
                seq_len=seq_len if shared_input is None else None,
                in_dim=in_dim if shared_input is None else None,
                **network_kwargs
            )
                
            # Store results
            results['trajectories'].append({
                'h': traj_h.numpy(),
                'c': traj_c.numpy()
            })
            results['hyperparameters'].append({
                'n_modules': n_modules,
                'n_hid': n_hid,
                'rho': rho,
                'input_scaling': input_scaling,
                'mixing_scaling': mixing_scaling,
                'seed': seed,
                'dt': dt,
                'gamma': gamma,
                'epsilon': epsilon
            })

    return results





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect trajectories for varying hyperparameters")
    
    # Basic arguments
    parser.add_argument("--seq_len", type=int, default=5000, help="Sequence length")
    parser.add_argument("--in_dim", type=int, default=1, help="Input dimension")
    parser.add_argument("--n_seeds", type=int, default=3, help="Number of random seeds per config")
    parser.add_argument("--use_same_input", action="store_true", 
                        help="Use same input sequence for all configs")
    parser.add_argument("--input_type", type=str, default="white_noise",
                        help="Type of input sequence: 'white_noise', 'sine_wave', or a constant float value")
    parser.add_argument("--output_path", type=str, default="trajectories_collection.pkl",
                        help="Output file path")
    parser.add_argument("--only_load", action="store_true", 
                        help="Only load existing results without collecting new ones")
    # Hyperparameter ranges
    parser.add_argument("--n_modules_list", type=int, nargs="+", default=[128, 64, 32, 16, 8, 4, 2, 1],
                        help="List of n_modules values")
    parser.add_argument("--n_hid_list", type=int, nargs="+", default=[4, 32, 128],
                        help="List of hidden dimension values")
    parser.add_argument("--rho_list", type=float, nargs="+", default=[0.5, 0.9, 0.99, 1., 2.],
                        help="List of rho values")
    parser.add_argument("--input_scaling_list", type=float, nargs="+", default=[0.1, 1.0, 10.],
                        help="List of input scaling values")
    parser.add_argument("--mixing_scaling_list", type=float, nargs="+", default=[0.1, 0.5, 1.0, 2.0])
    # Fixed network parameters
    parser.add_argument("--dt_list", type=tuple, nargs="+", default=[1.])
    parser.add_argument("--epsilon_list", type=float, nargs="+", default=[1.])
    parser.add_argument("--gamma_list", type=float, nargs="+", default=[1.])
    group = parser.add_argument_group("fixed_network_args")
    group.add_argument("--diffusive_gamma", type=float, default=0.0)
    group.add_argument("--connection_matrix", type=str, default="cycle",
                       choices=["random", "full", "cycle", "deep", "line", "star", "local", "bidirectional"])
    group.add_argument("--p", type=float, default=0.2, help="Connection probability for random matrix")
    args = parser.parse_args()
    
    # Prepare base network kwargs (fixed parameters)
    base_network_kwargs = {
        'diffusive_gamma': args.diffusive_gamma,
        'connection_matrix': args.connection_matrix,
        'p': args.p
    }
    
    print("=" * 80)
    print("Collecting trajectories with hyperparameter grid search")
    print("=" * 80)
    print(f"Sequence length: {args.seq_len}")
    print(f"Input dimension: {args.in_dim}")
    print(f"Number of seeds per config: {args.n_seeds}")
    print(f"Use same input: {args.use_same_input}")
    print(f"\nHyperparameter ranges:")
    print(f"  n_modules: {args.n_modules_list}")
    print(f"  n_hid: {args.n_hid_list}")
    print(f"  rho: {args.rho_list}")
    print(f"  input_scaling: {args.input_scaling_list}")
    print(f"\nFixed parameters: {base_network_kwargs}")
    print("=" * 80)
    if not args.only_load:
    
        # Collect trajectories
        results = collect_trajectories_grid(
            seq_len=args.seq_len,
            in_dim=args.in_dim,
            n_modules_list=args.n_modules_list,
            n_hid_list=args.n_hid_list,
            rho_list=args.rho_list,
            input_scaling_list=args.input_scaling_list,
            base_network_kwargs=base_network_kwargs,
            mixing_scaling_list=args.mixing_scaling_list,
            dt_list=args.dt_list,
            gamma_list=args.gamma_list,
            epsilon_list=args.epsilon_list,
            n_seeds=args.n_seeds,
            use_same_input=args.use_same_input
        )

        # Save results
        save_results(results, args.output_path)
        
        print(f"\nCollection complete!")
        print(f"Total configurations: {len(results['trajectories'])}")
    
    if args.only_load:
        print(f"\nLoading existing results from {args.output_path}")
        results = load_results(args.output_path)
        print(f"Loaded {len(results['trajectories'])} configurations")
        print(f"Example hyperparameters: {results['hyperparameters'][0]}")
        print(f"Example trajectory shape: {results['trajectories'][0]['h'].shape}")