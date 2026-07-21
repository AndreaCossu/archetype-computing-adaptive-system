# Code adapted from https://github.com/MRamirez25/unicycle-network by Mariano Ramirez

import argparse
import math
import os

import numpy as np
import torch
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from acds.archetypes import UnicycleReservoir
from acds.benchmarks import get_mnist_data


EXPERIMENT_CONFIG = {
    "n_units": 20,
    "aligned_orientations": False,
    "ang_input": True,
    "ang_connections": True,
    "lin_stiff_min": 0.1,
    "lin_stiff_max": 0.5,
    "ang_stiff_min": 0.1,
    "ang_stiff_max": 0.3,
    "lin_damping_min": 0.1,
    "lin_damping_max": 0.2,
    "ang_damping_min": 0.1,
    "ang_damping_max": 0.2,
    "dt": 0.001,
    "inp_bias": 0.0,
    "lin_input_non_zero": 20,
    "lin_input_min": -1.0,
    "lin_input_max": 1.0,
    "ang_input_non_zero": 20,
    "ang_input_min": -1.0,
    "ang_input_max": 1.0,
    "n_connections_fraction": 0.5,
    "anchor_con_fraction": 0.3,
    "n_connections_ang_fraction": 0.5,
    "anchor_con_fraction_ang": 0.3,
    "washup_steps": 1000,
    "n_past_steps_readout": 0,
    "eq_dist_min": 0.5,
    "eq_dist_max": 1.0,
    "eq_dist_min_ang": -math.pi,
    "eq_dist_max_ang": math.pi,
}


def make_input_map(n_units, num_non_zero, min_value, max_value, device):
    """Create a sparse random input map for a unicycle reservoir.

    :param n_units: Number of reservoir units.
    :param num_non_zero: Number of non-zero entries to sample.
    :param min_value: Minimum sampled value.
    :param max_value: Maximum sampled value.
    :param device: Torch device for the returned tensor.
    :return: Tensor with shape ``(1, n_units)``.
    """
    input_map = torch.zeros(1, n_units, device=device)
    num_non_zero = min(max(num_non_zero, 0), n_units)
    if num_non_zero == 0:
        return input_map

    non_zero_indices = torch.randperm(n_units, device=device)[:num_non_zero]
    non_zero_values = torch.rand(num_non_zero, device=device) * (max_value - min_value) + min_value
    input_map[0, non_zero_indices] = non_zero_values
    return input_map


def move_static_tensors(model, device):
    """Move static reservoir tensors to the selected device.

    :param model: Unicycle reservoir instance.
    :param device: Target Torch device.
    """
    model.lin_input_map = model.lin_input_map.to(device)
    model.ang_input_map = model.ang_input_map.to(device)
    model.unicycle_network.lin_damping = model.unicycle_network.lin_damping.to(device)
    model.unicycle_network.ang_damping = model.unicycle_network.ang_damping.to(device)
    model.unicycle_network.mass_vector = model.unicycle_network.mass_vector.to(device)
    model.unicycle_network.j_vector = model.unicycle_network.j_vector.to(device)


def make_initial_state(batch_size, n_units, aligned_orientations, device):
    """Sample an initial unicycle reservoir state.

    :param batch_size: Number of batch states to create.
    :param n_units: Number of reservoir units.
    :param aligned_orientations: If ``True``, all units share one angle.
    :param device: Torch device for returned tensors.
    :return: Tuple ``(x, z, theta, s, omega)``.
    """
    x = torch.rand(batch_size, n_units, device=device)
    z = torch.rand(batch_size, n_units, device=device)
    theta = torch.rand(batch_size, n_units, device=device) * (4 * math.pi) - (2 * math.pi)
    if aligned_orientations:
        theta[:] = torch.rand(1, device=device) * (4 * math.pi) - (2 * math.pi)
    s = torch.zeros(batch_size, n_units, device=device)
    omega = torch.zeros(batch_size, n_units, device=device)
    s[:, 0] = 0
    return x, z, theta, s, omega


def set_model_initial_state(model, batch_size, initial_state):
    """Copy a template initial state into a model for a batch.

    :param model: Unicycle reservoir instance.
    :param batch_size: Batch size to expand the state to.
    :param initial_state: Tuple returned by :func:`make_initial_state` or
        :func:`run_washup`.
    """
    model.x_init = initial_state[0].expand(batch_size, -1).clone()
    model.z_init = initial_state[1].expand(batch_size, -1).clone()
    model.theta_init = initial_state[2].expand(batch_size, -1).clone()
    model.s_init = initial_state[3].expand(batch_size, -1).clone()
    model.omega_init = initial_state[4].expand(batch_size, -1).clone()


@torch.no_grad()
def run_washup(model, initial_state, washup_steps, device):
    """Run zero-input washup dynamics and return the settled state.

    :param model: Unicycle reservoir instance.
    :param initial_state: Initial ``(x, z, theta, s, omega)`` tuple.
    :param washup_steps: Number of zero-input simulation steps.
    :param device: Torch device used for generated inputs.
    :return: Detached settled state tuple.
    """
    x, z, theta, s, omega = initial_state
    u_lin = torch.zeros((1, washup_steps, 1), device=device)
    u_ang = torch.zeros_like(u_lin, device=device)

    for t in range(washup_steps):
        linear_input = u_lin[:, t] @ model.lin_input_map
        angular_input = u_ang[:, t] @ model.ang_input_map
        x, z, theta, s, omega = model.unicycle_network(
            linear_input, angular_input, x, z, theta, s, omega
        )

    return tuple(state.detach() for state in (x, z, theta, s, omega))


@torch.no_grad()
def collect_activations(data_loader, model, initial_state, device, desc):
    """Collect reservoir activations and labels from a data loader.

    :param data_loader: Loader yielding image batches and labels.
    :param model: Unicycle reservoir used as a feature extractor.
    :param initial_state: Template reservoir state for each batch.
    :param device: Torch device for inference.
    :param desc: Progress-bar description and error context.
    :return: Tuple ``(activations, labels)`` as NumPy arrays.
    :raises RuntimeError: If reservoir states contain NaNs.
    """
    activations, ys = [], []
    for images, labels in tqdm(data_loader, desc=desc):
        batch_size = images.shape[0]
        images = images.reshape(batch_size, 1, 784).permute(0, 2, 1).to(device)
        set_model_initial_state(model, batch_size, initial_state)
        _, _, mid_states = model(images, images)

        if torch.isnan(mid_states).any():
            raise RuntimeError(f"NaN detected in reservoir states during {desc}.")

        activations.append(mid_states.cpu())
        ys.append(labels.cpu())

    return torch.cat(activations, dim=0).numpy(), torch.cat(ys, dim=0).numpy()


def score_esn(data_loader, model, classifier, scaler, initial_state, device, desc):
    """Score a readout on scaled reservoir activations.

    :param data_loader: Loader to evaluate.
    :param model: Reservoir feature extractor.
    :param classifier: Fitted classifier exposing ``score``.
    :param scaler: Fitted scaler exposing ``transform``.
    :param initial_state: Template reservoir state.
    :param device: Torch device for inference.
    :param desc: Progress-bar description.
    :return: Classifier score.
    """
    activations, ys = collect_activations(data_loader, model, initial_state, device, desc)
    activations = scaler.transform(activations)
    return classifier.score(activations, ys)


def run_experiment(config, dataroot, batch_size, device):
    """Run the fixed MNIST unicycle-reservoir experiment.

    :param config: Experiment configuration dictionary.
    :param dataroot: Root directory for the MNIST dataset.
    :param batch_size: Batch size for train/validation/test loaders.
    :param device: Torch device for model execution.
    :return: Dictionary with ``train``, ``validation``, and ``test`` scores.
    :raises RuntimeError: If collected activations contain NaNs.
    """
    n_units = config["n_units"]
    lin_input_map = make_input_map(
        n_units,
        config["lin_input_non_zero"],
        config["lin_input_min"],
        config["lin_input_max"],
        device,
    )
    if config["ang_input"]:
        ang_input_map = make_input_map(
            n_units,
            config["ang_input_non_zero"],
            config["ang_input_min"],
            config["ang_input_max"],
            device,
        )
    else:
        ang_input_map = torch.zeros(1, n_units, device=device)

    n_connections = int(n_units * config["n_connections_fraction"])
    n_connections_anchor = int(n_units * config["anchor_con_fraction"])
    if config["ang_connections"]:
        n_connections_ang = int(n_units * config["n_connections_ang_fraction"])
        n_connections_anchor_ang = int(n_units * config["anchor_con_fraction_ang"])
    else:
        n_connections_ang = 0
        n_connections_anchor_ang = 0

    model = UnicycleReservoir(
        n_inp=1,
        n_units=n_units,
        dt=config["dt"],
        n_out=10,
        lin_stiff_min=config["lin_stiff_min"],
        lin_stiff_max=config["lin_stiff_max"],
        ang_stiff_min=config["ang_stiff_min"],
        ang_stiff_max=config["ang_stiff_max"],
        lin_damping_min=config["lin_damping_min"],
        lin_damping_max=config["lin_damping_max"],
        ang_damping_min=config["ang_damping_min"],
        ang_damping_max=config["ang_damping_max"],
        eq_dist_min=config["eq_dist_min"],
        eq_dist_max=config["eq_dist_max"],
        eq_dist_min_ang=config["eq_dist_min_ang"],
        eq_dist_max_ang=config["eq_dist_max_ang"],
        n_connections=n_connections,
        inp_bias=config["inp_bias"],
        lin_input_map=lin_input_map,
        n_connections_anchor=n_connections_anchor,
        ang_input_map=ang_input_map,
        n_connections_ang=n_connections_ang,
        n_connections_anchor_ang=n_connections_anchor_ang,
        n_past_steps_readout=config["n_past_steps_readout"],
    ).to(device)
    move_static_tensors(model, device)

    initial_state = make_initial_state(
        1, n_units, config["aligned_orientations"], device
    )
    initial_state = run_washup(
        model, initial_state, config["washup_steps"], device
    )
    print(
        "Initial state NaNs - "
        f"x: {torch.isnan(initial_state[0]).any()}, "
        f"z: {torch.isnan(initial_state[1]).any()}, "
        f"theta: {torch.isnan(initial_state[2]).any()}, "
        f"s: {torch.isnan(initial_state[3]).any()}, "
        f"omega: {torch.isnan(initial_state[4]).any()}"
    )

    train_loader, valid_loader, test_loader = get_mnist_data(
        dataroot, batch_size, batch_size
    )

    activations, ys = collect_activations(
        train_loader, model, initial_state, device, "train"
    )
    if np.isnan(activations).any():
        raise RuntimeError("NaN detected in training activations.")

    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    classifier = LogisticRegression(max_iter=1000).fit(activations, ys)

    return {
        "train": classifier.score(activations, ys),
        "validation": score_esn(
            valid_loader, model, classifier, scaler, initial_state, device, "validation"
        ),
        "test": score_esn(
            test_loader, model, classifier, scaler, initial_state, device, "test"
        ),
    }


def parse_args():
    """Parse command-line arguments for the fixed MNIST experiment.

    :return: Parsed ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(description="Run one fixed MNIST reservoir experiment.")
    parser.add_argument("--dataroot", type=str, default=os.getcwd())
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Using device: {device}")
    print(f"Running one fixed MNIST experiment with n_units={EXPERIMENT_CONFIG['n_units']}")

    scores = run_experiment(EXPERIMENT_CONFIG, args.dataroot, args.batch_size, device)
    print(
        "Scores - "
        f"train: {scores['train']:.4f}, "
        f"validation: {scores['validation']:.4f}, "
        f"test: {scores['test']:.4f}"
    )
