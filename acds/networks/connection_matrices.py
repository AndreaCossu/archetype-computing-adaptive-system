import torch
import networkx as nx
import matplotlib.pyplot as plt

def cycle_matrix(n):
    if n == 1:
        return torch.zeros((1, 1))
    m = torch.zeros((n, n))
    m[torch.arange(1, n, dtype=torch.int), torch.arange(0, n-1, dtype=torch.int)] = 1 
    m[0, n-1] = 1
    return m

def full_matrix(n):
    if n == 1:
        return torch.zeros((1, 1))
    m = torch.ones((n, n))
    m[torch.arange(n, dtype=torch.int), torch.arange(n, dtype=torch.int)] = 0
    return m


def random_matrix(n, p=0.5, seed=None):
    if n == 1:
        return torch.zeros((1, 1))
    if seed is not None:
        torch.manual_seed(seed)
    probs = torch.empty((n, n)).uniform_(0, 1)
    m = torch.bernoulli(probs, p=p)
    m[torch.arange(n, dtype=torch.int), torch.arange(n, dtype=torch.int)] = 0
    return m


def star_matrix(n):
    if n == 1:
        return torch.zeros((1, 1))

    m = torch.zeros((n, n))
    m[1, :] = 1
    m[:, 1] = 1
    return m


def deep_reservoir(n):
    return torch.diag(torch.ones(n-1), -1)


def local_connections(n):
    m = torch.diag(torch.ones(n-1), -1)
    # fill the upper diagonal
    m[torch.arange(0, n-1, dtype=torch.int), torch.arange(1, n, dtype=torch.int)] = 1
    return m

def grid_local_connections(n_rows, n_cols, torus=False, diag=False):
    """Grid local connections. Neurons are connected to their neighbors in a 2D grid. Optionally, wrap around edges to form a torus and include diagonal connections.

    Args:
        n_rows (int): number of rows in the grid.
        n_cols (int): number of columns in the grid.
        torus (bool, optional): Whether to wrap around edges to form a torus. Defaults to False.
        diag (bool, optional): Whether to include diagonal connections. Defaults to False.
    """

    m = torch.zeros((n_rows * n_cols, n_rows * n_cols))

    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c

            # Connect to the right neighbor
            if c < n_cols - 1:
                m[idx, idx + 1] = 1
            elif torus:
                m[idx, r * n_cols] = 1  # wrap around

            # Connect to the left neighbor
            if c > 0:
                m[idx, idx - 1] = 1
            elif torus:
                m[idx, r * n_cols + (n_cols - 1)] = 1  # wrap around

            # Connect to the bottom neighbor
            if r < n_rows - 1:
                m[idx, idx + n_cols] = 1
            elif torus:
                m[idx, c] = 1  # wrap around

            # Connect to the top neighbor
            if r > 0:
                m[idx, idx - n_cols] = 1
            elif torus:
                m[idx, (n_rows - 1) * n_cols + c] = 1  # wrap around

            if diag:
                # Diagonal connections
                if r < n_rows - 1 and c < n_cols - 1:
                    m[idx, idx + n_cols + 1] = 1  # bottom-right
                if r < n_rows - 1 and c > 0:
                    m[idx, idx + n_cols - 1] = 1  # bottom-left
                if r > 0 and c < n_cols - 1:
                    m[idx, idx - n_cols + 1] = 1  # top-right
                if r > 0 and c > 0:
                    m[idx, idx - n_cols - 1] = 1  # top-left
    return m


def visualize_matrix_as_graph(matrix, layout='spring', save_path=None):
    plt.figure(figsize=(8, 8))
    G = nx.from_numpy_array(matrix.numpy())
    if layout == 'spring':
        pos = nx.spring_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'grid':
        pos = {i: (i % 5, i // 5) for i in range(matrix.shape[0])}
    else:
        raise ValueError("Unsupported layout type. Choose from 'spring', 'circular', 'kamada_kawai'.")
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()