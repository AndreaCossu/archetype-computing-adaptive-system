import torch


def cycle_matrix(n):
    m = torch.zeros((n, n))
    m[torch.arange(1, n, dtype=torch.int), torch.arange(0, n-1, dtype=torch.int)] = 1 
    m[0, n-1] = 1
    return m

def full_matrix(n):
    m = torch.ones((n, n))
    m[torch.arange(n, dtype=torch.int), torch.arange(n, dtype=torch.int)] = 0
    return m


def random_matrix(n, p=0.5, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    probs = torch.empty((n, n)).uniform_(0, 1)
    m = torch.bernoulli(probs, p=p)
    m[torch.arange(n, dtype=torch.int), torch.arange(n, dtype=torch.int)] = 0
    return m


def star_matrix(n):
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


if __name__ == '__main__':
    print(cycle_matrix(5), "\n")
    print(random_matrix(5), "\n")
    print(random_matrix(5, p=0.33), "\n")