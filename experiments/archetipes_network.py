import torch
import sys
sys.path.append("..")
from typing import List, Sequence
from torch import nn
from acds import archetypes
from einops import einsum, rearrange# I like using einops because it allows strings as axes names instead of just letters, and for other stuff
# from torch import einsum
class ArchetipesNetwork(nn.Module):
    def __init__(
        self,
        archetypes: Sequence[nn.Module],
        connection_matrix: torch.Tensor
    ):
        """A network of interconnected archetipes with any topology

        Args:
            archetipes_list (List[nn.Module]): A list of N archetipes
            connections (torch.Tensor): A NxN binary matrix specifying how the archetipes are connected
        """
        super().__init__()
        self.archetipes = archetypes
        self.connection_matrix = nn.Parameter(connection_matrix)

    def _step(self, x, prev_states, prev_outs):
        """Perform one step of forward pass

        Args:
            x (Tensor of shape (h_dim)): external input at time t
            prev_states(Tensor of shape (n_modules, n_states, h_dim)): state(s) for each archetipe at time t-1, which are also the outputs
            prev_outs (Tensor of shape (n_modules, h_dim)): output of the models in the previous timestep, (e.g. h for ESN or h_y for RON)
        """
        new_ins = einsum(self.connection_matrix, prev_outs, "exp_out exp_in, exp_in hdim -> exp_out hdim") # actually implementable simply with matmul idk if it's more efficient
        new_ins = einsum(new_ins, x, "n_mod idim, n_mod idim -> n_mod idim")  # TODO: could pass the external input only to some modules -> n+1 x n connection matrix
        new_outs = []
        for model, x, states in zip(self.archetipes, new_ins, prev_states):
            x = rearrange(x, "in_dim -> 1 in_dim")
            #states = rearrange(states, "h_dim -> 1 h_dim")
            states:tuple = model.cell(x, states[0], states[1])
            new_outs.append(torch.concat(states))
        return torch.stack(new_outs), new_ins
    
    def forward(self, x:torch.Tensor, initial_states, initial_outs=None):
        input_list = [x[0]]
        states = initial_states
        state_list = [states]
        if initial_outs is None:
            outs = torch.zeros_like(x[0])
        for x_t in x:
            states, ins = self._step(x_t, states, outs)
            outs = states[:, 0] # we assume the first state is the "output" one
            state_list.append(states)
            input_list.append(ins)
        return state_list, input_list
    
    def __iter__(self):
        for model in self.archetipes:
            yield(model)



def main():
    N_MODULES = 10
    HDIM = 16
    SEQ_LEN = 100
    from acds.archetypes import ReservoirCell
    from experiments.connection_matrices import cycle_matrix
    archetypes = [ReservoirCell(HDIM, HDIM) for _ in range(N_MODULES)]
    connection_matrix = cycle_matrix(N_MODULES)
    net = ArchetipesNetwork(archetypes, connection_matrix)
    print(net.archetipes)
    states = torch.stack([torch.empty(HDIM).normal_() for _ in range(N_MODULES)])
    x = torch.empty(SEQ_LEN, HDIM).normal_(0, 0.01)
    print(len(net.forward(x, states)))

if __name__ == "__main__":
    main()