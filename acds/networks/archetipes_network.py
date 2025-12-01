import torch
import sys
sys.path.append("..")
from typing import List, Sequence
from torch import nn
from acds.archetypes import RandomizedOscillatorsNetwork as RON

from einops import einsum, rearrange # I like using einops because it allows strings as axes names instead of just letters, and for other stuff
# from torch import einsum
class ArchetipesNetwork(nn.Module):
    def __init__(
        self,
        archetypes: Sequence[RON],
        connection_matrix: torch.Tensor,
    ):
        """A network of interconnected archetipes with any topology

        Args:
            archetipes_list (List[nn.Module]): A list of N archetipes
            connections (torch.Tensor): A NxN binary matrix specifying how the archetipes are connected
        """
        super().__init__()
        self.archetipes = archetypes
        self.n_hid = self.archetipes[0].n_hid
        self.n_inp = self.archetipes[0].n_inp
        self.n_modules = len(archetypes)
        self.connection_matrix = nn.Parameter(connection_matrix)
        self.wm = torch.nn.Linear(self.n_hid, self.n_inp, bias=False)

    def _step(self, x, prev_states, prev_outs):
        """Perform one step of forward pass

        Args:
            x (Tensor of shape (h_dim)): external input at time t
            prev_states(Tensor of shape (n_modules, n_states, h_dim)): state(s) for each archetipe at time t-1, which are also the outputs
            prev_outs (Tensor of shape (n_modules, h_dim)): output of the models in the previous timestep, (e.g. h for ESN or h_y for RON)
        """
        new_ins = einsum(self.connection_matrix, prev_outs, "exp_out exp_in, exp_in hdim -> exp_out hdim") # actually implementable simply with matmul idk if it's more efficient
        new_ins = self.wm(new_ins)
        new_ins = einsum(new_ins, x, "exp idim, idim -> exp idim") 
        # torch version of einsum
        # new_ins = einsum("oi,ih -> oh", self.connection_matrix, prev_outs)
        # new_ins = einsum("mi,mi->mi", new_ins, x)
        new_outs = []
        for model, x, states in zip(self.archetipes, new_ins, prev_states):
            x = rearrange(x, "in_dim -> 1 in_dim")
            states = model.cell(x, states[0], states[1])
            new_outs.append(torch.concat(states))

        return torch.stack(new_outs), new_ins
    
    def forward(self, x:torch.Tensor, initial_states, initial_outs=None):
        """Forward of the network

        Args:
            x (torch.Tensor): input sequence of shape (seq_len, in_dim)
            initial_states (torch.Tensor): Initial states of shape (n_modules, 2, h_dim), where 2 is the n. of states in a RON model
            initial_outs (torch.Tensor, optional): Initial outputs of the networks, of shape (n_modules, h_dim) If None, they are initialized to torch.zeros. Defaults to None.

        Returns:
            _type_: _description_
        """
        input_list = [x[0]]
        states = initial_states
        state_list = [states]
        if initial_outs is None:
            outs = torch.zeros((self.n_modules, self.n_hid))
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
    from acds.networks.connection_matrices import cycle_matrix
    archetypes = [ReservoirCell(HDIM, HDIM) for _ in range(N_MODULES)]
    connection_matrix = cycle_matrix(N_MODULES)
    net = ArchetipesNetwork(archetypes, connection_matrix)
    print(net.archetipes)
    states = torch.stack([torch.empty(HDIM).normal_() for _ in range(N_MODULES)])
    x = torch.empty(SEQ_LEN, HDIM).normal_(0, 0.01)
    print(len(net.forward(x, states)))

if __name__ == "__main__":
    main()