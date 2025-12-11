import torch
import sys

from tqdm import tqdm
sys.path.append("..")
from typing import List, Sequence
from torch import nn
from acds.archetypes import InterconnectionRON as RON

from einops import einsum, rearrange # I like using einops because it allows strings as axes names instead of just letters, and for other stuff
# from torch import einsum
class ArchetipesNetwork(nn.Module):
    def __init__(
        self,
        archetypes: Sequence[RON],
        connection_matrix: torch.Tensor,
        rho_m: float = 1.0,
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
        self.wm = torch.nn.Linear(self.n_hid, self.n_hid, bias=False)
        spec_rad = torch.linalg.eigvals(self.wm.weight).abs().max()
        self.wm.weight.data *= rho_m / spec_rad # rescale to have spectral radius rho_m


    def _step(self, x, prev_states, prev_outs):
        """Perform one step of forward pass

        Args:
            x (Tensor of shape (h_dim)): external input at time t
            prev_states(Tensor of shape (n_modules, n_states, h_dim)): state(s) for each archetipe at time t-1, which are also the outputs
            prev_outs (Tensor of shape (n_modules, h_dim)): output of the models in the previous timestep, (e.g. h for ESN or h_y for RON)
        """
        prev_outs_transformed = self.wm(prev_outs) # transform the outputs before feeding them back
        feedback = self.connection_matrix @ prev_outs_transformed 
        #feedback = einsum(self.connection_matrix, prev_outs_transformed, "exp_out exp_in, exp_in hdim -> exp_out hdim") # actually implementable simply with matmul idk if it's more efficient
        # torch version of einsum
        # new_ins = einsum("oi,ih -> oh", self.connection_matrix, prev_outs)
        # new_ins = einsum("mi,mi->mi", new_ins, x)
        new_outs = []
        #x = rearrange(x, "in_dim -> 1 in_dim")
        x = x.unsqueeze(0)  # shape (1, in_dim)
        for model, fb, states in zip(self.archetipes, feedback, prev_states):
            states = model.cell(x, states[0], states[1], fb)
            new_outs.append(torch.concat(states))
        return torch.stack(new_outs), feedback
    
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
    


def main():
    N_MODULES = 2
    HDIM = 2
    SEQ_LEN = 3000
    from acds.archetypes import InterconnectionRON
    from acds.networks.connection_matrices import cycle_matrix
    archetypes = [
        InterconnectionRON(HDIM, HDIM, dt=1, gamma=0.9, epsilon=0.5)
        for _ in range(N_MODULES)
    ]
    connection_matrix = cycle_matrix(N_MODULES)
    net = ArchetipesNetwork(archetypes, connection_matrix)
    print(net.archetipes)
    for i in tqdm(range(100)):
        x = torch.randn((SEQ_LEN, HDIM))
        initial_states = torch.zeros((N_MODULES, 2, HDIM))
        states, ins = net(x, initial_states)
        
    


if __name__ == "__main__":
    main()