import torch
import sys

from tqdm import tqdm
sys.path.append("..")
from typing import List, Sequence
from torch import nn
from acds.archetypes import InterconnectionRON as RON
from torch.func import functional_call, vmap
from einops import einsum, rearrange # I like using einops because it allows strings as axes names instead of just letters, and for other stuff
from functools import partial
from acds.networks.utils import stack_state
# from torch import einsum

# Utility class to make stuff work:
# torch.func.functional_call requires a callable nn.Module as a function, but we want to call RON.cell 

class Cell(nn.Module):
    def __init__(self, ron):
        super().__init__()
        self.ron = ron
    def __call__(self, *args):
        return self.ron.cell(*args)



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
        params, buffers = stack_state(archetypes)
        self.archetype_structure = Cell(archetypes[0])
        self.archetipes_params = params
        self.archetipes_buffers = buffers 
        self.n_hid = archetypes[0].n_hid
        self.n_inp = archetypes[0].n_inp
        self.n_modules = len(archetypes)
        self.interconnection_matrix = nn.Parameter(connection_matrix)
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

        prev_outs = self.wm(prev_outs) # transform the outputs before feeding them back
        ic_feedback = einsum(self.interconnection_matrix, prev_outs, "n_modules_in n_modules_out, n_modules_out n_hid -> n_modules_in n_hid") # inter-connection feedback

        @partial(vmap, in_dims=(None, 0, 0, None, 0, 0))
        def call_module(model, params, buffers, x, hs, feedback):
            new_states = functional_call(model, (params, buffers), (x, hs[0], hs[1], feedback))
            return torch.stack(new_states)
        return call_module(self.archetype_structure, self.archetipes_params, self.archetipes_buffers, x, prev_states, ic_feedback), ic_feedback
        

         
        

        


    
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