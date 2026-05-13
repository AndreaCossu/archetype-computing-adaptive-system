import math
import torch
import sys
sys.path.append("..")
from typing import Optional, Sequence
from torch import nn
from acds.archetypes import InterconnectionRON as RON
from torch.func import functional_call, vmap
from einops import einsum, rearrange, repeat
from functools import partial
from acds.networks.utils import stack_state


# Utility class to make stuff work:
# torch.func.functional_call requires a callable nn.Module as a function, but we want to call RON.cell 

class Cell(nn.Module):
    def __init__(self, ron):
        super().__init__()
        self.ron = ron
    def __call__(self, *args):
        return self.ron.cell(*args)


class ArchetipesNetwork(nn.Module):

    connection_weights: torch.Tensor    
    input_mask: torch.Tensor

    def __init__(
        self,
        archetypes: Sequence[RON],
        connection_topology: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        output_mask: Optional[torch.Tensor] = None,
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
        self.register_buffer("connection_weights", connection_topology)
        # init connection topology
        self.register_buffer("connection_scaling",torch.where( self.connection_weights.sum(1).float() > 0, 1.0 / self.connection_weights.sum(1).float(), 0. )) # scale by the number of input modules for each row
        wm = torch.empty(self.n_modules, self.n_modules, self.n_hid, self.n_hid).uniform_(-2, 2) # one connection matrix for each pair of modules
        spec_rad = torch.vmap(torch.linalg.eigvals)(rearrange(wm, "m1 m2 n_h1 n_h2 -> (m1 m2) n_h1 n_h2")).abs().amax(1)
        self.wm = einsum(wm, 1 / rearrange(spec_rad, "(m1 m2) -> m1 m2", m1 = math.isqrt(len(spec_rad))), "m1 m2 n_h1 n_h2, m1 m2 -> m1 m2 n_h1 n_h2") * rho_m
        # init input and output mask
        if input_mask is None:
            input_mask = torch.ones(self.n_modules)
        if output_mask is None:
            output_mask = torch.ones(self.n_modules)
        self.register_buffer("input_mask", input_mask)
        self.register_buffer("output_mask", output_mask)

    def _step(self, x:torch.Tensor, prev_states:torch.Tensor, prev_outs:torch.Tensor, batched):
        """Perform one step of forward pass
        Args:
            x (Tensor of shape (h_dim)) or (batch_size, h_dim): external input at time t
            prev_states (Tensor of shape (n_modules, n_states, h_dim) or (n_modules, batch_size, n_states, h_dim)): state(s) for each archetipe at time t-1, which are also the outputs
            prev_outs (Tensor of shape (n_modules, h_dim) or (n_modules, batch_size, h_dim)): output of the models in the previous timestep, (e.g. h for ESN or h_y for RON)
        """

        # compute feedback from the modular connections
        ic_feedback = einsum(self.wm, prev_outs, "n_modules_in n_modules_out n_hid n_hid, n_modules_out ... n_hid -> n_modules_in ... n_modules_out n_hid") # transform the outputs before feeding them back
        ic_feedback_masked = einsum(self.connection_weights, ic_feedback, "n_modules_in n_modules_out, n_modules_in ... n_modules_out n_hid -> n_modules_in ... n_hid") # inter-connection feedback
        ic_feedback_scaled = einsum(ic_feedback_masked, self.connection_scaling, "n_modules ... n_hid, n_modules -> n_modules ... n_hid") # rescale the summed states feedback
        
        # duplicate and mask the external input
        x = einsum(x, self.input_mask, "..., n_modules -> n_modules ...")

        @partial(vmap, in_dims=(None, 0, 0, 0, 0, 0)) # call all modules in parallel
        def call_module(model, params, buffers, x, hs, feedback):
            new_states = functional_call(model, (params, buffers), (x, hs[:, 0] if batched else hs[0], hs[:, 1] if batched else hs[1], feedback))
            return torch.stack(new_states, dim=-2) # unbatched: (2, h_dim) batched (batch_size, 2, hdim)
        return call_module(self.archetype_structure, self.archetipes_params, self.archetipes_buffers, x, prev_states, ic_feedback_scaled), ic_feedback_scaled

    def forward(self, x:torch.Tensor, initial_states=None, initial_outs=None):
        """Forward of the network

        Args:
            x (torch.Tensor): input sequence of shape (seq_len, in_dim) or (batch_size, seq_len, in_dim)
            initial_states (torch.Tensor): Initial states of shape (n_modules, 2, h_dim) or (n_modules, batch_size, 2, h_dim), where 2 is the n. of states in a RON model
            initial_outs (torch.Tensor, optional): Initial outputs of the networks, of shape (n_modules, h_dim) or (n_modules, batch-size, h_dim) If None, they are initialized to torch.zeros. Defaults to None.

        Returns:
            (state_list, input_list): list of states h_i and interconnection inputs for each 
        """
        batched = (len(x.shape)==3)
        fb_list = []

        if initial_states is None:
            initial_states = torch.zeros((self.n_modules, 2, self.n_hid))
            if batched:
                initial_states = repeat(initial_states, "n_modules ... -> n_modules bs ...", bs=x.shape[0])
        states = initial_states
        state_list = []

        if initial_outs is None:
            outs = torch.zeros((self.n_modules, x.shape[0], self.n_hid)) if batched else torch.zeros((self.n_modules, self.n_hid))

        if batched:
            x = torch.permute(x, dims=(1, 0, 2))   
        for x_t in x:
            states, fbs = self._step(x_t, states, outs, batched)
            outs = states[:, :, 0] if batched else states[:, 0]
            state_list.append(states)
            fb_list.append(fbs) 
        return torch.stack(state_list), torch.stack(fb_list)

    def __repr__(self) -> str:
        return super().__repr__() + f"\nConnection weights:\n{self.connection_weights.__repr__()} \nInput mask: {self.input_mask.__repr__()}\nOutput mask: {self.output_mask.__repr__()}"
    


def main():
    import einops
    N_MODULES = 5
    HDIM = 3
    SEQ_LEN = 3000
    from acds.archetypes import InterconnectionRON
    from acds.networks.connection_matrices import cycle_matrix
    archetypes = [
        InterconnectionRON(HDIM, HDIM, dt=1, gamma=0.9, epsilon=0.5)
        for _ in range(N_MODULES)
    ]
    connection_matrix = cycle_matrix(N_MODULES)
    net = ArchetipesNetwork(archetypes, connection_matrix)
    print(net)
    x = torch.randn((SEQ_LEN, HDIM))
    initial_states = torch.zeros((N_MODULES, 2, HDIM))
    states, fbs = net(x, initial_states)
    print(states.shape)

    # batched 
    BATCH_SIZE = 64

    x = einops.repeat(x, " ... ->  bs ...", bs=BATCH_SIZE)

    initial_states = einops.repeat(initial_states, "n_modules ... -> n_modules bs ...", bs=BATCH_SIZE)
    states, fbs = net(x, initial_states)
    print(states.shape)
        
    


if __name__ == "__main__":
    main()