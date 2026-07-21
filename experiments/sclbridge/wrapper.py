from typing import Tuple
import torch
from acds.archetypes import RandomizedOscillatorsNetwork


class AvalancheRON(RandomizedOscillatorsNetwork):
    """Adapt :class:`RandomizedOscillatorsNetwork` for Avalanche strategies.

    :param n_inp: Number of input features per time step.
    :param n_hid: Reservoir hidden size.
    :param dt: Reservoir integration step.
    :param gamma: Damping parameter or sampling range.
    :param epsilon: Stiffness parameter or sampling range.
    :param rho: Spectral radius.
    :param input_scaling: Input scaling factor.
    :param reservoir_scaler: Scaling for structured reservoirs.
    :param sparsity: Reservoir sparsity.
    :param device: Torch device.
    :param n_classes: Number of output classes.
    """
    def __init__(self, n_inp: int, n_hid: int, dt: float, 
                 gamma: float | Tuple[float, float], epsilon: float | Tuple[float, float], 
                 rho: float = 0.99, input_scaling: float = 1, 
                 reservoir_scaler=0, sparsity=0, device="cpu", n_classes: int = 10):
        self.n_inp = n_inp
        super().__init__(n_inp, n_hid, dt, gamma, epsilon, 0, rho, input_scaling, "full", reservoir_scaler, sparsity, device)

        self.classifier = torch.nn.Linear(n_hid, n_classes).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # type: ignore
        """Run the reservoir and classify the final hidden state.

        :param x: Input tensor accepted by Avalanche minibatches.
        :return: Class logits.
        """
        x = x.view(x.size(0), -1, self.n_inp)
        hs, _ = super().forward(x, None)

        out = self.classifier(hs[:, -1])
        return out
