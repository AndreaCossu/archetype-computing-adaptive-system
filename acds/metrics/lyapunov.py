"""
Lyapunov exponent computation for discrete-time, input-driven RNNs.

Provides a unified ``compute_lyapunov`` that attempts the fast C/ctypes
implementation first, falling back to a pure-Python/NumPy version when the
shared library is unavailable.

References:
    G. Benettin et al., "Lyapunov characteristic exponents …", Meccanica, 1980.
    A. Pikovsky & A. Politi, *Lyapunov Exponents*, Cambridge Univ. Press, 2016.
"""

from __future__ import annotations

import ctypes
import logging
import os
from typing import Optional

import numpy as np
from numpy.linalg import qr
from acds.networks.utils import unstack_state

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Locate the C shared library (best-effort, non-fatal)
# ---------------------------------------------------------------------------
_LIB_DIR = os.path.join(os.path.dirname(__file__), "lyapunov_c")
_LIBLYAPUNOV: Optional[ctypes.CDLL] = None

for _name in ("liblyapunov.so", "liblyapunov.dylib"):
    _path = os.path.join(_LIB_DIR, _name)
    if os.path.isfile(_path):
        try:
            _LIBLYAPUNOV = ctypes.CDLL(_path)
        except OSError:
            pass
        break

# Also try system-wide (original behaviour of metrics.py)
if _LIBLYAPUNOV is None:
    try:
        _LIBLYAPUNOV = ctypes.CDLL("liblyapunov.so")
    except OSError:
        pass

if _LIBLYAPUNOV is None:
    logger.info(
        "C Lyapunov library not found; will use pure-Python fallback."
    )


# ---------------------------------------------------------------------------
# Pure-Python implementation (ported from lyap_discreteRNN.py) for ESNs
# ---------------------------------------------------------------------------

def _compute_lyapunov_esn(
    nl: int,
    W: np.ndarray,
    V: np.ndarray,
    b: np.ndarray,
    h_traj: np.ndarray,
    u_traj: np.ndarray,
    fb_traj: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Benettin algorithm with QR decomposition (pure NumPy).

    Args:
        nl: Number of Lyapunov exponents to compute.
        W: Recurrent weight matrix (N, N).
        V: Input weight matrix (N, input_dim).
        b: Bias vector (N,).
        h_traj: Hidden-state trajectory (T, N).
        u_traj: Input trajectory (T, input_dim).
        fb_traj: Optional feedback trajectory (T, N).

    Returns:
        Array of Lyapunov exponents of shape (nl,).
    """
    N = h_traj.shape[1]
    T = h_traj.shape[0]

    V_tan = np.random.randn(N, nl)
    V_tan, _ = qr(V_tan, mode="reduced")

    log_norms = np.zeros(nl)

    for t in range(T - 1):
        x = W @ h_traj[t] + V @ u_traj[t] + b
        if fb_traj is not None:
            x += fb_traj[t]
        phi_prime = 1.0 - np.tanh(x) ** 2
        V_tan = phi_prime[:, np.newaxis] * (W @ V_tan)
        Q, R = qr(V_tan, mode="reduced")
        log_norms += np.log(np.abs(np.diag(R)))
        V_tan = Q

    return log_norms / (T - 1)



# ---------------------------------------------------------------------------
# (Future) Pure-Python implementation for general RONs (not yet implemented)
# ---------------------------------------------------------------------------

def _compute_lyapunov_ron(
    nl: int,
    W: np.ndarray,
    V: np.ndarray,
    b: np.ndarray,
    gamma: np.ndarray,
    epsilon: np.ndarray,
    dt: float,
    h_traj: np.ndarray,
    u_traj: np.ndarray,
    fb_traj: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Benettin algorithm with QR decomposition (pure NumPy) for RONs.

    The RON state is :math:`(h_y, h_z) \in \mathbb{R}^{2N}`.  The
    forward-Euler dynamics are:

    .. math::
        h_z' = h_z + dt \cdot (\tanh(h_y W + x V + b + fb) - \gamma\,h_y - \varepsilon\,h_z)

        h_y' = h_y + dt \cdot h_z'

    Because the pre-activation uses the convention :math:`h_y W` (row-vector
    times weight matrix), the Jacobian of :math:`\tanh(h_y W + \ldots)`
    with respect to :math:`h_y` is :math:`\mathrm{diag}(D_t)\, W^\top`.

    The linearised tangent-vector propagation is therefore:

    .. math::
        \delta h_z' = \delta h_z + dt \big(D_t \odot (W^\top \,\delta h_y) - \gamma\,\delta h_y - \varepsilon\,\delta h_z\big)

        \delta h_y' = \delta h_y + dt \cdot \delta h_z'

    followed by QR re-orthonormalisation at every step.

    Args:
        nl: Number of Lyapunov exponents to compute (max 2*N).
        W: Effective recurrent weight matrix (N, N).
            Should be ``h2h - diffusive_matrix`` when a diffusive term is
            present.
        V: Input weight matrix (input_dim, N).
        b: Bias vector (N,).
        gamma: Damping factor – scalar or array of shape (N,).
        epsilon: Stiffness factor – scalar or array of shape (N,).
        dt: Integration time step.
        h_traj: Hidden-state (:math:`h_y`) trajectory (T, N).
        u_traj: Input trajectory (T, input_dim).
        fb_traj: Optional feedback trajectory (T, N).

    Returns:
        Array of Lyapunov exponents of shape (nl,).
    """
    N = h_traj.shape[1]
    T = h_traj.shape[0]
    state_dim = 2 * N

    # Ensure gamma/epsilon are (N,) arrays
    gamma = np.atleast_1d(np.asarray(gamma, dtype=np.float64)).ravel()
    epsilon = np.atleast_1d(np.asarray(epsilon, dtype=np.float64)).ravel()
    if gamma.size == 1:
        gamma = np.full(N, gamma[0])
    if epsilon.size == 1:
        epsilon = np.full(N, epsilon[0])

    # Initialise tangent vectors in the full (h_y, h_z) phase space
    V_tan = np.random.randn(state_dim, nl)
    V_tan, _ = qr(V_tan, mode="reduced")

    V_y = V_tan[:N, :]   # (N, nl) – perturbation of h_y
    V_z = V_tan[N:, :]   # (N, nl) – perturbation of h_z

    log_norms = np.zeros(nl)
    W_T = W.T  # (N, N) – precompute transpose for Jacobian

    for t in range(T - 1):
        # Pre-activation (h @ W convention, matching the RON forward pass)
        s = h_traj[t] @ W + u_traj[t] @ V + b
        if fb_traj is not None:
            s += fb_traj[t]

        D_t = 1.0 - np.tanh(s) ** 2  # (N,)

        # Propagate tangent vectors through the linearised dynamics
        WtVy = W_T @ V_y  # (N, nl)
        new_V_z = V_z + dt * (
            D_t[:, np.newaxis] * WtVy
            - gamma[:, np.newaxis] * V_y
            - epsilon[:, np.newaxis] * V_z
        )
        new_V_y = V_y + dt * new_V_z

        # QR re-orthonormalisation
        V_tan_full = np.vstack([new_V_y, new_V_z])  # (2N, nl)
        Q, R = qr(V_tan_full, mode="reduced")
        log_norms += np.log(np.abs(np.diag(R)))

        V_y = Q[:N, :]
        V_z = Q[N:, :]

    return log_norms / (T - 1)

# ---------------------------------------------------------------------------
# C/ctypes implementation for ESNs (ported from lyap_discreteRNN.c)
# ---------------------------------------------------------------------------

def _compute_lyapunov_c(
    nl: int,
    W: np.ndarray,
    V: np.ndarray,
    b: np.ndarray,
    h_traj: np.ndarray,
    u_traj: np.ndarray,
    fb_traj: np.ndarray,
) -> np.ndarray:
    """Compute Lyapunov exponents using the compiled C library via ctypes.

    Raises ``RuntimeError`` if the C library is not available.
    """
    if _LIBLYAPUNOV is None:
        raise RuntimeError("C Lyapunov library is not loaded.")

    N = h_traj.shape[1]
    n_steps = h_traj.shape[0]
    input_dim = u_traj.shape[1] if u_traj.ndim > 1 else 1

    W_c = np.ascontiguousarray(W, dtype=np.float64)
    V_c = np.ascontiguousarray(V, dtype=np.float64)
    b_c = np.ascontiguousarray(b, dtype=np.float64)
    h_c = np.ascontiguousarray(h_traj, dtype=np.float64)
    u_c = np.ascontiguousarray(u_traj, dtype=np.float64)
    fb_c = np.ascontiguousarray(fb_traj, dtype=np.float64)

    lyap_exponents = (ctypes.c_double * nl)()

    _LIBLYAPUNOV.compute_lyapunov(
        N,
        nl,
        n_steps,
        input_dim,
        W_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        V_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        b_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        h_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        u_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        fb_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        lyap_exponents,
    )

    return np.array([lyap_exponents[i] for i in range(nl)])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_lyapunov_ron(
    nl: int,
    W: np.ndarray,
    V: np.ndarray,
    b: np.ndarray,
    gamma: np.ndarray,
    epsilon: np.ndarray,
    dt: float,
    h_traj: np.ndarray,
    u_traj: np.ndarray,
    fb_traj: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute Lyapunov exponents for a discrete-time RON.

    Pure-Python implementation using the Benettin algorithm with QR
    re-orthonormalisation.  The RON phase space is 2N-dimensional
    (position :math:`h_y` + velocity :math:`h_z`), so up to 2N exponents
    can be requested.

    Args:
        nl: Number of Lyapunov exponents (max 2*N).
        W: Effective recurrent weight matrix (N, N).  Usually
            ``h2h - diffusive_gamma * I``.
        V: Input weight matrix (input_dim, N).
        b: Bias vector (N,).
        gamma: Damping factor – scalar or (N,).
        epsilon: Stiffness factor – scalar or (N,).
        dt: Integration time step.
        h_traj: :math:`h_y` trajectory (T, N).
        u_traj: Input trajectory (T, input_dim).
        fb_traj: Optional feedback trajectory (T, N).

    Returns:
        Array of Lyapunov exponents of shape (nl,).
    """
    return _compute_lyapunov_ron(
        nl, W, V, b, gamma, epsilon, dt, h_traj, u_traj, fb_traj
    )


def compute_lyapunov(
    nl: int,
    W: np.ndarray,
    V: np.ndarray,
    b: np.ndarray,
    h_traj: np.ndarray,
    u_traj: np.ndarray,
    fb_traj: Optional[np.ndarray] = None,
    *,
    prefer_c: bool = True,
) -> np.ndarray:
    """Compute Lyapunov exponents for a discrete-time input-driven RNN.

    Attempts the C implementation first (faster), falling back to pure Python
    if the shared library is unavailable or ``prefer_c`` is ``False``.

    Args:
        nl: Number of Lyapunov exponents to compute.
        W: Recurrent weight matrix (N, N).
        V: Input weight matrix (N, input_dim).
        b: Bias vector (N,).
        h_traj: Hidden-state trajectory (T, N).
        u_traj: Input trajectory (T, input_dim).
        fb_traj: Optional feedback trajectory (T, N).
        prefer_c: If True (default), use C library when available.

    Returns:
        Array of Lyapunov exponents of shape (nl,).
    """
    if prefer_c and _LIBLYAPUNOV is not None and fb_traj is not None:
        try:
            return _compute_lyapunov_c(nl, W, V, b, h_traj, u_traj, fb_traj)
        except Exception:
            logger.warning(
                "C Lyapunov computation failed; falling back to Python.",
                exc_info=True,
            )

    return _compute_lyapunov_esn(nl, W, V, b, h_traj, u_traj, fb_traj)


def compute_lyapunov_from_model(
    model,
    trajectory: np.ndarray,
    inputs: np.ndarray,
    feedbacks: np.ndarray,
    n_lyap: int,
    transient: int = 4000,
    model_type = "esn",
    **ron_kwargs
) -> list[list[float]]:
    """Compute Lyapunov exponents for every module in an ArchetipesNetwork.

    This is a convenience wrapper that extracts per-module weights and
    trajectories from the model and delegates to :func:`compute_lyapunov`
    (ESN) or :func:`compute_lyapunov_ron` (RON).

    Args:
        model: An ``ArchetipesNetwork`` instance.
        trajectory: Trajectory of shape (N_steps, N_modules, N_h).
        inputs: Inputs of shape (N_steps, N_inp) — shared across modules.
        feedbacks: Feedbacks of shape (N_steps, N_modules, N_h).
        n_lyap: Number of Lyapunov exponents to compute.
        transient: Number of initial steps to discard.
        model_type: ``"esn"`` or ``"ron"``.
        **ron_kwargs: Extra keyword arguments required when
            ``model_type="ron"``:

            - **dt** (*float*, required): Integration time step used by the
              RON.
            - **diffusive_gamma** (*float*, default ``0.0``): Coefficient
              of the diffusive term subtracted from ``h2h``.

    Returns:
        List of arrays, one per module, each of shape (n_lyap,).
    """

    exponents = []
    for module_idx, module in enumerate(
        unstack_state(model.archetipes_params, model.archetipes_buffers)
    ):

        trajectory_i = trajectory[:, module_idx, :]
        feedbacks_i = feedbacks[:, module_idx, :]

        if model_type == "esn":
            W = module['h2h'].detach().numpy()
            V = module['x2h'].detach().numpy()
            b = module['bias'].detach().numpy()

            exp = compute_lyapunov(
                n_lyap, W, V, b, trajectory_i, inputs, feedbacks_i
            )

        elif model_type == "ron":
            dt = ron_kwargs.get("dt")
            diffusive_gamma = ron_kwargs.get("diffusive_gamma", 0.0)
            if dt is None:
                raise ValueError(
                    "'dt' must be provided via **ron_kwargs for "
                    "model_type='ron'."
                )

            # Get RON effective parameters 
            h2h = module["h2h"].detach().numpy()
            W = h2h - diffusive_gamma * np.eye(model.n_hid)
            V = module["x2h"].detach().numpy()
            b = module["bias"].detach().numpy()
            gamma_val = module["gamma"].detach().numpy()
            epsilon_val = module["epsilon"].detach().numpy()

            exp = _compute_lyapunov_ron(
                n_lyap, W, V, b, gamma_val, epsilon_val, dt,
                trajectory_i, inputs, feedbacks_i,
            )

        else:
            raise NotImplementedError(
                f"Model type {model_type} not supported yet "
                "for Lyapunov computation."
            )

        exponents.append(exp.tolist())

    return exponents


__all__ = ["compute_lyapunov", "compute_lyapunov_ron", "compute_lyapunov_from_model"]
