# =====================================================================
# EX2D2 — thermo-mechanical ACTUATOR (four-phase {void, Ti, Cu, Steel}).
# Simultaneous analysis-and-design (m-PIGP) on a single uniform grid with
# a sparse direct adjoint, Helmholtz filtering, Ti-Steel interface
# exclusion, and a switchable physics model (see PHYSICS MODEL SELECTION
# in the config block):
#   baseline      : linear elasticity + properties anchored at T = TD
#   high-fidelity : quadratic-Hencky + temperature-dependent properties
# Companion script: run_gripper.py (EX2D1).  See README.md for usage.
# =====================================================================

import os
import json
import torch
import torch.nn.functional as F
import copy
import numpy as np
import math
import time
import shutil
from datetime import datetime
from models.lmgp import LMGP
from gpytorch.settings import cholesky_jitter
from tqdm import tqdm
from utils.utils_general import set_seed0
from utils.get_training_data import get_data_EX2D2_thermomech_actuator_four_phase_source as get_data

# =====================================================================
# Enable TF32 for faster float32 matmul on Ampere+ GPUs
# =====================================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


# =====================================================================
# Quadratic-Hencky (logarithmic-strain) constitutive law
# -------------------------------------------------------------------
# Closed-form 2x2 material Hencky strain eps_H = 1/2 log(F^T F), parameterised
# through q = r^2 with the atanh kernel so the eigenvalue-coalescence NaN at
# C = I is avoided and gradients (and the adjoint HVP's 2nd derivative) stay
# finite.  (Verified against an eigendecomposition reference to 1e-15.)
#
# THERMO-MECHANICAL thermal split:  for ISOTROPIC thermal expansion the
# multiplicative split  F = F_e * (vartheta I),  vartheta = exp(alpha (T-T_inf)),
# gives  log U = log U_e + alpha(T-T_inf) I  EXACTLY (vartheta I commutes), so
# the elastic Hencky strain is the total Hencky strain minus an ADDITIVE,
# isotropic log-eigenstrain  eps_th = alpha (T - T_inf)  on the normal
# components only (no thermal shear).  This additive split is EXACT at finite
# strain — unlike the Green-Lagrange eigenstrain of a St. Venant-Kirchhoff
# formulation, which is only first-order in the thermal strain.  (Verified: eps_H(F) - eps_th I == eps_H(F_e)
# to 1e-16.)
# =====================================================================

def _f_atanh(q):
    """f(q) = atanh(sqrt(q))/sqrt(q), smooth on [0,1); double-where keeps the
    unused branch NaN-free in the backward pass."""
    small = q < 1e-8
    q_safe = torch.where(small, torch.full_like(q, 0.5), q)
    sq = torch.sqrt(q_safe)
    exact = torch.atanh(sq) / sq
    series = 1.0 + q / 3.0 + q * q / 5.0 + q**3 / 7.0 + q**4 / 9.0
    return torch.where(small, series, exact)


# distortion clamp r^2 = q in [0, Q_MAX]; caps the 1/(1-q) blow-up of the atanh
# kernel's 2nd derivative (the adjoint HVP) on near-degenerate elements.
Q_MAX = 1.0 - 1e-3


def _hencky_eps_from_C(C11, C22, C12):
    """In-plane material Hencky strain eps_H = 1/2 log(C) from the right
    Cauchy-Green components (C11, C22, C12).  Core shared by the H-input
    kinematics wrapper and the strain-space (from-C) adjoint tangent, so the
    two entry points are byte-for-byte the SAME log-strain law."""
    I1   = C11 + C22
    detC = (C11 * C22 - C12 * C12).clamp_min(1e-12)
    disc = (C11 - C22) ** 2 + 4.0 * C12 * C12
    q    = (disc / (I1 * I1)).clamp(0.0, Q_MAX)
    beta   = _f_atanh(q) / I1
    tr_eps = 0.5 * torch.log(detC)
    alpha  = 0.5 * tr_eps - 0.5 * beta * I1
    eps11 = alpha + beta * C11
    eps22 = alpha + beta * C22
    gam12 = 2.0 * beta * C12
    return eps11, eps22, gam12


def hencky_strain_2d(H):
    """In-plane material Hencky strain eps_H = 1/2 log(F^T F).
    H : [...,2,2] displacement gradient dU_i/dX_j (total Lagrangian).
    Returns eps11, eps22, gam12 (engineering shear = 2 eps12), each [...]."""
    F11 = 1.0 + H[..., 0, 0]; F12 = H[..., 0, 1]
    F21 = H[..., 1, 0];       F22 = 1.0 + H[..., 1, 1]
    C11 = F11 * F11 + F21 * F21
    C22 = F12 * F12 + F22 * F22
    C12 = F11 * F12 + F21 * F22
    return _hencky_eps_from_C(C11, C22, C12)


def W_density_hencky_thermal(H, C1, C2, nu, eps_th):
    """Plane-stress quadratic-Hencky strain-energy density with an isotropic
    log-thermal eigenstrain eps_th subtracted from the normal log-strains.
    C1 = E/(1-nu^2), C2 = G = E/(2(1+nu)).  eps_th broadcastable to eps11."""
    e11, e22, g12 = hencky_strain_2d(H)
    e11 = e11 - eps_th
    e22 = e22 - eps_th
    t11 = C1 * (e11 + nu * e22)
    t22 = C1 * (nu * e11 + e22)
    t12 = C2 * g12
    return 0.5 * (t11 * e11 + t22 * e22 + t12 * g12)


def W_density_hencky_thermal_fromC(C11, C22, C12, C1, C2, nu, eps_th):
    """SAME plane-stress quadratic-Hencky energy as W_density_hencky_thermal,
    but parameterised directly by the right Cauchy-Green components instead of
    the displacement gradient.  Used by the Gauss-Newton adjoint tangent: with
    C advanced LINEARLY in the adjoint parameter (C = C0 + 2*tau*dE), the
    d2/dtau2 HVP returns the exact MATERIAL log-strain curvature dE:C_H:dE and
    the geometric/initial-stress term S:(grad l^T grad l) vanishes identically
    (it is the second derivative of E, which is zero on a strain-linear path).
    Equals the H-input energy exactly when C = (I+H)^T(I+H)."""
    e11, e22, g12 = _hencky_eps_from_C(C11, C22, C12)
    e11 = e11 - eps_th
    e22 = e22 - eps_th
    t11 = C1 * (e11 + nu * e22)
    t22 = C1 * (nu * e11 + e22)
    t12 = C2 * g12
    return 0.5 * (t11 * e11 + t22 * e22 + t12 * g12)


def W_density_linear_thermal(H, C1, C2, nu, eps_th):
    """Small-strain plane-stress energy (void branch of the Wang interpolation)
    with the same isotropic thermal eigenstrain subtracted."""
    e11 = H[..., 0, 0] - eps_th
    e22 = H[..., 1, 1] - eps_th
    g12 = H[..., 0, 1] + H[..., 1, 0]
    t11 = C1 * (e11 + nu * e22)
    t22 = C1 * (nu * e11 + e22)
    t12 = C2 * g12
    return 0.5 * (t11 * e11 + t22 * e22 + t12 * g12)


# =====================================================================
# Temperature-dependent thermal properties  kappa(rho,T), alpha(rho,T)
# ---------------------------------------------------------------------
# Per-phase MULTIPLICATIVE polynomial corrections in dT = T - T_ref:
#     kappa_i(T) = kappa[i] * poly(kappa_T_coef[i], dT)
#     alpha_i(T) = alpha[i] * poly(alpha_T_coef[i], dT)     (INSTANTANEOUS CTE)
# The thermal eigenstrain uses the EXACT integral of the instantaneous CTE,
#     eps_th = sum_i w_i * alpha[i] * [ P(T - T_ref) - P(T_inf - T_ref) ] ,
#     P = antiderivative of the polynomial,
# which reduces IDENTICALLY to alpha*(T - T_inf) for coef=[1.0], for ANY
# T_ref (the constant of integration cancels in the difference).
# Horner forms below work on tensors and python floats alike.
# =====================================================================

def _poly_eval(coef, x):
    """sum_k coef[k] * x**k  (Horner; x tensor or float)."""
    y = 0.0
    for c in reversed(coef):
        y = y * x + c
    return y


def _poly_deriv(coef, x):
    """d/dx of _poly_eval:  sum_k k*coef[k] * x**(k-1)."""
    y = 0.0
    for k in range(len(coef) - 1, 0, -1):
        y = y * x + k * coef[k]
    return y


def _poly_antideriv(coef, x):
    """Antiderivative with zero constant:  sum_k coef[k]/(k+1) * x**(k+1)."""
    y = 0.0
    for k in range(len(coef) - 1, -1, -1):
        y = y * x + coef[k] / (k + 1)
    return y * x


# =====================================================================
# Helmholtz PDE Filter — Sparse FEM, GPU-native, fully differentiable
# Works on arbitrary (unstructured) quad meshes.
# =====================================================================

def _spmm(A, x):
    """Sparse matrix × dense multiply that works for both CSR and COO."""
    if A.layout == torch.sparse_csr:
        return torch.sparse.mm(A, x)
    else:
        return torch.sparse.mm(A, x)


@torch.no_grad()
def _sparse_pcg_solve(H_sparse, b, inv_diag=None, x0=None,
                      tol=1e-6, max_iter=300):
    """
    Jacobi-preconditioned conjugate-gradient solver for  H x = b.

    Parameters
    ----------
    H_sparse : sparse [N, N]  (CSR or COO)
    b        : [N] or [N, C]  (multiple RHS solved simultaneously)
    inv_diag : [N] or None    — 1/diag(H), the Jacobi preconditioner.
               If None, falls back to unpreconditioned CG.
    """
    single = (b.dim() == 1)
    if single:
        b = b.unsqueeze(-1)

    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()
        if x.dim() == 1:
            x = x.unsqueeze(-1)

    r = b - _spmm(H_sparse, x)

    # Jacobi preconditioner: z = M⁻¹ r  where M = diag(H)
    if inv_diag is not None:
        z = inv_diag.unsqueeze(-1) * r
    else:
        z = r

    p = z.clone()
    rz_old = (r * z).sum(dim=0)  # [C]

    if (r * r).sum(dim=0).max().sqrt() < tol:
        return x.squeeze(-1) if single else x

    for _ in range(max_iter):
        Ap = _spmm(H_sparse, p)
        pAp = (p * Ap).sum(dim=0)             # [C]
        alpha = rz_old / (pAp + 1e-30)        # [C]

        x = x + alpha.unsqueeze(0) * p
        r = r - alpha.unsqueeze(0) * Ap

        r_norm = (r * r).sum(dim=0).max().sqrt()
        if r_norm < tol:
            break

        if inv_diag is not None:
            z = inv_diag.unsqueeze(-1) * r
        else:
            z = r

        rz_new = (r * z).sum(dim=0)
        beta = rz_new / (rz_old + 1e-30)
        p = z + beta.unsqueeze(0) * p
        rz_old = rz_new

    return x.squeeze(-1) if single else x


def _assemble_sparse_matrices(conn, node_coords, r_min, device, dtype):
    """
    Vectorised assembly of global sparse  H = r²K + M  and  M  matrices
    for 4-node bilinear quads with 2×2 Gauss quadrature.

    Parameters
    ----------
    conn         : LongTensor [N_elem, 4]  — element connectivity
    node_coords  : Tensor     [N_node, 2]  — nodal coordinates
    r_min        : float                   — Helmholtz filter radius
    device, dtype: target device / dtype

    Returns
    -------
    H_sparse, M_sparse : sparse CSR tensors  [N_node, N_node]
    H_diag             : dense [N_node] — diagonal of H for Jacobi precond.
    """
    N_elem = conn.shape[0]
    N_node = node_coords.shape[0]

    gp = 1.0 / math.sqrt(3.0)
    gauss_pts = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]

    # Element node coordinates: [N_elem, 4, 2]
    xe = node_coords[conn]

    ke = torch.zeros(N_elem, 4, 4, dtype=dtype, device=device)
    me = torch.zeros(N_elem, 4, 4, dtype=dtype, device=device)

    for xi, eta in gauss_pts:
        N_shape = torch.tensor(
            [(1 - xi) * (1 - eta) / 4, (1 + xi) * (1 - eta) / 4,
             (1 + xi) * (1 + eta) / 4, (1 - xi) * (1 + eta) / 4],
            dtype=dtype, device=device,
        )
        dN_dxi = torch.tensor(
            [-(1 - eta) / 4,  (1 - eta) / 4,
              (1 + eta) / 4, -(1 + eta) / 4],
            dtype=dtype, device=device,
        )
        dN_deta = torch.tensor(
            [-(1 - xi) / 4, -(1 + xi) / 4,
              (1 + xi) / 4,  (1 - xi) / 4],
            dtype=dtype, device=device,
        )

        # Jacobian components: [N_elem]
        J00 = (dN_dxi  * xe[:, :, 0]).sum(dim=1)
        J01 = (dN_dxi  * xe[:, :, 1]).sum(dim=1)
        J10 = (dN_deta * xe[:, :, 0]).sum(dim=1)
        J11 = (dN_deta * xe[:, :, 1]).sum(dim=1)

        detJ = J00 * J11 - J01 * J10       # [N_elem]
        inv_detJ = 1.0 / detJ

        # Physical derivatives: [N_elem, 4]
        dN_dx = inv_detJ.unsqueeze(1) * ( J11.unsqueeze(1) * dN_dxi  - J01.unsqueeze(1) * dN_deta)
        dN_dy = inv_detJ.unsqueeze(1) * (-J10.unsqueeze(1) * dN_dxi  + J00.unsqueeze(1) * dN_deta)

        # Element stiffness: ke += detJ * (dNdx⊗dNdx + dNdy⊗dNdy)   [N_elem, 4, 4]
        ke += detJ.unsqueeze(1).unsqueeze(2) * (
            dN_dx.unsqueeze(2) * dN_dx.unsqueeze(1) +
            dN_dy.unsqueeze(2) * dN_dy.unsqueeze(1)
        )

        # Element mass: me += detJ * N⊗N   [N_elem, 4, 4]
        N_col = N_shape.unsqueeze(0).expand(N_elem, -1)
        me += detJ.unsqueeze(1).unsqueeze(2) * (
            N_col.unsqueeze(2) * N_col.unsqueeze(1)
        )

    # H_e = r²K_e + M_e
    he = r_min ** 2 * ke + me               # [N_elem, 4, 4]

    # ---------- scatter into global sparse COO ----------
    row_idx = conn.unsqueeze(2).expand(-1, -1, 4).reshape(-1)   # [N_elem*16]
    col_idx = conn.unsqueeze(1).expand(-1, 4, -1).reshape(-1)   # [N_elem*16]
    indices = torch.stack([row_idx, col_idx], dim=0)

    H_sparse = torch.sparse_coo_tensor(
        indices, he.reshape(-1), (N_node, N_node),
    ).coalesce().to_sparse_csr()

    M_sparse = torch.sparse_coo_tensor(
        indices, me.reshape(-1), (N_node, N_node),
    ).coalesce().to_sparse_csr()

    # Extract diagonal of H for Jacobi preconditioning
    diag_mask = (row_idx == col_idx)
    diag_nodes = row_idx[diag_mask]
    diag_vals = he.reshape(-1)[diag_mask]
    H_diag = torch.zeros(N_node, dtype=dtype, device=device)
    H_diag.scatter_add_(0, diag_nodes, diag_vals)

    return H_sparse, M_sparse, H_diag


def _build_projection_matrices(conn, N_node, device, dtype):
    """
    Build sparse elem↔node projection matrices.

    N2E : [N_elem, N_node] — nodal → element  (mean of 4 nodes)
    E2N : [N_node, N_elem] — element → nodal   (row-normalised scatter)
    """
    N_elem = conn.shape[0]

    # --- N2E: each element = mean of its 4 nodes ---
    row = torch.arange(N_elem, device=device, dtype=torch.long).repeat_interleave(4)
    col = conn.reshape(-1)
    vals = torch.full((N_elem * 4,), 0.25, dtype=dtype, device=device)
    N2E = torch.sparse_coo_tensor(
        torch.stack([row, col]), vals, (N_elem, N_node),
    ).coalesce().to_sparse_csr()

    # --- E2N: transpose of N2E, then row-normalise ---
    E2N_unnorm = torch.sparse_coo_tensor(
        torch.stack([col, row]), vals, (N_node, N_elem),
    ).coalesce()
    row_sums = torch.sparse.sum(E2N_unnorm, dim=1).to_dense()   # [N_node]
    idx = E2N_unnorm.indices()
    E2N = torch.sparse_coo_tensor(
        idx, E2N_unnorm.values() / row_sums[idx[0]],
        (N_node, N_elem),
    ).coalesce().to_sparse_csr()

    return N2E, E2N


class _SparseCGFn(torch.autograd.Function):
    """
    Custom autograd for the CG solve:  x = H⁻¹ b.
    Backward:  ∂L/∂b = H⁻ᵀ ∂L/∂x = H⁻¹ ∂L/∂x   (H symmetric).
    Uses Jacobi-preconditioned CG for faster convergence.
    """

    @staticmethod
    def forward(ctx, b, H_sparse, H_inv_diag, cg_tol, cg_max_iter,
                ws_cache=None, ws_key=None):
        # warm start from the previous solve of the SAME (mesh, #columns)
        # system: pure initialisation -- tol/max_iter unchanged, CG residual
        # is monotone from any x0, so the result is equal-or-better than a
        # cold start (and typically converges in a handful of iterations
        # under Adam's per-epoch design drift).
        x0 = None
        if ws_cache is not None:
            x0 = ws_cache.get(('fwd',) + ws_key)
            if x0 is not None and x0.shape != b.shape:
                x0 = None
        x = _sparse_pcg_solve(H_sparse, b, inv_diag=H_inv_diag, x0=x0,
                              tol=cg_tol, max_iter=cg_max_iter)
        if ws_cache is not None:
            ws_cache[('fwd',) + ws_key] = x.detach().clone()
        ctx.H_sparse = H_sparse
        ctx.H_inv_diag = H_inv_diag
        ctx.cg_tol = cg_tol
        ctx.cg_max_iter = cg_max_iter
        ctx.ws_cache = ws_cache
        ctx.ws_key = ws_key
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x0 = None
        if ctx.ws_cache is not None:
            x0 = ctx.ws_cache.get(('bwd',) + ctx.ws_key)
            if x0 is not None and x0.shape != grad_output.shape:
                x0 = None
        grad_b = _sparse_pcg_solve(
            ctx.H_sparse, grad_output, inv_diag=ctx.H_inv_diag, x0=x0,
            tol=ctx.cg_tol, max_iter=ctx.cg_max_iter,
        )
        if ctx.ws_cache is not None:
            ctx.ws_cache[('bwd',) + ctx.ws_key] = grad_b.detach().clone()
        return grad_b, None, None, None, None, None, None


class HelmholtzFilterSparse(torch.nn.Module):
    """
    GPU-native, fully differentiable Helmholtz PDE filter for
    **arbitrary (unstructured) quad meshes**.

    Solves per mesh:
        (r² K + M) ρ̃_node = M ρ_node
    with Jacobi-preconditioned sparse CG (CSR format), using elem↔node
    projections to go from / to element-centred densities.

    Assembled matrices are cached per mesh_key so the O(N_elem) assembly
    cost is paid only once per unique mesh (51 meshes → 51 assemblies
    across the whole 20 000 epoch run).
    """

    def __init__(self, r_min, device="cuda", dtype=torch.float32,
                 cg_tol=1e-6, cg_max_iter=300):
        super().__init__()
        self.r_min = r_min
        self.device = device
        self.dtype = dtype
        self.cg_tol = cg_tol
        self.cg_max_iter = cg_max_iter
        self._cache = {}           # mesh_key → (H, M, H_inv_diag, N2E, E2N)
        self._ws = {}              # warm-start cache: previous CG solutions
                                   # keyed by (fwd/bwd, mesh_key, n_columns)

        min_feature = 2 * r_min * math.sqrt(3)
        print(f"[HelmholtzFilter-Sparse] r_min = {r_min:.2f}, "
              f"min feature ≈ {min_feature:.1f} (phys. units)")

    def _get_or_build(self, conn, node_coords, mesh_key):
        """Return cached (H, M, H_inv_diag, N2E, E2N) or assemble & cache."""
        if mesh_key in self._cache:
            return self._cache[mesh_key]

        N_node = node_coords.shape[0]
        H, M, H_diag = _assemble_sparse_matrices(
            conn, node_coords, self.r_min, self.device, self.dtype,
        )
        # Jacobi preconditioner: inv(diag(H)), clamped to avoid division by ~zero
        H_inv_diag = 1.0 / H_diag.clamp(min=1e-30)

        N2E, E2N = _build_projection_matrices(
            conn, N_node, self.device, self.dtype,
        )
        entry = (H, M, H_inv_diag, N2E, E2N)
        self._cache[mesh_key] = entry
        return entry

    def forward(self, rho_elem, conn, node_coords, mesh_key="default"):
        """
        Filter element-centred densities.

        Parameters
        ----------
        rho_elem    : [N_elem, num_phase] or [N_elem]
        conn        : LongTensor [N_elem, 4]   — element connectivity
        node_coords : Tensor     [N_node, 2]   — nodal coordinates
        mesh_key    : hashable                  — cache key for this mesh

        Returns
        -------
        Filtered density, same shape as input (differentiable w.r.t. rho_elem).
        """
        squeeze_at_end = False
        if rho_elem.dim() == 1:
            rho_elem = rho_elem.unsqueeze(-1)
            squeeze_at_end = True

        H, M, H_inv_diag, N2E, E2N = self._get_or_build(
            conn, node_coords, mesh_key,
        )

        # elem → node  (E2N is detached sparse constant → grad flows to rho_elem)
        rho_node = _spmm(E2N, rho_elem)                      # [N_node, C]

        # RHS = M @ rho_node
        rhs = _spmm(M, rho_node)                             # [N_node, C]

        # Solve  H @ rho_filt_node = rhs   (Jacobi-preconditioned CG,
        # warm-started from the previous epoch's solution on this mesh)
        rho_filt_node = _SparseCGFn.apply(
            rhs, H, H_inv_diag, self.cg_tol, self.cg_max_iter,
            self._ws, (mesh_key, rho_elem.shape[1]),
        )

        # node → elem
        rho_filt_elem = _spmm(N2E, rho_filt_node)            # [N_elem, C]

        if squeeze_at_end:
            rho_filt_elem = rho_filt_elem.squeeze(-1)

        return rho_filt_elem


def build_helmholtz_filter(r_min, device, dtype=torch.float32, **kw):
    """Build sparse Helmholtz filter (mesh-independent constructor)."""
    return HelmholtzFilterSparse(
        r_min=r_min, device=device, dtype=dtype, **kw,
    )


# =====================================================================
# calculate_T_mean
# =====================================================================

def calculate_T_mean(model_list):
    collocation_x = model_list[0].collocation_x.clone()

    for model in model_list:
        model.train()

    m_col_T = model_list[2].mean_module_NN_All(collocation_x)

    # --- T adjoint GP ---
    g_t = model_list[2].independent_kernels[0](model_list[2].train_inputs_per_output[0], collocation_x).evaluate().detach()

    # ====================================================
    # Compute Cholesky decompositions and offsets
    # ====================================================
    if model_list[2].chol_decomp is None:
        with cholesky_jitter(1e-5):
            model_list[2].chol_decomp = []
            for i in range(model_list[2].num_output):
                K_i = model_list[2].independent_kernels[i](model_list[2].train_inputs_per_output[i]).evaluate().detach()
                L_i = torch.linalg.cholesky(K_i + 1e-5 * torch.eye(K_i.shape[0], device=K_i.device))
                model_list[2].chol_decomp.append(L_i)
    # --- Temperature ---
    K_inv_offset_T = torch.cholesky_solve(
        model_list[2].train_target_per_output[0].unsqueeze(-1)
        - model_list[2].mean_module_NN_All(model_list[2].train_inputs_per_output[0])[:, 0].unsqueeze(-1),
        model_list[2].chol_decomp[0]
    )

    Node_T = (m_col_T[:,0].unsqueeze(-1) + g_t.t() @ K_inv_offset_T)

    T_mean = Node_T.mean()

    return T_mean


# =====================================================================
# Fingerprint-cached GP cross-covariances
# ---------------------------------------------------------------------
# g = k(X_train_i, X).evaluate().detach() is a CONSTANT on a fixed mesh with
# frozen kernel hyperparameters (the same standing assumption under which
# chol_decomp is computed once and never refreshed).  Cache per
# (tag, output, mesh_key); the hyperparameter fingerprint (sum, |sum| per
# parameter tensor) auto-invalidates if the kernel parameters ever change,
# so a cache hit returns a tensor BIT-IDENTICAL to the per-epoch eval.
# Saves four [N_bc x N_node] kernel evaluations per epoch (u, v, T, T_adj;
# plus the phase kernels when design_flag).
# =====================================================================

def _cached_cross_cov(model, out_idx, X, MP, tag):
    kern = model.independent_kernels[out_idx]
    fp = tuple((float(p.detach().sum()), float(p.detach().abs().sum()))
               for p in kern.parameters())
    key = (tag, out_idx, MP.get('cur_mesh_key', 'default'))
    cache = MP.setdefault('_gcov_cache', {})
    hit = cache.get(key)
    if hit is not None and hit[0] == fp and hit[1].shape[1] == X.shape[0]:
        return hit[1]
    g = kern(model.train_inputs_per_output[out_idx], X).evaluate().detach()
    cache[key] = (fp, g)
    return g


# =====================================================================
# Displacement gradient at the integration points (total-Lagrangian)
# =====================================================================
def _disp_gradient(node_field, conn, B_T, N_elem_w):
    """
    Displacement gradient H_ij = ∂u_i/∂X_j at the integration points
    (total-Lagrangian: derivatives w.r.t. the REFERENCE coordinates X).

    node_field : [N_node, 2]   nodal vector field (u or λ)
    B_T        : [..., N_int, 2, 4]  shape-function gradients dN/dX
                 (the same operator already used for ∇T)
    returns    : [N_elem_w, N_int, 2, 2]
    """
    u_nodes = node_field[conn].reshape(N_elem_w, -1, 2)   # [N_elem, 4, 2]
    u_nodes = u_nodes.unsqueeze(1)                        # [N_elem, 1, 4, 2]
    # (B_T @ u)_{ji} = Σ_a dN_a/dX_j · u_a,i  =  (Hᵀ)_{ji}
    H_T = torch.matmul(B_T, u_nodes)                      # [N_elem, N_int, 2, 2]
    return H_T.transpose(-1, -2)                          # H_{ij} = u_{i,j}


# =====================================================================
# DIRECT ADJOINT (adjoint_mode='direct'):  K_T(u*) lambda = f_adj solved by a
# sparse direct factorization (torch-sla, cuDSS LDL^T on GPU / SuperLU on CPU).
#
# The adjoint problem is LINEAR; the only reason it was ever a minimisation is
# the NN parametrisation of lambda.  Solving it directly gives the EXACT
# full-tangent adjoint (geometric/initial-stress term retained), is safe for
# INDEFINITE K_T (LDL^T / LU), and removes the adjoint displacement net, its
# optimizer, and its loss term from the training loop entirely.
#
# K_T is assembled from batched per-element Hessians of the interpolated
# three-term Wang energy via torch.func.vmap(hessian) — no closed-form Hencky
# tangent needed.  Assembly verified against the training-loop nested-autograd
# HVP to machine precision (w^T K w rel. err 0, matvec 2e-16), and the
# torch-sla COO solve against a dense reference to 4e-14.
#
# Dirichlet: lambda = 0 at exactly the adjoint model's GP conditioning points
# (the same points where the DEM enforced the BC softly), matched to mesh
# nodes.  Override with MP['adjoint_fixed_dofs'] (LongTensor of global DOFs)
# if the conditioning points do not coincide with mesh nodes.
# =====================================================================

def _elem_energy_dofs(d, BT_e, g_e, c1_e, c2_e, eth_e, wdetJ_e, nu, thickness,
                      linelas):
    """Interpolated three-term Wang energy of ONE element as a function of its
    8 nodal DOFs d = [u0,v0,u1,v1,u2,v2,u3,v3] (vmapped over elements; the
    per-element Hessian of this scalar is the exact consistent tangent K_e).
    linelas=True: small-strain law — the Wang form collapses identically to
    psi_L(u) (gamma drops out; g_e is ignored) and K_e is the constant linear
    element stiffness (no geometric term)."""
    un = d.view(4, 2)
    H = torch.matmul(BT_e, un).transpose(-1, -2)              # [N_int, 2, 2]
    if linelas:
        W = W_density_linear_thermal(H, c1_e, c2_e, nu, eth_e)
    else:
        W = (W_density_hencky_thermal(g_e * H, c1_e, c2_e, nu, eth_e)
             + W_density_linear_thermal(H, c1_e, c2_e, nu, eth_e)
             - W_density_linear_thermal(g_e * H, c1_e, c2_e, nu, eth_e))
    return (W * wdetJ_e).sum() * thickness


def _chunked_elem_hessians(d_all, BT_e, g_e, c1_e, c2_e, eth_e, wdetJ_e,
                           nu, thickness, linelas, chunk=None):
    """Batched per-element tangents  K_e = d2 W_e / d d^2  via vmap(hessian),
    evaluated in element CHUNKS so the forward-over-reverse intermediates of
    the vmapped double differentiation never exceed O(chunk) elements of
    transient memory.  This is EXACT block-wise ASSEMBLY: the element tangents
    are mutually independent, so chunking changes nothing numerically (unlike
    blocking the SOLVE, where independent per-block solves drop the interface
    coupling of the elliptic operator = one block-Jacobi sweep = wrong lambda).
    chunk=None -> single batch (legacy behaviour)."""
    hess = torch.func.vmap(
        torch.func.hessian(_elem_energy_dofs, argnums=0),
        in_dims=(0, 0, 0, 0, 0, 0, 0, None, None, None))
    Ne = d_all.shape[0]
    if chunk is None or chunk >= Ne:
        return hess(d_all, BT_e, g_e, c1_e, c2_e, eth_e, wdetJ_e,
                    nu, thickness, linelas)
    Ke = torch.empty(Ne, 8, 8, device=d_all.device, dtype=d_all.dtype)
    for s0 in range(0, Ne, chunk):
        s1 = min(s0 + chunk, Ne)
        Ke[s0:s1] = hess(d_all[s0:s1], BT_e[s0:s1], g_e[s0:s1], c1_e[s0:s1],
                         c2_e[s0:s1], eth_e[s0:s1], wdetJ_e[s0:s1],
                         nu, thickness, linelas)
    return Ke


def _linelas_elem_tangents(d_all, B_T_e, g_e, c1_e, c2_e, eth_e, wdetJ_e,
                           nu, thickness, MP):
    """EXACT closed-form linelas element tangents (unigrid fast path).

    The small-strain energy is QUADRATIC in the dofs: its Hessian is
    independent of the state d and of the eigenstrain, and LINEAR in
    (C1, C2) separately, so per integration point
        K_e(ip) = wdetJ_ip * ( C1_ip * K1(ip) + C2_ip * K2(ip) ).
    The two channels are kept SEPARATE (not collapsed to a single E scale
    via the exact nu-ratio): the stash computes C1 = E/(1-nu^2) and
    C2 = E/(2(1+nu)) as independently ROUNDED fp32 divisions, so their
    ratio deviates from exact at fp32 epsilon and a single-scale
    reconstruction differs from the stashed operator by ~1e-7 -- which the
    fp64 self-check below correctly flags.  On the unigrid all elements
    share one geometry, so the per-IP unit tangents K1, K2 [N_int, 8, 8]
    are built ONCE per mesh by calling
    torch.func.hessian on the SAME `_elem_energy_dofs` used by the general
    path (unit modulus, one-hot wdetJ) -- the two paths are the same
    constitutive code by construction.  Every subsequent assembly is a
    single einsum over the [Ne, N_int] modulus field (which carries the
    per-IP E(rho, T) of the tdep runs).  A one-time self-check compares the
    reconstruction against `_chunked_elem_hessians` on a random element
    subset; any mismatch prints, disables the fast path, and falls back.
    Replaces the vmapped double differentiation -- its cost AND its
    transient-memory spike -- on every refactor in linelas mode.

    PRECISION: assembled entirely in FLOAT64.  With allow_tf32=True (set at
    module import) the einsum contraction and the vmapped reference both run
    their fp32 matmuls in TF32 (~1e-3 relative), so an fp32 fast path and an
    fp32 reference disagree at TF32 level -- the original GPU self-check
    failure.  fp64 is immune to TF32, costs nothing at this size, and hands
    the downstream fp64 direct solve a strictly cleaner operator than the
    fp32/TF32 vmapped path; the self-check below therefore compares against
    an fp64 reference, where agreement is machine-precision and decisive."""
    dev = d_all.device
    dt = torch.float64                # TF32-immune assembly (see docstring)
    Ne, N_int = wdetJ_e.shape
    mesh_key = MP.get('cur_mesh_key', 'default')
    cache = MP.setdefault('_linelas_Kunit', {})
    Kunit = cache.get(mesh_key)
    if Kunit is None:
        BT0 = B_T_e[0].double()                              # shared geometry
        d0 = torch.zeros(8, device=dev, dtype=dt)
        g0 = torch.ones(1, device=dev, dtype=dt)
        one = torch.ones(1, device=dev, dtype=dt)
        zero = torch.zeros(1, device=dev, dtype=dt)
        e0 = torch.zeros(N_int, device=dev, dtype=dt)
        hess1 = torch.func.hessian(_elem_energy_dofs, argnums=0)
        K1s, K2s = [], []
        for ip in range(N_int):
            w1 = torch.zeros(N_int, device=dev, dtype=dt)
            w1[ip] = 1.0
            K1s.append(hess1(d0, BT0, g0, one, zero, e0, w1, nu, 1.0, True))
            K2s.append(hess1(d0, BT0, g0, zero, one, e0, w1, nu, 1.0, True))
        Kunit = (torch.stack(K1s, dim=0), torch.stack(K2s, dim=0))
        cache[mesh_key] = Kunit                              # 2 x [N_int, 8, 8]
        cache[('checked', mesh_key)] = False
    K1u, K2u = Kunit
    w64 = wdetJ_e.double()
    Ke = (torch.einsum('ei,iab->eab', c1_e.double().expand(Ne, N_int) * w64, K1u)
        + torch.einsum('ei,iab->eab', c2_e.double().expand(Ne, N_int) * w64, K2u)
          ) * thickness                                      # float64
    # ---- one-time self-verification against the general vmapped path -------
    if not cache.get(('checked', mesh_key), True):
        cache[('checked', mesh_key)] = True
        n_chk = min(8, Ne)
        idx = torch.randperm(Ne, device=dev)[:n_chk]
        Ke_ref = _chunked_elem_hessians(
            d_all[idx].double(), B_T_e[idx].double(), g_e[idx].double(),
            c1_e[idx].double(), c2_e[idx].double(), eth_e[idx].double(),
            wdetJ_e[idx].double(), nu, thickness, True, chunk=None)
        err = ((Ke[idx] - Ke_ref).abs().max()
               / Ke_ref.abs().max().clamp_min(1e-30)).item()
        if err > 1e-10:
            print(f"[direct adjoint] linelas fast-tangent self-check FAILED "
                  f"(rel err {err:.3e}); disabling the fast path.")
            MP['adjoint_linelas_fast'] = False
            return _chunked_elem_hessians(
                d_all, B_T_e, g_e, c1_e, c2_e, eth_e, wdetJ_e, nu, thickness,
                True, chunk=MP.get('adjoint_hess_chunk', 2048))
        print(f"[direct adjoint] linelas fast tangent verified vs "
              f"vmap-Hessian on {n_chk} elements (rel err {err:.3e}).")
    return Ke


def _audit_adjoint_vs_primal_bcs(model_list, MP, fixed_adj):
    """One-time (per mesh_key) audit: does the adjoint's Dirichlet set equal the
    PRIMAL GP's conditioning set?  The reciprocity identity f_adj.u = lam.f_th
    requires u and lambda to inhabit the SAME constrained space; any dof pinned
    on one side but not the other breaks it and corrupts the sensitivity.
    Prints the exact set-difference with coordinates.  Cheap, runs once."""
    m_u = model_list[0]
    X = m_u.collocation_x
    dev = X.device
    def snap(Xb):
        d = torch.cdist(Xb.to(X.dtype), X)
        return d.argmin(dim=1)                          # nearest node per BC pt
    # primal-pinned dofs: output 0 -> u (even), output 1 -> v (odd)
    prim = torch.cat([2 * snap(m_u.train_inputs_per_output[0]),
                      2 * snap(m_u.train_inputs_per_output[1]) + 1]).unique()
    fa = torch.as_tensor(fixed_adj, device=dev).unique()
    only_adj = fa[~torch.isin(fa, prim)]
    only_prim = prim[~torch.isin(prim, fa)]
    def show(dofs, tag):
        if len(dofs) == 0:
            print(f"    {tag}: none")
            return
        for d in dofs[:8].tolist():
            xy = X[d // 2]
            print(f"    {tag}: node {d//2} ({'u' if d%2==0 else 'v'}) "
                  f"x={xy[0].item():.4g} y={xy[1].item():.4g}")
        if len(dofs) > 8:
            print(f"    {tag}: (+{len(dofs)-8} more)")
    print("[BC audit] slice '%s': adjoint pins %d dofs, primal pins %d dofs; "
          "symmetric-difference = %d"
          % (str(MP.get('cur_mesh_key', 'default')), len(fa), len(prim),
             len(only_adj) + len(only_prim)))
    show(only_adj,  "PINNED in adjoint, FREE in primal (over-constrains lambda)")
    show(only_prim, "PINNED in primal, FREE in adjoint (under-constrains lambda)")
    return len(only_adj) + len(only_prim)


def _find_adjoint_fixed_dofs(collocation_x, model_adj, MP):
    """Global DOFs where the adjoint is Dirichlet-zero, resolved in priority:

    1. MP['adjoint_fixed_dofs']    : LongTensor of global DOFs, or a dict
                                     keyed by mesh_key (for sliced meshes).
    2. MP['adjoint_fixed_dofs_fn'] : callable(collocation_x) -> LongTensor.
                                     RECOMMENDED — pin the FULL constrained
                                     edges geometrically (exact FEM BCs on any
                                     mesh slice; see the EX2D2 default in the
                                     config block).
    3. fallback: SNAP each of the adjoint model's GP conditioning points to
       its nearest mesh node (output i -> displacement component i).  The
       conditioning points are built on the ideal uniform grid and need not
       coincide with the mesh-file nodes; snapping mirrors what the GP's soft
       conditioning does.  Raises only if a point is farther than ~the local
       nodal spacing from any node."""
    override = MP.get('adjoint_fixed_dofs', None)
    if override is not None:
        if isinstance(override, dict):
            override = override[MP.get('cur_mesh_key', 'default')]
        return override.to(collocation_x.device)
    fn = MP.get('adjoint_fixed_dofs_fn', None)
    if fn is not None:
        return fn(collocation_x).to(collocation_x.device)
    # --- snap-to-nearest fallback --------------------------------------------
    # Every conditioning point pins its nearest mesh node, unconditionally —
    # the GP conditions softly at whatever offset these points have from the
    # mesh-file nodes, and pinning the nearest node is the FEM analogue.  The
    # distance check is informational only (flags a grossly wrong point set).
    N = collocation_x.shape[0]
    sub = collocation_x[torch.randperm(N, device=collocation_x.device)[:min(N, 4096)]]
    dd = torch.cdist(sub, collocation_x)
    dd.scatter_(1, dd.argmin(dim=1, keepdim=True), float('inf'))  # drop self
    h = dd.min(dim=1).values.median()                  # local nodal spacing
    tol = MP.get('adjoint_bc_tol', None)
    tol = (2.0 * h) if tol is None else tol
    fixed = []
    for i, Xb in enumerate(model_adj.train_inputs_per_output):
        d = torch.cdist(Xb.to(collocation_x.dtype), collocation_x)
        mind, idx = d.min(dim=1)
        fixed.append(2 * idx.unique() + i)
    return torch.cat(fixed).unique()


def _coalesce_cached(coal_cache, r, c, v, ndof):
    """Sum-coalesce duplicated COO entries with a CACHED index map.

    First call (per cache dict): unique-sort the linearised keys r*ndof + c
    and store (unique rows, unique cols, inverse permutation).  Every later
    call only scatter-adds the new values -- exact (index_add_ sums
    duplicates precisely like COO coalescing), and it removes both the
    per-solve sort and ~7/8 of the nnz handed to the factorization.
    The (r, c) pattern MUST be identical to the cached one (it is: the
    symmetrized assembly pattern is constant per mesh)."""
    if 'inv' not in coal_cache:
        lin = r * ndof + c
        ulin, inv = torch.unique(lin, sorted=True, return_inverse=True)
        coal_cache['ur'] = ulin // ndof
        coal_cache['uc'] = ulin % ndof
        coal_cache['inv'] = inv
        coal_cache['nnz_raw'] = int(r.numel())
        # transpose permutation (the pattern is symmetric by construction:
        # (r, c) were symmetrized upstream), for the exact symmetrization
        # below.  searchsorted is valid because ulin is sorted.
        lin_t = coal_cache['uc'] * ndof + coal_cache['ur']
        tperm = torch.searchsorted(ulin, lin_t)
        assert bool((ulin[tperm] == lin_t).all()), \
            "coalescing cache: COO pattern is not symmetric"
        coal_cache['tperm'] = tperm
    assert r.numel() == coal_cache['nnz_raw'], \
        "COO pattern changed under the coalescing cache"
    # FLOAT64 accumulation + exact transpose-symmetrization.  Summing the
    # duplicates in the value dtype (fp32) breaks the EXACT (i,j) == (j,i)
    # equality the upstream symmetrization established (same addend sets,
    # different index_add_ order -> ~1e-7 relative asymmetry); cuDSS with
    # matrix_type='symmetric' factorizes the LOWER triangle while the
    # residual spmv uses both, and the E_max/E_min contrast amplifies the
    # mismatch into an apparent ~1e-5 residual -- the exact pathology the
    # symmetrization comment in solve_adjoint_direct documents.  (torch-sla's
    # internal coalesce summed in fp64, which is why the pre-cache path did
    # not show it.)  fp64 sums + averaging with the transpose-permuted self
    # make the factorized and checked operators bit-identical-symmetric;
    # residual returns to solver accuracy (~1e-12).
    v_coal = torch.zeros(coal_cache['ur'].numel(), device=v.device,
                         dtype=torch.float64)
    v_coal.index_add_(0, coal_cache['inv'], v.double())
    v_coal = 0.5 * (v_coal + v_coal[coal_cache['tperm']])
    return coal_cache['ur'], coal_cache['uc'], v_coal


def _sparse_direct_solve(vals, rows, cols, ndof, f, MP):
    """Solve the (possibly indefinite) symmetric system in float64 via
    torch_sla.spsolve: cuDSS on CUDA (LDL^T by default), scipy on CPU."""
    v64, f64 = vals.double(), f.double()
    method = MP.get('adjoint_direct_method', 'ldlt')
    # MP['adjoint_solve_device']='cpu' routes the factorization to scipy
    # SuperLU on the HOST: zero VRAM for the solve (only a few-MB COO/RHS
    # round-trip).  At the adjoint refresh cadence the ~0.1-0.3 s host solve
    # amortizes to noise; belt-and-suspenders against driver-crash-by-
    # saturation when the assembly transient already sits near the ceiling.
    if MP.get('adjoint_solve_device', 'gpu') == 'cpu':
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import splu
        A = coo_matrix((v64.cpu().numpy(),
                        (rows.cpu().numpy(), cols.cpu().numpy())),
                       shape=(ndof, ndof)).tocsc()
        return torch.from_numpy(splu(A).solve(f64.cpu().numpy())).to(f.device)
    try:
        import torch_sla
        if vals.is_cuda:
            try:
                lam = torch_sla.spsolve(v64, rows, cols, (ndof, ndof), f64,
                                        backend='cudss', method=method,
                                        matrix_type='symmetric', is_symmetric=True)
            except TypeError:   # older torch-sla signature
                lam = torch_sla.spsolve(v64, rows, cols, (ndof, ndof), f64,
                                        backend='cudss', method=method)
        else:
            lam = torch_sla.spsolve(v64, rows, cols, (ndof, ndof), f64,
                                    backend='scipy', method='lu',
                                    is_symmetric=True)
    except ImportError:
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import splu
        A = coo_matrix((v64.cpu().numpy(),
                        (rows.cpu().numpy(), cols.cpu().numpy())),
                       shape=(ndof, ndof)).tocsc()
        lam = torch.from_numpy(splu(A).solve(f64.cpu().numpy())).to(f.device)
    return lam


# =====================================================================
# SINGLE UNIFORM GRID  (MP['unigrid'] = (Ngx, Ngy))
# ---------------------------------------------------------------------
# An earlier version of this framework kept the PRIMAL on fine mesh slices and ran
# the adjoint on a coarse auxiliary grid, paying for it with kNN-IDW state
# restriction, area-mean/SIMP material homogenization, bilinear load /
# spring transfer and lambda prolongation -- plus the coarsening slack in
# vw_ratio (banking gate widened to 0.50).  This version removes ALL of
# that by putting the primal itself on the adjoint's grid:
#
#   * ONE uniform Ngx x Ngy quad mesh over the domain box carries the
#     primal DEM (u, T), the element design field, the fused sensitivity
#     AND the direct adjoint.  solve_adjoint_direct's plain same-mesh path
#     assembles K_T from the SAME B_T / quadrature / element data the
#     primal energy uses -- no restriction, prolongation, or transfer
#     operators exist anywhere in this file.
#   * FULL 2x2 Gauss quadrature everywhere (MP['wt'] overridden to
#     ones(4) after get_data): the fine reduced 1-pt rule would make the
#     standalone K_T hourglass-singular on a grid this coarse, and the
#     DEM energy only gains accuracy from the richer rule.
#   * Geometry: unigrid cells whose footprint contains no fine-mesh
#     element center (holes / notches) are DELETED and unreferenced nodes
#     dropped + reindexed -- dangling zero-stiffness dofs would make K_T
#     singular (the assembly diagonal-fills FIXED dofs only).
#   * Loads: each fine output-spring / adjoint-load point snaps to its
#     nearest kept node (duplicate snaps aggregated, first-occurrence
#     order preserved so u_out = disp_K_out[0,0] still reads the port
#     node); snap offsets are printed -- expect ~0 when the port sits on
#     a grid-commensurate location, else <= h/2.
#   * vw_ratio and the linelas reciprocity are exact same-mesh identities
#     again (vw_tol back to 0.15); adj_direct_res stays ~1e-12.
#   * Memory: 180x60 -> 10,800 elements / <=11,041 nodes / <=22,082 dofs.
#     The standing GP/DEM graph (kernel cross-covariances, energy graph,
#     cdist) shrinks ~10x vs a fine slice; the K_T factorization is the
#     size the 160x80 coarse solve already was.
#
# The fine Nelx x Nely mesh from get_data is used ONLY to source the
# geometry mask and the load / spring point definitions, then freed.
# =====================================================================

def _build_unigrid_mesh_data(MP, mesh_data_fine, dev, dtype):
    """Build ONE uniform MP['unigrid'] = (Ngx, Ngy) quad mesh (full 2x2
    Gauss) and return it as mesh_data {'GPU0': {'Mesh_01': {...}}} with the
    exact fields run_worker assigns.  Consumes the fine mesh_data only for
    the geometry mask (hole/notch cells) and the load / spring points."""
    Ngx, Ngy = MP['unigrid']
    xmin, xmax = MP['domain']['x']
    ymin, ymax = MP['domain']['y']
    hx = (xmax - xmin) / Ngx
    hy = (ymax - ymin) / Ngy
    xs = torch.linspace(xmin, xmax, Ngx + 1, device=dev, dtype=dtype)
    ys = torch.linspace(ymin, ymax, Ngy + 1, device=dev, dtype=dtype)
    Yg, Xg = torch.meshgrid(ys, xs, indexing='ij')     # node id = j*(Ngx+1)+i
    nodes = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)
    Jg, Ig = torch.meshgrid(torch.arange(Ngy, device=dev),
                            torch.arange(Ngx, device=dev), indexing='ij')
    n0 = (Jg * (Ngx + 1) + Ig).reshape(-1)
    conn = torch.stack([n0, n0 + 1, n0 + Ngx + 2, n0 + Ngx + 1], dim=1).long()

    # --- FULL 2x2 Gauss operators (shared, broadcast over elements) ---------
    # B_T [1,4,2,4] (dN/dX), N [1,4,1,4] (shape fns for the T interpolation),
    # detJ [1,4], integration weights = ones(4) (MP['wt'] set by the caller).
    gp = 1.0 / math.sqrt(3.0)
    pts = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
    BT = torch.zeros(1, 4, 2, 4, device=dev, dtype=dtype)
    Nsh = torch.zeros(1, 4, 1, 4, device=dev, dtype=dtype)
    for k, (xi, eta) in enumerate(pts):
        Nsh[0, k, 0, :] = torch.tensor(
            [(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
             (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)],
            device=dev, dtype=dtype) / 4.0
        dN_dxi = torch.tensor([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                              device=dev, dtype=dtype) / 4.0
        dN_deta = torch.tensor([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)],
                               device=dev, dtype=dtype) / 4.0
        BT[0, k, 0, :] = dN_dxi * (2.0 / hx)
        BT[0, k, 1, :] = dN_deta * (2.0 / hy)
    detJ = torch.full((1, 4), hx * hy / 4.0, device=dev, dtype=dtype)

    # --- geometry mask from the reference fine mesh (holes / notches) -------
    gk = sorted(mesh_data_fine['GPU0'].keys())[0]
    fine = mesh_data_fine['GPU0'][gk]
    ex_f = fine['X_elem']
    ei = ((ex_f[:, 0] - xmin) / hx).floor().clamp(0, Ngx - 1).long()
    ej = ((ex_f[:, 1] - ymin) / hy).floor().clamp(0, Ngy - 1).long()
    ebin = ej * Ngx + ei
    cnt = torch.zeros(Ngx * Ngy, device=dev, dtype=torch.float32)
    cnt.scatter_add_(0, ebin, torch.ones_like(ebin, dtype=torch.float32))
    keep_e = cnt > 0
    if (~keep_e).any():
        print("[unigrid] dropped %d/%d cells containing no fine elements "
              "(holes/notches)." % (int((~keep_e).sum()), Ngx * Ngy))
    conn = conn[keep_e]
    # drop unreferenced nodes + reindex conn (zero-stiffness dangling dofs
    # would make K_T singular; the assembly diagonal-fills FIXED dofs only)
    used = torch.zeros(nodes.shape[0], dtype=torch.bool, device=dev)
    used[conn.reshape(-1)] = True
    new_id = torch.full((nodes.shape[0],), -1, dtype=torch.long, device=dev)
    new_id[used] = torch.arange(int(used.sum()), device=dev)
    nodes = nodes[used]
    conn = new_id[conn]

    Ne = conn.shape[0]
    elem_x = nodes[conn].mean(dim=1)                    # cell centers [Ne, 2]
    elem_vol = torch.full((Ne,), hx * hy, device=dev, dtype=dtype)

    # --- output spring / adjoint load: snap to nearest kept node ------------
    Xf = fine['X_node']

    def _snap(idx_f, mag_f, tag):
        pts_f = Xf[idx_f].to(dtype)
        d = torch.cdist(pts_f, nodes)
        dmin, nid = d.min(dim=1)
        print("[unigrid] %s: %d point(s) snapped, max offset %.3e "
              "(h = %.3g x %.3g)" % (tag, len(idx_f), dmin.max().item(),
                                     hx, hy))
        mag = mag_f.reshape(len(idx_f), -1).to(dtype)
        # aggregate duplicate snaps, FIRST-OCCURRENCE order preserved:
        # u_out = disp_K_out[0,0] must keep reading the port node
        uniq, inv = torch.unique(nid, return_inverse=True)
        first = torch.full((len(uniq),), len(nid), dtype=torch.long,
                           device=dev)
        first.scatter_reduce_(0, inv, torch.arange(len(nid), device=dev),
                              reduce='amin', include_self=True)
        order = torch.argsort(first)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(len(uniq), device=dev)
        mag_a = torch.zeros(len(uniq), mag.shape[1], device=dev, dtype=dtype)
        mag_a.index_add_(0, rank[inv], mag)
        return uniq[order], mag_a

    K_out_index, K_out_magnitude = _snap(fine['K_out_index'],
                                         fine['K_out_magnitude'], 'K_out')
    f_adj_index, f_adj_magnitude = _snap(fine['f_adj_index'],
                                         fine['f_adj_magnitude'], 'f_adj')

    mesh = {'X_node': nodes, 'X_elem': elem_x, 'elem_vol': elem_vol,
            'conn': conn,
            'B': BT,        # placeholder: B is read but unused downstream
            'B_T': BT, 'N': Nsh, 'detJ': detJ,
            'K_out_index': K_out_index, 'K_out_magnitude': K_out_magnitude,
            'f_adj_index': f_adj_index, 'f_adj_magnitude': f_adj_magnitude}
    print("[unigrid] %dx%d grid: %d elements, %d nodes, %d dofs -- primal, "
          "T, rho, sensitivity and adjoint ALL on this one mesh (no "
          "restriction/prolongation)." % (Ngx, Ngy, Ne, nodes.shape[0],
                                          2 * nodes.shape[0]))
    return {'GPU0': {'Mesh_01': mesh}}


def solve_adjoint_direct(model_list, MP):
    """Assemble K_T at the stashed frozen state and solve K_T lambda = f_adj.
    Returns (lambda [N_node,2] in model dtype, relative residual).
    Under MP['unigrid'] the primal already lives on this same uniform mesh,
    so this plain same-mesh path is the ONLY adjoint path."""
    st = MP.get('_adj_stash', None)
    if st is None:
        return None, float('nan')
    m0 = model_list[0]
    conn, B_T, detJ = m0.conn, m0.B_T, m0.detJ
    wt = MP['wt']
    nu, thickness = MP['nu'], MP['domain']['thickness']
    Node_disp = st['Node_disp']
    Ne = conn.shape[0]
    N_node = Node_disp.shape[0]
    ndof = 2 * N_node
    dev = Node_disp.device

    # --- per-element frozen data --------------------------------------------
    d_all = Node_disp[conn].reshape(Ne, 8)
    B_T_e = B_T.expand(Ne, -1, -1, -1) if B_T.shape[0] == 1 else B_T
    detJ_e = detJ.expand(Ne, -1) if detJ.shape[0] == 1 else detJ
    wdetJ = wt * detJ_e                                        # [Ne, N_int]
    gamma_e = st['gamma']                                      # [Ne, 1]
    C1_e, C2_e = st['C1'], st['C2']            # [Ne,1]; [Ne,N_int] with E(T)
    eps_th_e = st['eps_th']                                    # [Ne, N_int]

    # --- batched element tangents  K_e = d2 W_e / d d^2  ---------------------
    linelas = MP.get('constitutive', 'hencky') == 'linelas'
    if linelas and MP.get('adjoint_linelas_fast', True) and B_T.shape[0] == 1:
        # closed-form quadratic-energy tangents (unigrid; self-verified once
        # against the vmapped path, exact fallback on mismatch)
        Ke = _linelas_elem_tangents(d_all, B_T_e, gamma_e, C1_e, C2_e,
                                    eps_th_e, wdetJ, nu, thickness, MP)
    else:
        Ke = _chunked_elem_hessians(d_all, B_T_e, gamma_e, C1_e, C2_e, eps_th_e,
                                    wdetJ, nu, thickness, linelas,
                                    chunk=MP.get('adjoint_hess_chunk', 2048))  # [Ne, 8, 8]

    # --- COO scatter (index pattern cached PER MESH SLICE) --------------------
    mesh_key = MP.get('cur_mesh_key', 'default')
    coo_all = MP.setdefault('_adj_coo_idx', {})
    coo = coo_all.get(mesh_key, None)
    if coo is None:
        dofs = (2 * conn.unsqueeze(-1)
                + torch.tensor([0, 1], device=dev)).reshape(Ne, 8)
        rows = dofs.unsqueeze(-1).expand(Ne, 8, 8).reshape(-1)
        cols = dofs.unsqueeze(-2).expand(Ne, 8, 8).reshape(-1)
        fixed = _find_adjoint_fixed_dofs(m0.collocation_x, model_list[1], MP)
        keep = ~(torch.isin(rows, fixed) | torch.isin(cols, fixed))
        coo_all[mesh_key] = coo = {'rows': rows, 'cols': cols,
                                   'keep': keep, 'fixed': fixed}
        # one-time BC-consistency audit for this slice (adjoint vs primal set)
        if MP.get('adjoint_bc_audit', True):
            try:
                _audit_adjoint_vs_primal_bcs(model_list, MP, fixed)
            except Exception as _e:
                print(f"[BC audit] skipped ({_e})")
    rows, cols, keep, fixed = coo['rows'], coo['cols'], coo['keep'], coo['fixed']
    vals = Ke.reshape(-1)

    # --- BCs (row/col elimination), output spring, adjoint load --------------
    r = torch.cat([rows[keep], fixed])
    c = torch.cat([cols[keep], fixed])
    v = torch.cat([vals[keep], torch.ones(len(fixed), device=dev, dtype=vals.dtype)])
    K_out_index = m0.K_out_index
    K_out_mag = (MP['K_out'] * m0.K_out_magnitude).reshape(len(K_out_index), -1)
    kd = torch.stack([2 * K_out_index, 2 * K_out_index + 1], -1).reshape(-1)
    r = torch.cat([r, kd]); c = torch.cat([c, kd])
    v = torch.cat([v, K_out_mag.reshape(-1).to(vals.dtype)])
    f = torch.zeros(ndof, device=dev, dtype=vals.dtype)
    f_adj_index = m0.f_adj_index
    f_adj_mag = (MP['f_out'] * m0.f_adj_magnitude).reshape(len(f_adj_index), -1)
    f[2 * f_adj_index] += f_adj_mag[:, 0]
    f[2 * f_adj_index + 1] += f_adj_mag[:, 1]
    # --- BC/load overlap guard ------------------------------------------------
    # If a Dirichlet-pinned dof carries nonzero adjoint load, the elimination
    # below silently DELETES that load: lambda solves a different problem while
    # adj_direct_res and vw_ratio stay perfect (both check the assembled system,
    # not the BC set).  Symptom downstream: dead sensitivity near the output
    # port and an artificially simplified topology.
    _killed = f[fixed].abs().sum().item()
    if _killed > 0:
        print("[direct adjoint] *** WARNING: fixed dofs overlap the adjoint "
              "load (|f| removed = %.3e on %d dofs, slice '%s'). lambda will "
              "be WRONG; check adjoint BC conditioning points / snap fallback."
              % (_killed, int((f[fixed] != 0).sum()),
                 str(MP.get('cur_mesh_key', 'default'))))
    f[fixed] = 0.0

    # --- exact symmetrization:  K <- (K + K^T)/2 ------------------------------
    # The vmapped element Hessians are symmetric only to fp32 roundoff; cuDSS
    # with matrix_type='symmetric' factorizes the LOWER triangle, so an
    # unsymmetrized COO makes the factorized operator differ from the assembled
    # one by ~1e-7 relative — amplified by the E_max/E_min stiffness contrast
    # into an apparent ~1e-5 residual.  Averaging with the transpose makes the
    # solved and checked operators identical (residual back to solver accuracy).
    r_s = torch.cat([r, c]); c_s = torch.cat([c, r])
    v_s = 0.5 * torch.cat([v, v])
    r, c, v = r_s, c_s, v_s

    # --- cached coalescing: constant (r, c) pattern per mesh -> unique-sort
    # once, scatter-add values per solve (exact; ~8x fewer nnz to cuDSS,
    # and the downstream residual spmv runs on the coalesced arrays too) ---
    r, c, v = _coalesce_cached(coo.setdefault('coal', {}), r, c, v, ndof)

    lam = _sparse_direct_solve(v, r, c, ndof, f, MP)

    # --- residual check (one spmv on the assembled system) -------------------
    Kl = torch.zeros(ndof, device=dev, dtype=torch.float64)
    Kl.index_add_(0, r, v.double() * lam[c])
    res = ((Kl - f.double()).norm() / f.double().norm().clamp_min(1e-30)).item()

    # --- BC-set validation: linelas reciprocity ------------------------------
    # For LINEAR thermo-elasticity, with the CORRECT constrained space,
    #   f_adj . u  =  lambda^T K u  =  lambda^T f_th ,
    # where f_th = -dPi_int/du|_{u=0} is the thermal-eigenstrain load.  Unlike
    # adj_direct_res / vw_ratio, this identity FAILS if lambda was solved on an
    # over- or under-pinned subspace, because u comes from the GP field
    # honouring the TRUE primal BCs.  Holds to primal-DEM-convergence accuracy;
    # a persistent |ratio-1| >> DEM residual level indicates a wrong fixed-dof
    # set.  Skipped in Hencky mode (identity is linear-only).
    if MP.get('constitutive', 'hencky') == 'linelas' \
            and MP.get('adjoint_bc_check', True):
        gradf = torch.func.vmap(
            torch.func.grad(_elem_energy_dofs, argnums=0),
            in_dims=(0, 0, 0, 0, 0, 0, 0, None, None, None))
        fe = gradf(torch.zeros_like(d_all), B_T_e, gamma_e, C1_e, C2_e,
                   eps_th_e, wdetJ, nu, thickness, True)       # = -f_th per elem
        dofs_e = (2 * conn.unsqueeze(-1)
                  + torch.tensor([0, 1], device=dev)).reshape(-1)
        f_th = torch.zeros(ndof, device=dev, dtype=torch.float64)
        f_th.index_add_(0, dofs_e, -fe.reshape(-1).double())
        lam_fth = (lam * f_th).sum().item()
        f_u = (f.double() * Node_disp.reshape(-1).double()).sum().item()
        recip = lam_fth / f_u if abs(f_u) > 1e-30 else float('nan')
        MP['_adj_recip_ratio'] = recip
        if math.isfinite(recip) and abs(recip - 1.0) > 0.05:
            # localize: ratio - 1 = lambda^T (f_th - K u) / (f_adj . u), so the
            # deficit lives where lambda_i * r_i is large.  Pin-violation shows
            # at pinned dofs; UNDER-pinning shows at primal-conditioned dofs we
            # did NOT pin; DEM non-equilibrium shows near the output point load.
            u_flat = Node_disp.reshape(-1).double()
            u_pin_max = u_flat[fixed].abs().max().item() if len(fixed) else 0.0
            Ku = torch.zeros(ndof, device=dev, dtype=torch.float64)
            Ku.index_add_(0, r, v.double() * u_flat[c])
            r_res = Ku - f_th
            contrib = (lam * r_res).abs()
            contrib[fixed] = 0.0
            top = torch.topk(contrib, k=min(5, ndof)).indices
            loc = "; ".join(
                "(x=%.4g, y=%.4g, %s: lam*r=%.3e)"
                % (m0.collocation_x[d // 2, 0].item(),
                   m0.collocation_x[d // 2, 1].item(),
                   'u' if d % 2 == 0 else 'v',
                   (lam[d] * r_res[d]).item())
                for d in top.tolist())
            # if MP.epoch % 100 ==0
            # print("[direct adjoint] note: reciprocity = %.4f on slice '%s'; "
            #     "max |u| at pinned dofs = %.3e; lambda-weighted primal "
            #     "residual accounts for the deficit (lam.r/f.u = %.4f). "
            #     "Top residual sites: %s"
            #     % (recip, str(MP.get('cur_mesh_key', 'default')), u_pin_max,
            #         ((lam * r_res).sum() / f_u).item() if abs(f_u) > 1e-30
            #         else float('nan'), loc))

    return lam.reshape(N_node, 2).to(Node_disp.dtype).detach(), res


# =====================================================================
# calculate_TO_loss  (with sparse Helmholtz filter)
# =====================================================================

def calculate_TO_loss(model_list):
    MP = model_list[0].MP
    direct_adj = MP.get('adjoint_mode', 'dem') == 'direct'
    # constitutive switch: 'hencky' (finite-strain, three-term Wang) or
    # 'linelas' (small strain — Wang form collapses to psi_L, gamma inert,
    # tangent has no geometric term, direct adjoint solves the standard
    # linear thermo-mech system).
    linelas = MP.get('constitutive', 'hencky') == 'linelas'
    num_phase = MP['num_phase']
    alpha_tensor = MP['alpha']
    kappa_tensor = MP['kappa']
    D = MP['D']
    P = MP['P']
    E_tensor = MP['E']
    wt = MP['wt']
    f_out = MP['f_out']
    a_u = 1/f_out
    a_T = MP['a_T']
    K_out = MP['K_out']
    nu = MP['nu']
    p = MP['p']
    s_tensor = model_list[0].MP['s']
    T_inf = model_list[0].MP['T_inf']
    thickness = MP['domain']['thickness']

    collocation_x = model_list[0].collocation_x.clone()
    elem_x = model_list[0].elem_x.clone()
    elem_vol = model_list[0].elem_vol
    f_adj_index = model_list[0].f_adj_index
    f_adj_magnitude = f_out * model_list[0].f_adj_magnitude
    K_out_index = model_list[0].K_out_index
    K_out_magnitude = K_out * model_list[0].K_out_magnitude
    conn = model_list[0].conn
    B = model_list[0].B
    B_T = model_list[0].B_T
    N = model_list[0].N
    detJ = model_list[0].detJ
    N_elem_w = elem_x.shape[0]

    for model in model_list:
        model.train()

    # ======================
    # 1. Mean predictions
    # ======================
    m_col_disp = model_list[0].mean_module_NN_All(collocation_x)
    m_col_disp_adj = None if direct_adj else model_list[1].mean_module_NN_All(collocation_x)
    m_col_T = model_list[2].mean_module_NN_All(collocation_x)
    m_col_T_adj = model_list[3].mean_module_NN_All(collocation_x)
    m_elem_phase = model_list[4].mean_module_NN_All(elem_x)

    # ====================================================
    # Independent kernels
    # ====================================================
    g_u = _cached_cross_cov(model_list[0], 0, collocation_x, MP, 'disp')
    g_v = _cached_cross_cov(model_list[0], 1, collocation_x, MP, 'disp')

    g_vd1 = None if direct_adj else _cached_cross_cov(model_list[1], 0, collocation_x, MP, 'disp_adj')
    g_vd2 = None if direct_adj else _cached_cross_cov(model_list[1], 1, collocation_x, MP, 'disp_adj')

    g_t = _cached_cross_cov(model_list[2], 0, collocation_x, MP, 'T')

    g_vt = _cached_cross_cov(model_list[3], 0, collocation_x, MP, 'T_adj')

    if model_list[0].MP['design_flag']:
        g_phases = [_cached_cross_cov(model_list[4], i, elem_x, MP, 'phase')
                    for i in range(model_list[4].num_output)]

    # ====================================================
    # Compute Cholesky decompositions and offsets
    # ====================================================
    if model_list[0].chol_decomp is None:
        with cholesky_jitter(1e-5):
            # --- Displacement kernels ---
            model_list[0].chol_decomp = []
            for i in range(model_list[0].num_output):
                K_i = model_list[0].independent_kernels[i](model_list[0].train_inputs_per_output[i]).evaluate().detach()
                L_i = torch.linalg.cholesky(K_i + 1e-5 * torch.eye(K_i.shape[0], device=K_i.device))
                model_list[0].chol_decomp.append(L_i)

            # --- Displacement adjoint kernels (skipped in direct mode) ---
            model_list[1].chol_decomp = []
            if not direct_adj:
                for i in range(model_list[1].num_output):
                    K_i = model_list[1].independent_kernels[i](model_list[1].train_inputs_per_output[i]).evaluate().detach()
                    L_i = torch.linalg.cholesky(K_i + 1e-5 * torch.eye(K_i.shape[0], device=K_i.device))
                    model_list[1].chol_decomp.append(L_i)

            # --- T kernels ---
            model_list[2].chol_decomp = []
            for i in range(model_list[2].num_output):
                K_i = model_list[2].independent_kernels[i](model_list[2].train_inputs_per_output[i]).evaluate().detach()
                L_i = torch.linalg.cholesky(K_i + 1e-5 * torch.eye(K_i.shape[0], device=K_i.device))
                model_list[2].chol_decomp.append(L_i)

            # --- T adjoint kernels ---
            model_list[3].chol_decomp = []
            for i in range(model_list[3].num_output):
                K_i = model_list[3].independent_kernels[i](model_list[3].train_inputs_per_output[i]).evaluate().detach()
                L_i = torch.linalg.cholesky(K_i + 1e-5 * torch.eye(K_i.shape[0], device=K_i.device))
                model_list[3].chol_decomp.append(L_i)

            # --- Phase kernels (only if needed) ---
            if model_list[0].MP['design_flag']:
                model_list[4].chol_decomp = []
                for i in range(model_list[4].num_output):
                    K_i = model_list[4].independent_kernels[i](model_list[4].train_inputs_per_output[i]).evaluate().detach()
                    L_i = torch.linalg.cholesky(K_i + 1e-5 * torch.eye(K_i.shape[0], device=K_i.device))
                    model_list[4].chol_decomp.append(L_i)

    # ====================================================
    # Compute K⁻¹·(y − μ)
    # ====================================================

    # --- Displacement ---
    K_inv_offset_u = torch.cholesky_solve(
        model_list[0].train_target_per_output[0].unsqueeze(-1)
        - model_list[0].mean_module_NN_All(model_list[0].train_inputs_per_output[0])[:, 0].unsqueeze(-1),
        model_list[0].chol_decomp[0]
    )

    K_inv_offset_v = torch.cholesky_solve(
        model_list[0].train_target_per_output[1].unsqueeze(-1)
        - model_list[0].mean_module_NN_All(model_list[0].train_inputs_per_output[1])[:, 1].unsqueeze(-1),
        model_list[0].chol_decomp[1]
    )

    # --- Displacement Adjoint (skipped in direct mode) ---
    if not direct_adj:
        K_inv_offset_vd1 = torch.cholesky_solve(
            model_list[1].train_target_per_output[0].unsqueeze(-1)
            - model_list[1].mean_module_NN_All(model_list[1].train_inputs_per_output[0])[:, 0].unsqueeze(-1),
            model_list[1].chol_decomp[0]
        )

        K_inv_offset_vd2 = torch.cholesky_solve(
            model_list[1].train_target_per_output[1].unsqueeze(-1)
            - model_list[1].mean_module_NN_All(model_list[1].train_inputs_per_output[1])[:, 1].unsqueeze(-1),
            model_list[1].chol_decomp[1]
        )

    # --- Temperature ---
    K_inv_offset_T = torch.cholesky_solve(
        model_list[2].train_target_per_output[0].unsqueeze(-1)
        - model_list[2].mean_module_NN_All(model_list[2].train_inputs_per_output[0])[:, 0].unsqueeze(-1),
        model_list[2].chol_decomp[0]
    )
    # --- Temperature Adjoint ---
    K_inv_offset_T_adj = torch.cholesky_solve(
        model_list[3].train_target_per_output[0].unsqueeze(-1)
        - model_list[3].mean_module_NN_All(model_list[3].train_inputs_per_output[0])[:, 0].unsqueeze(-1),
        model_list[3].chol_decomp[0]
    )

    # --- Phase (if design_flag = True) ---
    if model_list[0].MP['design_flag']:
        K_inv_offsets_phase = []
        for i in range(model_list[4].num_output):
            y_i = model_list[4].train_target_per_output[i]
            X_i = model_list[4].train_inputs_per_output[i]
            mu_i = model_list[4].mean_module_NN_All(X_i)[:, i].unsqueeze(-1)
            offset_i = y_i.unsqueeze(-1) - mu_i
            K_inv_offsets_phase.append(
                torch.cholesky_solve(offset_i, model_list[4].chol_decomp[i])
            )

    u = (m_col_disp[:,0].unsqueeze(-1) + g_u.t() @ K_inv_offset_u)
    v = (m_col_disp[:,1].unsqueeze(-1) + g_v.t() @ K_inv_offset_v)
    Node_T = (m_col_T[:,0].unsqueeze(-1) + g_t.t() @ K_inv_offset_T)
    Node_T_adj = (m_col_T_adj[:,0].unsqueeze(-1) + g_vt.t() @ K_inv_offset_T_adj)

    Node_disp = torch.cat([u, v], dim=1)        # (N_node_w, 2)
    if direct_adj:
        # lambda is a SOLVED nodal field (exact full-tangent adjoint via sparse
        # direct factorization), stored per mesh slice; zeros until the first
        # solve for this slice (epoch 0 only in single-mesh runs).
        lam_all = MP.get('lambda_nodal', None)
        lam = lam_all.get(MP.get('cur_mesh_key', 'default')) \
            if isinstance(lam_all, dict) else lam_all
        if lam is None or lam.shape != Node_disp.shape:
            lam = torch.zeros_like(Node_disp)
        Node_disp_adj = lam
    else:
        vd1 = (m_col_disp_adj[:,0].unsqueeze(-1) + g_vd1.t() @ K_inv_offset_vd1)
        vd2 = (m_col_disp_adj[:,1].unsqueeze(-1) + g_vd2.t() @ K_inv_offset_vd2)
        Node_disp_adj = torch.cat([vd1, vd2], dim=1) # (N_node_w, 2)

    if model_list[0].MP['design_flag']:
        phases = [m_elem_phase[:, i].unsqueeze(-1) + g_phases[i].t() @ K_inv_offsets_phase[i] for i in range(len(K_inv_offsets_phase))]
        phase_weights = torch.cat(phases, dim=1)
    else:
        phase_weights = m_elem_phase

    # =====================================================================
    # >>> HELMHOLTZ FILTER: smooth the density field for length-scale control
    # =====================================================================
    phase_weights_unfilt = phase_weights       # keep raw field for grey/cost diagnostics
    helmholtz_filter = MP.get('helmholtz_filter')
    if helmholtz_filter is not None:
        mesh_key = MP.get('cur_mesh_key', 'default')
        phase_weights = helmholtz_filter(
            phase_weights, conn, collocation_x, mesh_key=mesh_key,
        )
    # POSITIVITY GUARD: everything downstream raises these weights to the ODD
    # SIMP power p=3, so any negative weight (filter over/undershoot on other
    # meshes, non-softmax NN head, CG tolerance) flips the SIGN of that
    # phase's kappa/E/s contribution; one int point with kappa_hat < 0 makes
    # the thermal DEM energy unbounded below (monotonic T runaway, no NaN).
    # clamp is a no-op when weights are already in [0,1].
    phase_weights = phase_weights.clamp(0.0, 1.0)
    # =====================================================================

    # =====================================================================
    # >>> BETA-CONTINUATION PROJECTION (three-field: rho -> filtered -> projected)
    # Smooth Heaviside on the SOLID fraction, same tanh form as the gamma
    # projection; beta is scheduled by the training loop via MP['beta_proj']
    # (beta0 -> beta_max, doubling every beta_double_every epochs after
    # beta_start_ep).  Kills grey hedging by making intermediate density
    # energetically pointless.  Two-phase only; the PROJECTED field is the
    # physical density and drives everything downstream: SIMP stiffness,
    # alpha, kappa, source, gamma, and the MASS CONSTRAINT (projection is
    # not mass-preserving, so constraining the raw field would let the
    # physical mass drift off target as beta grows).
    # =====================================================================
    beta_p = MP.get('beta_proj', None)
    if (beta_p is not None) and (num_phase == 2):
        eta_p = MP.get('eta_proj', 0.5)
        ws = phase_weights[:, 1:2].clamp(0.0, 1.0)
        den_p = math.tanh(beta_p * eta_p) + math.tanh(beta_p * (1.0 - eta_p))
        ws_bar = (math.tanh(beta_p * eta_p) + torch.tanh(beta_p * (ws - eta_p))) / den_p
        phase_weights = torch.cat([1.0 - ws_bar, ws_bar], dim=1)
    # =====================================================================

    # =====================================================================
    # >>> INTERFACE INCOMPATIBILITY: penalise adjacency of incompatible
    # material pairs (default: Ti-Fe, phases 1-3 -- TiFe/TiFe2 brittle
    # intermetallics).
    # ---------------------------------------------------------------------
    # Smooth Heaviside indicator  H = sigmoid(beta_if*(w_i - eta_if)) turns
    # "material present" toward 1 and absent toward 0.  NB the grey floor is
    # NOT small: H(w=0.25) ~ 0.31-0.45 for beta_if in [4,16] at eta=0.30, so
    # on a grey field the penalty is a GLOBAL Ti+Fe suppressor, not an
    # interface detector -- which is why the ramp below starts only after the
    # mass ramp, once the field has binarised (genuine interfaces w ~ 0.8 ->
    # H ~ 1 then dominate).  Each phase's indicator is
    # neighbour-averaged through the Helmholtz filter's existing sparse
    # elem->node->elem projections (E2N then N2E: the incident-element
    # stencil of the 4 nodes), and the SYMMETRIC product
    #     sum_e [ H_a * <H_b>_neigh + H_b * <H_a>_neigh ] * V_e / V_total
    # is volume-normalised so its magnitude is mesh-size-independent.
    # Computed on the FILTERED weights (the physical field);
    # beta_if is ramped 4 -> 16 by the training loop (continuation).
    # Requires the filter (for N2E/E2N); zero when the filter is off.
    # =====================================================================
    if_pairs = MP.get('interface_pairs', [(1, 3)])
    if helmholtz_filter is not None:
        mesh_key_if = MP.get('cur_mesh_key', 'default')
        _, _, _, N2E_if, E2N_if = helmholtz_filter._get_or_build(
            conn, collocation_x, mesh_key_if,
        )
        beta_if = MP.get('beta_interface', 8.0)   # sharpness (ramped by loop)
        eta_if  = MP.get('eta_interface', 0.30)   # below -> "absent"
        V_total = elem_vol.sum()
        interface_penalty = torch.zeros((), device=elem_x.device,
                                        dtype=elem_x.dtype)
        for (ia, ib) in if_pairs:
            H_a = torch.sigmoid(beta_if * (phase_weights[:, ia] - eta_if))
            H_b = torch.sigmoid(beta_if * (phase_weights[:, ib] - eta_if))
            H_b_neigh = _spmm(N2E_if, _spmm(E2N_if, H_b.unsqueeze(-1))).squeeze(-1)
            H_a_neigh = _spmm(N2E_if, _spmm(E2N_if, H_a.unsqueeze(-1))).squeeze(-1)
            interface_penalty = interface_penalty + (
                (H_a * H_b_neigh * elem_vol).sum()
                + (H_b * H_a_neigh * elem_vol).sum()
            ) / V_total
    else:
        interface_penalty = torch.tensor(0.0, device=elem_x.device,
                                         dtype=elem_x.dtype)
    # =====================================================================

    # Mass: on the PROJECTED (physical) density; cost/grey kept on the raw
    # field as before (wp inert; grey diagnoses the raw design field)
    phase_density_phys = (phase_weights * D).sum(dim=1)   # filtered+projected
    mass = (phase_density_phys * elem_vol).sum()
    M0 = (elem_vol).sum()

    phase_cost_unfilt = (phase_weights_unfilt * P).sum(dim=1)
    cost = (phase_cost_unfilt * elem_vol).sum()
    cost0 = (P.max() * torch.ones_like(phase_cost_unfilt) * elem_vol).sum()

    # Grey element count also on unfiltered (diagnostic of raw design field)
    grey_counts = []
    weights_local = phase_weights_unfilt.detach()  # [N_elem_w, num_phase]

    for i in range(num_phase):
        mask = (weights_local[:, i] > MP['rho_min']) & (weights_local[:, i] < MP['rho_max'])
        grey_counts.append(mask.sum())

    grey_counts = torch.stack(grey_counts, dim=0)  # [num_phase]
    count_total = torch.tensor(weights_local.shape[0], device=weights_local.device)

    # calculate material constants (with gradient for density) for the adjoint terms
    # --- These use FILTERED phase_weights for physics ---
    E_hat = (E_tensor * phase_weights ** p).sum(dim = 1) # [n_elem]
    C1_hat = (E_hat / (1 - nu ** 2))[:, None] # [N_elem_w, 1]
    C2_hat = (E_hat / (2 * (1 + nu)))[:, None]
    kappa_hat = (kappa_tensor * phase_weights ** p).sum(dim = 1).unsqueeze(-1) # [n_elem,1]
    alpha_hat = (alpha_tensor * phase_weights).sum(dim = 1).unsqueeze(-1) # [n_elem,1]
    s_hat = (s_tensor * phase_weights ** p).sum(dim = 1).unsqueeze(-1) # [n_elem,1]

    # calculate material constants (no gradient for density) for DEM terms
    kappa_no_grad = kappa_hat.detach()
    alpha_no_grad = alpha_hat.detach()
    s_no_grad = s_hat.detach()
    C1_no_grad = C1_hat.detach()
    C2_no_grad = C2_hat.detach()

    # ========================
    # 1. estimate primal equations for temperature
    # ========================
    T_elem = Node_T[conn].reshape(N_elem_w, -1)
    T_elem = T_elem.unsqueeze(1).unsqueeze(-1)
    T_elem = T_elem.expand(-1, N.shape[1], -1, -1)

    T_int_pt = torch.matmul(N, T_elem).squeeze(-1)

    # ---------------------------------------------------------------------
    # >>> TEMPERATURE-DEPENDENT PROPERTIES  kappa(rho,T), alpha(rho,T)
    # Picard/staggered treatment, consistent with the rest of the loop: the
    # property T-arguments are the DETACHED current temperature at the
    # integration points (T is the OTHER state).  The primal T stationarity
    # then reads  int kappa(T*) grad(T).grad(w) = int s w, which at the Adam
    # fixed point T* = T is EXACTLY the nonlinear weak form
    #     div( kappa(T) grad T ) + s = 0
    # (Picard linearisation — detaching T in kappa avoids the spurious
    # 1/2 kappa'(T)|grad T|^2 term a live-T energy would create).
    # rho stays LIVE through the phase weights, so kappa_hat and the
    # eigenstrain below carry the exact d kappa/d rho and d eps_th/d rho
    # pathways AT FROZEN T for the `heat` / mech_sens sensitivity terms
    # (dT/drho effects are what the adjoints are for, as before).
    # Overwrites the constant-property hats computed above; shapes go
    # [N_elem,1] -> [N_elem,N_int] (broadcasts identically downstream).
    # ---------------------------------------------------------------------
    tdep = MP.get('tdep_props', None)
    alpha_inst_ng = None      # instantaneous CTE at T*  (T-adjoint coupling)
    dkappa_dT_ng = None       # d kappa/dT at T*         (T-adjoint tangent term)
    eratio_ng = None          # (dE/dT)/E at T*          (T-adjoint E-pathway)
    if tdep is not None:
        _T_ref = tdep.get('T_ref', T_inf)
        T_ip_det = T_int_pt.squeeze(-1).detach()                  # [N_elem_w, N_int]
        # --- fit-validity clamp (CRITICAL) ---------------------------------
        # The polynomial fits are only valid on tdep['T_valid'] (here
        # [293, 1100] K -- Ti alpha-phase to the 1155 K transus; the steel
        # Curie point 1043 K is smoothed through inside it).  Outside the
        # anchor range the unguarded fits eventually turn NEGATIVE (the
        # quadratic E factors and the cubic kappa factors of ANY phase); one
        # bad phase at one int point is enough.  The T field is
        # an NN: transients and unconstrained void regions CAN leave the
        # window, and one int point with kappa < 0 makes the thermal DEM
        # energy unbounded below -> T runaway -> garbage design gradients
        # with no NaN anywhere.  Constant extrapolation: evaluate every
        # polynomial at the CLAMPED dT and kill the derivative factors
        # (dkappa/dT, alpha_inst = d eps_th/dT, dE/dT) outside the window,
        # so primal, adjoint sources and sensitivities all describe the SAME
        # clamped model (FD-consistent).
        _Tlo, _Thi = tdep.get('T_valid', (293.0, 1100.0))
        dTp_raw = T_ip_det - _T_ref
        dTp  = dTp_raw.clamp(_Tlo - _T_ref, _Thi - _T_ref)
        _inw = ((T_ip_det > _Tlo) & (T_ip_det < _Thi)).to(T_ip_det.dtype)
        # diagnostics for run_worker logging (detached floats).  clamp_frac
        # counts ALL int points and is dominated by void-region T drift
        # (kappa_void ~ 1e-8 leaves the NN field energetically unconstrained
        # there; harmless under the clamp).  clamp_frac_solid masks to
        # elements with solid fraction >= 0.5 -- the number that matters:
        # clamped SOLID points carry zero d(prop)/dT, silently removing the
        # tdep content of the T-adjoint and the sensitivities there.
        _solid_e = (phase_weights[:, 1:].sum(dim=1, keepdim=True)
                    >= 0.5).to(T_ip_det.dtype).detach()          # [Ne, 1]
        MP['_tdep_diag'] = {
            'T_min': T_ip_det.min().item(), 'T_max': T_ip_det.max().item(),
            'clamp_frac': (1.0 - _inw).mean().item(),
            'clamp_frac_solid': (((1.0 - _inw) * _solid_e).sum()
                                 / (_solid_e.sum() * T_ip_det.shape[1])
                                   .clamp_min(1.0)).item()}
        dTp0 = float(min(max(T_inf, _Tlo), _Thi) - _T_ref)
        w_p_t = phase_weights ** p                                # [N_elem_w, num_phase]
        # ---- prop_eval='fixed_TD': the FROZEN-AT-TD counterpart model ------
        # kappa and E become CONSTANTS evaluated at T = TD; alpha becomes the
        # per-phase SECANT  abar_i = [F_i(dTD) - F_i(dT0)] / (TD - T_inf)
        # (the unique constant reproducing the tdep eigenstrain EXACTLY at
        # T = TD; alpha_inst(TD) would overshoot), and the eigenstrain stays
        # field-dependent:  eps_th(x) = abar(rho) * (T(x) - T_inf).
        # ALL d(prop)/dT pathways vanish: dkappa_dT_ng = eratio_ng = None,
        # so the T-adjoint loses its dk/dT tangent term and its E-pathway
        # source and collapses to the classic constant-property structure
        # (P1 coupling only, with alpha_inst_ng = abar); the displacement
        # adjoint changes only in coefficient values.  The clamp is a single
        # in-window scalar evaluation (diagnostics report zero).
        _fixed_TD = MP.get('prop_eval', 'tdep') == 'fixed_TD'
        if _fixed_TD:
            _dT_star = float(min(max(MP['TD'], _Tlo), _Thi) - _T_ref)
            _den_sec = max(_dT_star - dTp0, 1e-30)
            _kap, _a_int, _a_inst = 0.0, 0.0, 0.0
            _dkap = None
            _Tdev_ip = (T_ip_det - T_inf)                     # [Ne, N_int]
            for i in range(num_phase):
                ck = tdep['kappa'][i]
                ca = tdep['alpha'][i]
                _kap = _kap + kappa_tensor[i] * _poly_eval(ck, _dT_star) \
                    * w_p_t[:, i:i+1]
                _abar_i = alpha_tensor[i] \
                    * (_poly_antideriv(ca, _dT_star)
                       - _poly_antideriv(ca, dTp0)) / _den_sec
                _a_int  = _a_int  + _abar_i * phase_weights[:, i:i+1] * _Tdev_ip
                _a_inst = _a_inst + _abar_i * phase_weights[:, i:i+1] \
                    * torch.ones_like(_Tdev_ip)
        else:
            _kap, _dkap, _a_int, _a_inst = 0.0, 0.0, 0.0, 0.0
            for i in range(num_phase):
                ck = tdep['kappa'][i]
                ca = tdep['alpha'][i]
                _kap   = _kap   + kappa_tensor[i] * _poly_eval(ck, dTp)  * w_p_t[:, i:i+1]
                _dkap  = _dkap  + kappa_tensor[i] * _poly_deriv(ck, dTp) * w_p_t[:, i:i+1]
                # eigenstrain: exact integral of the instantaneous CTE over
                # [T_inf, T*]; the T_ref constant of integration cancels.
                _a_int = _a_int + alpha_tensor[i] \
                    * (_poly_antideriv(ca, dTp) - _poly_antideriv(ca, dTp0)) \
                    * phase_weights[:, i:i+1]
                _a_inst = _a_inst + alpha_tensor[i] * _poly_eval(ca, dTp) \
                    * phase_weights[:, i:i+1]
        kappa_hat = _kap                                          # rho-LIVE
        kappa_no_grad = kappa_hat.detach()
        # derivative factors are ZERO outside the validity window (the clamped
        # model is constant there) -- keeps the T-adjoint consistent with the
        # clamped primal.  fixed_TD: identically None (constant model).
        dkappa_dT_ng = (_dkap * _inw).detach() if torch.is_tensor(_dkap) else None
        eps_th_int_hat = _a_int                                   # rho-LIVE eigenstrain
        alpha_inst_ng = (_a_inst.detach() if _fixed_TD
                         else (_a_inst * _inw).detach())
        MP['_tdep_diag']['kappa_min'] = float(kappa_no_grad.min())
        # E_min recorded after the E(rho,T) override below (see _Eh)
        # --- E(rho,T): overrides E_hat / C1_hat / C2_hat computed above -------
        # Same SIMP interpolation, per-phase multiplicative poly in dT; shapes
        # go [Ne,1] -> [Ne,N_int] (broadcast fine everywhere downstream, incl.
        # the vmapped K_T element Hessians, which slice c1_e per element).
        # The design pathway dE/drho at frozen T is carried automatically by
        # the rho-live weights inside C1_hat/C2_hat (consumed by mech_sens);
        # dE/dT feeds the T-adjoint through eratio_ng below.
        _ce_all = tdep.get('E', None)
        if _ce_all is not None:
            _Eh, _dEh = 0.0, 0.0
            for i in range(num_phase):
                ce = _ce_all[i]
                if _fixed_TD:
                    _Eh = _Eh + E_tensor[i] * _poly_eval(ce, _dT_star) \
                        * w_p_t[:, i:i+1]
                    _dEh = None
                else:
                    _Eh  = _Eh  + E_tensor[i] * _poly_eval(ce, dTp)  * w_p_t[:, i:i+1]
                    _dEh = _dEh + E_tensor[i] * _poly_deriv(ce, dTp) * w_p_t[:, i:i+1]
            E_hat = _Eh                                           # rho-LIVE, [Ne, N_int]
            C1_hat = E_hat / (1 - nu ** 2)
            C2_hat = E_hat / (2 * (1 + nu))
            C1_no_grad = C1_hat.detach()
            C2_no_grad = C2_hat.detach()
            MP['_tdep_diag']['E_min'] = float(E_hat.detach().min())
            if torch.is_tensor(_dEh):
                # ratio is exact because EVERY term of the interpolated energy
                # (all three Wang terms, both C1 and C2) is proportional to the
                # SINGLE scalar field E_hat(x):  dW/dT|_E = (dE/dT / E) * W.
                # _inw: zero outside the clamp window (clamped-model adjoint).
                eratio_ng = (_dEh * _inw / _Eh).detach()
    # ---------------------------------------------------------------------

    sT = s_no_grad.unsqueeze(-1) * T_int_pt
    sT_int = (sT.squeeze(-1) * detJ * wt.unsqueeze(0)) * thickness
    source_energy = sT_int.sum()

    dT = torch.matmul(B_T, T_elem).squeeze(-1)
    dT_square = (dT ** 2).sum(dim=-1)
    kdT2 =  kappa_no_grad * dT_square
    kdT2_int = (kdT2 * detJ * wt.unsqueeze(0)) * thickness
    heat_energy = 0.5 * kdT2_int.sum()

    # ========================
    # 2. estimate primal equations for displacement
    #    FINITE STRAIN: total-Lagrangian St. Venant–Kirchhoff (SVK)
    # ========================
    # Kinematics:  H = Grad u,  F = I + H,  E = ½(H + Hᵀ + HᵀH)  (Green–Lagrange)
    # Constitutive: S = C(ρ) : (E − E_th),  E_th = α(T−T_inf)·I  (Duhamel,
    #               additive eigenstrain — valid for small THERMAL strains
    #               even at large rotations).  Same plane-stress C as before.
    # Energy interpolation (Wang–Lazarov–Sigmund–Jensen, IJNME 2014):
    #     ψ_e = ψ_NL(γ_e·u) + (1 − γ_e²)·ψ_L(u),   γ_e = H_β(ρ_solid)
    # → solid elements (γ≈1) see the full nonlinear energy, void elements
    #   (γ≈0) see linear energy with E_min and cannot distort/invert.
    # γ_e is DETACHED: it is a numerical stabilisation device, not a design
    # sensitivity pathway.
    # Reduces exactly to the previous small-strain code as H → 0, γ → 1.
    external_work = 0

    # output-port spring: acts on displacement components — unchanged
    disp_K_out = Node_disp[K_out_index]
    external_work_K_out = 0.5 * torch.sum((disp_K_out**2) * K_out_magnitude)
    spring_energy = external_work_K_out

    # --- energy-interpolation factor γ_e (per element, detached) ---
    # multi-material: phase index 0 = void, indices 1.. = materials (Ti, Cu, Fe);
    # rho_solid = TOTAL material fraction (sum over all non-void phases)
    if MP.get('energy_interp', True):
        rho_solid = phase_weights[:, 1:].sum(dim=1).detach()       # [N_elem]
        bg = MP.get('gamma_beta', 100.0)
        eg = MP.get('gamma_eta',  0.10)
        gamma = (math.tanh(bg * eg) + torch.tanh(bg * (rho_solid - eg))) / \
                (math.tanh(bg * eg) + math.tanh(bg * (1.0 - eg)))
        gamma = gamma.clamp(0.0, 1.0).unsqueeze(-1)                # [N_elem, 1]
    else:
        gamma = torch.ones(N_elem_w, 1, device=elem_x.device, dtype=elem_x.dtype)
    gamma2 = gamma ** 2
    gamma4d = gamma.view(-1, 1, 1, 1)                              # for [.,.,2,2] tensors

    # --- displacement gradient at integration points ---
    H_u = _disp_gradient(Node_disp, conn, B_T, N_elem_w)           # [N_elem, N_int, 2, 2]

    # --- thermal log-strain eigenstrain  eps_th = alpha(rho,T) * (T - T_inf) ---
    # EXACT additive split in Hencky/log-strain space for ISOTROPIC expansion.
    # T is detached (it is the OTHER state); alpha_hat carries the rho-grad that
    # the fused sensitivity reuses for the alpha(rho) (eigenstrain) pathway.
    T_dev = (T_int_pt.squeeze(-1).detach() - T_inf)               # [N_elem_w, N_int]
    if tdep is not None:
        # exact  eps_th = int_{T_inf}^{T*} alpha(rho,tau) dtau  (rho-live,
        # T detached); reduces to alpha_hat*(T-T_inf) for coef=[1.0].
        eps_th_grad = eps_th_int_hat
    else:
        eps_th_grad = alpha_hat * T_dev                           # rho-live (mech_sens)
    eps_th = eps_th_grad.detach()                                 # detached (primal + adjoint)

    # --- interpolated Hencky strain energy (Wang et al., THREE-term form) ---
    #     psi_e = psi_H(gamma u; eps_th) + psi_L(u; eps_th) - psi_L(gamma u; eps_th)
    # NOTE: the two-term shortcut psi_H(gamma u) + (1-gamma^2) psi_L(u) relies on
    # psi_L(gamma u) = gamma^2 psi_L(u), which FAILS with a thermal eigenstrain
    # (the -ctr*eps_th*tr(eps) cross term is linear in strain).  The three-term
    # form is exact for any eps_th and cancels the spurious pure-eigenstrain
    # energy at gamma=0 identically (psi_H(0;eps_th) == psi_L(0;eps_th)).
    #     detached material (C*_no_grad) + detached gamma: design grad lives only
    #     in mech_sens.  Reduces to the small-strain code as H -> 0, gamma -> 1.
    # linelas: psi_H == psi_L makes the three-term form collapse to psi_L(u)
    # identically — implemented directly so gamma is fully inert.
    if linelas:
        SE_ip = W_density_linear_thermal(H_u, C1_no_grad, C2_no_grad, nu, eps_th)
    else:
        SE_ip = (W_density_hencky_thermal(gamma4d * H_u, C1_no_grad, C2_no_grad, nu, eps_th)
                 + W_density_linear_thermal(H_u, C1_no_grad, C2_no_grad, nu, eps_th)
                 - W_density_linear_thermal(gamma4d * H_u, C1_no_grad, C2_no_grad, nu, eps_th))
    SE_vector = torch.sum(SE_ip * wt * detJ * thickness, dim=1)
    strain_energy = SE_vector.sum()

    # =====================================================================
    # >>> DEAD-MATERIAL penalty (island killer for EQUALITY-mass runs)
    # ---------------------------------------------------------------------
    # Under the equality mass constraint, mass dumped as disconnected
    # islands is objective-neutral (zero load path -> zero lambda^T dR/drho)
    # and the constraint actively rewards it.  This term makes dead material
    # cost something: solid fraction weighted by an indicator of ~zero
    # ELASTIC strain-energy density,
    #     psi_dead = exp( -SE_e / (dead_se_frac * mean(SE)) )   (detached),
    #     P_dead   = sum_e rho_s,e * psi_dead,e * V_e / V_total.
    # Islands expand freely against the thermal eigenstrain (eps = eps_th,
    # SE = 0 -> psi ~ 1) while load-bearing members carry SE >> 0 (psi ~ 0),
    # so d P_dead / d rho_s > 0 EXACTLY in dead regions: a monotone local
    # dissolving gradient with no barrier (unlike a perimeter penalty, whose
    # gradient DENSIFIES blobs instead of removing them -- verified and
    # rejected).  SE is detached: this is a regularizer, not physics; the
    # adjoint is untouched.  CAVEATS: (i) SE is the NN-equilibrium estimate,
    # noisy early -> gated by the post-mass-ramp continuation schedule;
    # (ii) material that is mechanically idle but THERMALLY functional
    # (pure heat-spreading paths) is also penalised -- keep w_dead modest.
    # =====================================================================
    if MP.get('w_dead', 0.0) > 0.0:
        rho_s_live = phase_weights[:, 1:].sum(dim=1)          # design-live
        SE_dens = (SE_vector / elem_vol.clamp_min(1e-30)).detach()
        SE_ref = (SE_dens.mean() * MP.get('dead_se_frac', 0.05)).clamp_min(1e-30)
        psi_dead = torch.exp(-SE_dens / SE_ref)               # detached
        dead_penalty = (rho_s_live * psi_dead * elem_vol).sum() / elem_vol.sum()
    else:
        dead_penalty = torch.tensor(0.0, device=elem_x.device,
                                    dtype=elem_x.dtype)

    # deformation-magnitude diagnostic (monitors validity of the NL regime)
    Hmax = H_u.detach().abs().amax()

    # ========================
    # 3. estimate adjoint equations for displacement (TANGENT form)
    # ========================
    # For a NONLINEAR state problem the adjoint is governed by the tangent
    # stiffness at the converged state:  K_T(u)·λ = f_adj.  Variationally,
    # λ minimises the quadratic functional (second variation of ψ_e):
    #
    #   Π_adj(λ) = ½∫ [ γ²( ΔE_λ : C : ΔE_λ  +  S : (Grad λᵀ Grad λ) )
    #                  + (1−γ²) ε(λ) : C : ε(λ) ] dV
    #              + ½ K_out λ_out²  −  f_adj·λ
    #
    # with ΔE_λ = sym(F_γᵀ Grad λ) the linearised Green–Lagrange strain
    # (MATERIAL part) and S : Grad λᵀ Grad λ the GEOMETRIC (initial-stress)
    # part.  F_γ and S are detached — the adjoint solve is at fixed state.
    # K_T is PD at a stable equilibrium; if a design passes near buckling
    # the geometric term can go indefinite — set MP['adjoint_geometric'] =
    # False to fall back to the material-only (Gauss–Newton) tangent.
    # Reduces to the previous self-adjoint linear solve as F → I, γ → 1.
    disp_f_adj = Node_disp_adj[f_adj_index]
    external_work_adj = torch.sum(disp_f_adj * f_adj_magnitude)

    disp_K_out_adj = Node_disp_adj[K_out_index]
    external_work_K_out_adj = 0.5 * torch.sum((disp_K_out_adj**2) * K_out_magnitude)
    spring_energy_adj = external_work_K_out_adj

    # Grad lambda at integration points (carries theta_lambda)
    H_l = _disp_gradient(Node_disp_adj, conn, B_T, N_elem_w)      # [N_elem, N_int, 2, 2]
    H_u_det = H_u.detach()                                        # frozen primal state

    # K_T(u*) via nested-autograd HVP on the interpolated thermal-Hencky energy:
    #   1/2 lambda^T K lambda = 1/2 d^2/dtau^2 Pi_int(H_u_det + tau H_l)|_0.
    # No hand-coded 4th-order Hencky tangent; d2 stays differentiable wrt
    # theta_lambda because H_l carries it.
    #
    # Which tangent is assembled here is set by MP['adjoint_solver'] and, for the
    # energy solver only, MP['adjoint_geometric']:
    #
    #   adjoint_solver='residual' (default) -> ALWAYS the FULL Newton tangent
    #     K_T = K_mat + K_geo.  The full tangent is the ONLY consistent adjoint
    #     operator (dropping K_geo gives the wrong lambda / a biased sensitivity).
    #     K_T can be indefinite near a limit point, so it is NOT solved by the
    #     energy (which would be unbounded below); the training loop instead
    #     minimises the KKT-stationarity residual  || d Pi_adj / d theta_l ||^2,
    #     which is bounded below (>=0), keeps the full indefinite K_T, and whose
    #     unique zero is the exact adjoint.  (Verified on a synthetic indefinite
    #     system: energy form diverges, residual form is exact.)
    #
    #   adjoint_solver='energy' -> legacy DEM energy min 1/2 l^T K l - f.l, with
    #     MP['adjoint_geometric'] selecting the tangent:
    #        True  -> full Newton tangent (exact but can run away near buckling)
    #        False -> Gauss-Newton material-only tangent: SPD (bounded energy),
    #                 but an APPROXIMATE adjoint (drops the geometric term).  The
    #                 GN material term is the exact log-strain curvature dE:C_H:dE
    #                 via the strain-linear C-path (thermal pre-stress retained);
    #                 the void branch is linear so already material-only.
    #
    # The temperature adjoint (Section 4) and the fused design sensitivity
    # (Section 5) consume the DETACHED lambda, so they are unaffected by the
    # solver choice — only the quality of lambda changes.
    _use_full_tangent = (MP.get('adjoint_solver', 'residual') == 'residual') \
                        or MP.get('adjoint_geometric', False) \
                        or direct_adj \
                        or linelas     # linear law: no geometric term, so the
                                       # full tangent IS the (SPD) material one.
                        # direct λ solves the FULL tangent, so the
                        # diagnostic must use it too: vw_ratio is
                        # then exactly 1 iff assembly==HVP tangent.
    tau = torch.zeros((), device=elem_x.device, dtype=elem_x.dtype, requires_grad=True)
    if _use_full_tangent:
        # ---- full Newton tangent (material + geometric); consistent adjoint ----
        Hs = H_u_det + tau * H_l
        if linelas:
            E_s = (W_density_linear_thermal(Hs, C1_no_grad, C2_no_grad, nu, eps_th)
                   * wt * detJ * thickness).sum()
        else:
            E_s = ((W_density_hencky_thermal(gamma4d * Hs, C1_no_grad, C2_no_grad, nu, eps_th)
                    + W_density_linear_thermal(Hs, C1_no_grad, C2_no_grad, nu, eps_th)
                    - W_density_linear_thermal(gamma4d * Hs, C1_no_grad, C2_no_grad, nu, eps_th))
                   * wt * detJ * thickness).sum()
    else:
        # ---- Gauss-Newton (material-only) tangent; SPD, APPROXIMATE adjoint ----
        # F_gamma = I + gamma*H_u  (frozen);  strain increment  gamma*grad(lambda)
        Fg = gamma4d * H_u_det                                    # gamma * H_u  (detached)
        Fg11 = 1.0 + Fg[..., 0, 0]; Fg12 = Fg[..., 0, 1]
        Fg21 = Fg[..., 1, 0];       Fg22 = 1.0 + Fg[..., 1, 1]
        C11_0 = Fg11 * Fg11 + Fg21 * Fg21                         # C_gamma at u* (frozen, SPD)
        C22_0 = Fg12 * Fg12 + Fg22 * Fg22
        C12_0 = Fg11 * Fg12 + Fg21 * Fg22
        gHl = gamma4d * H_l                                       # gamma * grad(lambda) (theta_l live)
        dA11 = gHl[..., 0, 0]; dA12 = gHl[..., 0, 1]
        dA21 = gHl[..., 1, 0]; dA22 = gHl[..., 1, 1]
        # dE_gamma = sym(F_gamma^T . (gamma grad lambda)):  M = F_gamma^T dA
        M11 = Fg11 * dA11 + Fg21 * dA21
        M22 = Fg12 * dA12 + Fg22 * dA22
        M12 = Fg11 * dA12 + Fg21 * dA22
        M21 = Fg12 * dA11 + Fg22 * dA21
        dE11 = M11
        dE22 = M22
        dE12 = 0.5 * (M12 + M21)
        # strain-LINEAR path:  C_gamma(tau) = C_gamma0 + 2 tau dE_gamma
        C11t = C11_0 + 2.0 * tau * dE11
        C22t = C22_0 + 2.0 * tau * dE22
        C12t = C12_0 + 2.0 * tau * dE12
        W_H_adj = W_density_hencky_thermal_fromC(C11t, C22t, C12t,
                                                 C1_no_grad, C2_no_grad, nu, eps_th)
        # linear branch of the THREE-term Wang form: psi_L(Hs) - psi_L(gamma Hs)
        # (exact with eigenstrain; d2/dtau2 of the subtracted term is
        #  -gamma^2 eps(l):C:eps(l), so the GN tangent stays SPD since gamma<=1)
        Hs_gn = H_u_det + tau * H_l
        W_L_adj = (W_density_linear_thermal(Hs_gn, C1_no_grad, C2_no_grad, nu, eps_th)
                   - W_density_linear_thermal(gamma4d * Hs_gn, C1_no_grad, C2_no_grad, nu, eps_th))
        E_s = ((W_H_adj + W_L_adj) * wt * detJ * thickness).sum()
    d1 = torch.autograd.grad(E_s, tau, create_graph=True)[0]
    # DIRECT mode: strain_energy_adj is a pure DIAGNOSTIC (lambda is a solved
    # constant, not in the loss), so d2 needs NO further graph.  With
    # create_graph=False this grad call also FREES d1's graph and E_s's saved
    # activations (all three Wang energy branches over every fine element)
    # immediately, instead of holding them until the tensors are rebound next
    # epoch — a multi-GB per-epoch retention under hencky+energy_interp.
    # DEM modes still need create_graph=True (strain_energy_adj enters the
    # loss / KKT residual and is differentiated w.r.t. theta_lambda).
    d2 = torch.autograd.grad(d1, tau, create_graph=not direct_adj)[0]  # = lambda^T K lambda
    strain_energy_adj = 0.5 * d2

    # ========================
    # 4. estimate adjoint equations for temperature
    # ========================
    T_adj_elem = Node_T_adj[conn].reshape(N_elem_w, -1)
    T_adj_elem = T_adj_elem.unsqueeze(1).unsqueeze(-1)
    T_adj_elem = T_adj_elem.expand(-1, N.shape[1], -1, -1)

    # dR_u/dT coupling (thermo-mechanical source for the temperature adjoint):
    #   d t/dT = -C:(alpha I)  =>  source proportional to alpha * tr(C : d eps_H/du . lambda).
    # For Hencky the trace of the linearised log-strain along grad(lambda) is
    # EXACTLY  d(tr eps_H)/du . lambda = F^{-T} : grad(lambda)   (tr eps_H = log J;
    # verified to 1e-16) -- this is the ONLY change from the SVK trace, which used
    # F : grad(lambda).  Interpolated like the residual: gamma to the FIRST power
    # on the NL branch (at F_gamma = I + gamma H_u), (1-gamma^2) on the linear
    # branch.  C-trace coefficient C1*(1+nu) = E/(1-nu).  All detached.
    H_l_det = H_l.detach()
    L11 = H_l_det[..., 0, 0]; L12 = H_l_det[..., 0, 1]
    L21 = H_l_det[..., 1, 0]; L22 = H_l_det[..., 1, 1]
    trace_coef = C1_no_grad * (1.0 + nu)                          # = E/(1-nu)  [N_elem,1]
    if linelas:
        # psi = psi_L(u) exactly: d(tr eps)/du . lambda = tr(grad lambda).
        tr_adj = (trace_coef * (L11 + L22)).detach()              # [N_elem_w, N_int]
    else:
        Hg_d = (gamma4d * H_u_det)                                # F_gamma - I (detached)
        Fg11 = 1.0 + Hg_d[..., 0, 0]; Fg12 = Hg_d[..., 0, 1]
        Fg21 = Hg_d[..., 1, 0];       Fg22 = 1.0 + Hg_d[..., 1, 1]
        Jg = (Fg11 * Fg22 - Fg12 * Fg21).clamp_min(1e-9)
        FinvT_gradL = (Fg22 * L11 - Fg21 * L12 - Fg12 * L21 + Fg11 * L22) / Jg   # F_gamma^{-T}:grad(lambda)
        # three-term Wang: psi_L(H) - psi_L(gamma H) gives (1 - gamma) * tr(grad l)
        # (the mixed d2 psi_L/du deps_th is LINEAR in the strain path, so psi_L(gamma H)
        #  contributes gamma^1, not gamma^2)
        tr_adj = (gamma * trace_coef * FinvT_gradL
                  + (1.0 - gamma) * trace_coef * (L11 + L22)).detach()     # [N_elem_w, N_int]
    # d(eps_th)/dT = alpha_inst(T*) exactly (derivative of the integral);
    # constant-alpha path uses alpha_no_grad as before.
    _alpha_dT_ng = alpha_no_grad if alpha_inst_ng is None else alpha_inst_ng
    s_adj = (a_u / a_T) * (_alpha_dT_ng * tr_adj)                # [N_elem_w, N_int]

    # --- E(T) pathway of the thermo-mechanical coupling dR_u/dT ---------------
    # The interpolated energy is linear homogeneous in E_hat (all three Wang
    # terms; C1, C2 both ~ E_hat), so per integration point EXACTLY
    #     d(lambda^T f_int)/dT |_E  =  (dE_hat/dT / E_hat) * dW/dtau|_0 ,
    # with dW/dtau|_0 the directional derivative of the frozen-state energy
    # density along lambda (the same quantity mech_sens integrates).  The
    # per-point field is obtained with ONE autograd call through a PER-POINT
    # tau tensor.  Sign matches the alpha pathway above:
    #     s_adj = -(a_u/a_T) * d(lambda^T f_int)/dT
    # (verified against the alpha term:  d t/d eps_th : eps(lambda)
    #  = -trace_coef * tr(eps_lambda), and the code carries +alpha*trace_coef).
    # All inputs detached except the scalar-per-point tau; costs one extra
    # energy eval + backward per epoch.
    if eratio_ng is not None:
        tauE = torch.zeros(H_u_det.shape[0], H_u_det.shape[1],
                           device=elem_x.device, dtype=elem_x.dtype,
                           requires_grad=True)
        HsE = H_u_det + tauE.unsqueeze(-1).unsqueeze(-1) * H_l_det
        if linelas:
            W_E = W_density_linear_thermal(HsE, C1_no_grad, C2_no_grad, nu, eps_th)
        else:
            W_E = (W_density_hencky_thermal(gamma4d * HsE, C1_no_grad, C2_no_grad, nu, eps_th)
                   + W_density_linear_thermal(HsE, C1_no_grad, C2_no_grad, nu, eps_th)
                   - W_density_linear_thermal(gamma4d * HsE, C1_no_grad, C2_no_grad, nu, eps_th))
        dWdtau_ip = torch.autograd.grad(W_E.sum(), tauE)[0].detach()  # [Ne, N_int]
        s_adj = s_adj - (a_u / a_T) * (eratio_ng * dWdtau_ip)

    T_adj_int_pt = torch.matmul(N, T_adj_elem).squeeze(-1)
    sT_adj = s_adj.unsqueeze(-1) * T_adj_int_pt
    sT_int_adj = (sT_adj.squeeze(-1) * detJ * wt.unsqueeze(0)) * thickness
    source_energy_adj = sT_int_adj.sum()

    dT_adj = torch.matmul(B_T, T_adj_elem).squeeze(-1)
    dT_adj_square = (dT_adj ** 2).sum(dim=-1)
    kdT2_adj =  kappa_no_grad * dT_adj_square
    kdT2_adj_int = (kdT2_adj * detJ * wt.unsqueeze(0)) * thickness
    heat_energy_adj = 0.5 * kdT2_adj_int.sum()

    # --- kappa'(T) tangent term of the temperature adjoint --------------------
    # With kappa = kappa(rho,T) the thermal residual's T-tangent gains the
    # NONSYMMETRIC term  int kappa'(T) dT_pert (grad T . grad w);  transposed
    # onto the adjoint it contributes  + int kappa'(T*) (grad T* . grad l_T) w
    # to the l_T weak form:
    #     int [ kappa grad(w).grad(l_T) + kappa'(T*) (grad T*.grad l_T) w ]
    #       = int s_adj w .
    # A nonsymmetric operator has no energy functional, so the term enters the
    # DEM loss Picard-lagged in l_T:
    #     Pi_conv = int kappa'(T*) (grad T* . grad l_T,det) l_T dV ,
    # whose theta_{l_T} gradient is exactly the required weak term once
    # l_T,det = l_T at the Adam fixed point (same staggered pattern as the
    # rest of the loop).  Everything except the LIVE l_T value is detached.
    if dkappa_dT_ng is not None:
        gTgL_det = (dT.detach() * dT_adj.detach()).sum(dim=-1)    # grad T . grad l_T
        conv_ip = dkappa_dT_ng * gTgL_det * T_adj_int_pt.squeeze(-1)
        conv_energy_adj = (conv_ip * detJ * wt.unsqueeze(0)).sum() * thickness
    else:
        conv_energy_adj = torch.zeros((), device=elem_x.device, dtype=elem_x.dtype)

    # ========================
    # 5. calculate the augmented objective function
    # ========================
    # FUSED design sensitivity (autograd; matches the FD-verified compliance
    # pattern, extended with the thermal alpha(rho) pathway).
    #   mech_sens = lambda^T f_int(u*; rho)
    #            = d/dtau2 [ Pi_int(H_u_det + tau2 H_l_det; rho) ] |_{tau2=0}
    # built with DETACHED state H_u_det and DETACHED adjoint H_l_det, but
    # rho-LIVE material (C1_hat, C2_hat), rho-LIVE gamma_g, and rho-LIVE thermal
    # eigenstrain eps_th_grad.  grad_rho(mech_sens) then carries ALL mechanical
    # design pathways exactly:
    #   (i)   dC/drho        (modulus)
    #   (ii)  dE_th/drho     via alpha(rho)   (thermal eigenstrain)
    #   (iii) dgamma/drho    (Wang energy-interpolation factor)
    # The NL-branch chain rule reproduces the residual gamma t(gamma u):grad(lambda)
    # (the explicit gamma and the F_gamma-evaluated stress both come out of
    # autograd); the linear branch gives (1-gamma^2) sigma_L(u):eps(lambda).
    # The geometric/initial-stress term adds NO rho-term (it lives in K_T, not
    # the residual).  Sign: obj += -mech_sens  =>  grad_rho(obj) = lambda^T dR_u/drho,
    # matching the previous -comp + thermal convention.
    if MP.get('energy_interp', True):
        rho_solid_g = phase_weights[:, 1:].sum(dim=1)             # WITH grad (filter + GP)
        bg = MP.get('gamma_beta', 100.0)
        eg = MP.get('gamma_eta',  0.10)
        gamma_g = (math.tanh(bg * eg) + torch.tanh(bg * (rho_solid_g - eg))) / \
                  (math.tanh(bg * eg) + math.tanh(bg * (1.0 - eg)))
        gamma_g = gamma_g.clamp(0.0, 1.0).unsqueeze(-1)          # [N_elem, 1]
    else:
        gamma_g = gamma
    gamma_g4d = gamma_g.view(-1, 1, 1, 1)

    tau2 = torch.zeros((), device=elem_x.device, dtype=elem_x.dtype, requires_grad=True)
    Hs2 = H_u_det + tau2 * H_l_det                               # detached state + detached adjoint
    if linelas:
        # psi_L is rho-live via C1_hat/C2_hat and eps_th_grad; the gamma(rho)
        # pathway is identically zero (Wang form collapsed), matching the
        # original linear-elastic sensitivity lambda^T dR/drho exactly.
        E_g = (W_density_linear_thermal(Hs2, C1_hat, C2_hat, nu, eps_th_grad)
               * wt * detJ * thickness).sum()
    else:
        E_g = ((W_density_hencky_thermal(gamma_g4d * Hs2, C1_hat, C2_hat, nu, eps_th_grad)
                + W_density_linear_thermal(Hs2, C1_hat, C2_hat, nu, eps_th_grad)
                - W_density_linear_thermal(gamma_g4d * Hs2, C1_hat, C2_hat, nu, eps_th_grad))
               * wt * detJ * thickness).sum()
    M_scalar = torch.autograd.grad(E_g, tau2, create_graph=True)[0]   # lambda^T f_int(rho)
    mech_sens = a_u * M_scalar

    # calculate the conductivity adjoint term:
    dTdvT = (dT * dT_adj).sum(dim=-1)
    kdTdvT =  kappa_hat * (dTdvT.detach())
    kdTdvT_int = (kdTdvT * detJ * wt.unsqueeze(0)) * thickness
    heat = a_T * kdTdvT_int.sum()

    # calculate the source adjoint term:
    source_ip = (s_hat.unsqueeze(-1) * T_adj_int_pt.detach())
    source_vector = torch.sum(source_ip.squeeze(-1) * wt * detJ * thickness, dim=1)
    source = a_T * source_vector.sum()

    # --- augmented objective ---
    # fused mech_sens = λᵀ(∂R/∂ρ) replaces the old separate (−comp + thermal)
    # pair and additionally carries the γ(ρ)-pathway (see Section 5 notes)
    #
    # SIGN (FD-established on a Newton-resolved reduced objective, ratio +1.000000
    # to 9 digits):  grad_rho( -mech_sens - heat + source ) = +d(u_out)/d(rho),
    # where u_out is the output displacement ALONG f_adj_magnitude.  Descent on
    # that convention MINIMISES the signed stroke — the compliance-minimisation
    # template carried over with the wrong sign for stroke MAXIMISATION (and it
    # contradicted the best-design banking, which maximises u_out).  For
    # maximisation all three adjoint terms flip together (T_adj is linear in λ).
    if MP.get('maximize_uout', False):
        obj = - disp_K_out[0,0].detach() + mech_sens + heat - source
    else:   # legacy convention: descent minimises signed u_out along f_adj
        obj = - disp_K_out[0,0].detach() - mech_sens - heat + source

    # ========================
    # Assemble loss dictionary
    # ========================
    loss_dict = {
        'obj_func': obj,
        'u_out': disp_K_out[0, 0].detach(),  # actual output displacement (signed, NN-equilibrium estimate)
        'strain_energy': strain_energy,
        'spring_energy': spring_energy,
        'external_work': external_work,
        'strain_energy_adj': strain_energy_adj,
        'spring_energy_adj': spring_energy_adj,
        'external_work_adj': external_work_adj,
        'heat_energy': heat_energy,
        'source_energy': source_energy,
        'heat_energy_adj': heat_energy_adj,
        'source_energy_adj': source_energy_adj,
        'conv_energy_adj': conv_energy_adj,   # kappa'(T) T-adjoint tangent term
        'mass': mass,
        'M0': M0,
        'cost': cost,
        'cost0': cost0,
        'grey_counts': grey_counts,
        'count_total': count_total,
        'Hmax': Hmax,                        # max |∂u_i/∂X_j| — finite-strain regime monitor
        'interface_penalty': interface_penalty,
        'dead_penalty': dead_penalty,
    }

    # --- stash the frozen state for the direct adjoint solve -----------------
    # (all detached references; consumed by solve_adjoint_direct every
    #  MP['adjoint_refactor_every'] epochs.  With the solved lambda, the
    #  adjoint energy terms above become pure diagnostics and the logged
    #  vw_ratio checks the assembly/solve consistency every epoch.)
    if direct_adj:
        MP['_adj_stash'] = {'Node_disp': Node_disp.detach(),
                            'gamma': gamma.detach(),
                            'C1': C1_no_grad, 'C2': C2_no_grad,
                            'eps_th': eps_th}

    return loss_dict


# =====================================================================
# Tangent / adjoint diagnostic  (option 3): is the adjoint tangent really
# indefinite at a CONVERGED primal, or is the indefiniteness an artifact of an
# under-converged primal NN (K_T evaluated off the true equilibrium)?
#
# Reports, at the current frozen state:
#   * Hmax = max|grad u|                          (finite-strain regime)
#   * ||d Pi_primal/d theta_u||                   (primal DEM convergence; if
#                                                  large, K_T is off-equilibrium)
#   * lambda_min, lambda_max of the PARAMETER-SPACE adjoint Hessian
#         H = d^2 Pi_adj / d theta_lambda^2  (= J^T K_T J, the operator whose
#     indefiniteness makes the energy-form adjoint run away), via matrix-free
#     Lanczos using autograd Hessian-vector products (full tangent, BCs baked in
#     through the trial map).  lambda_min < 0  =>  genuinely indefinite in the
#     trial space (real limit point) -> the residual solver is required.
#     lambda_min >= 0 at a converged primal => the runaway was an off-equilibrium
#     artifact; tightening the primal / load ramping fixes it.
# =====================================================================
def diagnose_tangent(model_list, epoch, n_lanczos=24, seed=0):
    ld = calculate_TO_loss(model_list)               # fresh forward, full tangent
    Pi_adj = ld['strain_energy_adj'] + ld['spring_energy_adj'] - ld['external_work_adj']
    params = [p for p in model_list[1].parameters() if p.requires_grad]
    g0 = torch.autograd.grad(Pi_adj, params, create_graph=True, allow_unused=True)
    active = [(p, g) for p, g in zip(params, g0) if g is not None]
    if not active:
        print(f"[Epoch {epoch}] tangent diagnostic: adjoint params not in graph.")
        return
    aparams = [p for p, _ in active]
    gflat = torch.cat([g.reshape(-1) for _, g in active])
    numel = [p.numel() for p in aparams]; shapes = [p.shape for p in aparams]

    def _unflat(v):
        out, i = [], 0
        for n, s in zip(numel, shapes):
            out.append(v[i:i + n].view(s)); i += n
        return out

    def hvp(vflat):
        vlist = _unflat(vflat)
        Hv = torch.autograd.grad(gflat, aparams, grad_outputs=vflat,
                                 retain_graph=True, allow_unused=True)
        return torch.cat([(hv.reshape(-1) if hv is not None else torch.zeros_like(p).reshape(-1))
                          for hv, p in zip(Hv, aparams)])

    # --- matrix-free Lanczos (full reorthogonalization) ---
    P = gflat.numel(); m = min(n_lanczos, P)
    gen = torch.Generator(device=gflat.device).manual_seed(seed)
    q = torch.randn(P, generator=gen, device=gflat.device, dtype=gflat.dtype); q /= q.norm()
    Qk, alpha, beta = [], [], []
    q_prev = torch.zeros_like(q); b = 0.0
    for _ in range(m):
        w = hvp(q)
        a = float(q @ w); alpha.append(a)
        w = w - a * q - b * q_prev
        for qi in Qk:
            w = w - (w @ qi) * qi
        Qk.append(q.clone()); b = float(w.norm())
        if b < 1e-10:
            break
        beta.append(b); q_prev = q; q = w / b
    T = torch.diag(torch.tensor(alpha, dtype=gflat.dtype))
    for i, bb in enumerate(beta):
        T[i, i + 1] = bb; T[i + 1, i] = bb
    ritz = torch.linalg.eigvalsh(T)
    lam_min, lam_max = float(ritz.min()), float(ritz.max())

    # --- primal DEM convergence residual ---
    Pi_primal = ld['strain_energy'] + ld['spring_energy'] - ld['external_work']
    u_params = [p for p in model_list[0].parameters() if p.requires_grad]
    gu = torch.autograd.grad(Pi_primal, u_params,
                             retain_graph=False, allow_unused=True)
    primal_res = float(torch.sqrt(sum((g * g).sum() for g in gu if g is not None)))
    Hmax_val = float(ld['Hmax'])

    print(f"[Epoch {epoch}] tangent diag: Hmax={Hmax_val:.3g}  "
          f"primal_res(||dPi/dtheta_u||)={primal_res:.3e}  "
          f"lambda_min={lam_min:+.3e}  lambda_max={lam_max:+.3e}  "
          f"indefinite={lam_min < 0}")


# ========================
# Worker: single GPU debug
# ========================
def run_worker(MP, NN_config_disp, NN_config_rho, train_data, index_CP, mesh_data):
    device = torch.device("cuda:0")

    Example = MP["Example"]
    Case = NN_config_disp["Case"]
    base_folder = MP["base_folder"]
    run_folder = f"{base_folder}/Results/{Example}/{Case}/"

    num_phase = MP["num_phase"]
    frac_decrease = MP['frac_decrease']
    TO_num_iter = NN_config_disp["TO_num_iter"]
    plotting_interval_TO = NN_config_disp["plotting_interval_TO"]
    delta = NN_config_disp["delta"]
    wc = NN_config_disp["wc"]
    wd = NN_config_disp["wd"]
    wm = NN_config_disp["wm"]
    wp = NN_config_disp["wp"]
    wtemp = NN_config_disp["wtemp"]
    w_interface = NN_config_disp.get("w_interface", 0.0)
    wd_adj = MP.get('w_adj_residual', None)
    wd_adj = wd if wd_adj is None else wd_adj   # weight on the KKT-residual adjoint term
    gradient_clip = NN_config_disp["gradient_clip"]
    nrmThreshold = NN_config_disp["nrmThreshold"]
    dynamic_weight = NN_config_disp["dynamic_weight"]
    random_state = NN_config_disp["random_state"]
    learning_rate_disp = NN_config_disp["learning_rate_disp"]
    learning_rate_disp_adj = NN_config_disp["learning_rate_disp_adj"]
    learning_rate_T = NN_config_disp["learning_rate_T"]
    learning_rate_T_adj = NN_config_disp["learning_rate_T_adj"]
    learning_rate_rho = NN_config_disp["learning_rate_rho"]
    Diff_type = MP["Diff_type"]

    # loop over random states (multiple runs)
    for i, state in enumerate(random_state, start=1):
        set_seed0(state)
        NN_config_disp["state"] = state

        # save folder with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_folder = f"{run_folder}Run_{state}_{timestamp}/"
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        os.makedirs(save_folder)
        NN_config_disp["save_folder"] = save_folder
        NN_config_rho["save_folder"] = save_folder

        # ========================
        # Build models
        # ========================
        model_disp = LMGP(train_data['disp']['X'], train_data['disp']['y'], NN_config_disp,
                        name_output="disp", MP=MP, num_output=2).to_device(device)
        model_disp_adj = LMGP(train_data['disp_adj']['X'], train_data['disp_adj']['y'], NN_config_disp,
                        name_output="adj", MP=MP, num_output=2).to_device(device)
        model_T = LMGP(train_data['T']['X'], train_data['T']['y'], NN_config_disp,
                        name_output="T", MP=MP, num_output=1).to_device(device)
        model_T_adj = LMGP(train_data['T_adj']['X'], train_data['T_adj']['y'], NN_config_disp,
                        name_output="T_adj", MP=MP, num_output=1).to_device(device)
        model_phase = LMGP(train_data['phase']['X'], train_data['phase']['y'], NN_config_rho,
                        name_output="rho", MP=MP, num_output=num_phase).to_device(device)

        model_list = [model_disp, model_disp_adj, model_T, model_T_adj, model_phase]

        # define the time history dict
        timeHistory = {
            'loss_total': [], 'loss_obj': [], 'loss_dem_disp': [],'loss_dem_disp_adj':[],
            'loss_dem_t': [],'loss_dem_t_adj':[],
            'loss_mConstraint': [], 'loss_cConstraint': [], 'loss_tv': [],
            'interface_penalty': [], 'w_interface': [],
            'strain_energy': [], 'external_work': [], 'spring_energy':[],
            'strain_energy_adj': [], 'external_work_adj': [], 'spring_energy_adj':[],
            'heat_energy': [], 'source_energy': [],
            'heat_energy_adj': [], 'source_energy_adj': [],
            'loss_pde_1': [], 'loss_pde_2': [],
            'massfrac': [], 'costfrac': [],
            'wc': [], 'wd': [], 'wm': [], 'wp': [],'wtemp':[],
            'grey': [], 'p': [], 'Hmax': [], 'adj_res': []
        }

        # ========================
        # Optimizers + schedulers
        # ========================
        optimizer_disp = torch.optim.Adam(model_list[0].parameters(),
                                          lr=learning_rate_disp, amsgrad=True)
        optimizer_disp_adj = torch.optim.Adam(model_list[1].parameters(),
                                          lr=learning_rate_disp_adj, amsgrad=True)
        # trainable adjoint params only: torch.autograd.grad rejects inputs with
        # requires_grad=False (the LMGP model carries frozen params/buffers).
        adj_params = [p for p in model_list[1].parameters() if p.requires_grad]
        optimizer_T = torch.optim.Adam(model_list[2].parameters(),
                                          lr=learning_rate_T, amsgrad=True)
        optimizer_T_adj = torch.optim.Adam(model_list[3].parameters(),
                                          lr=learning_rate_T_adj, amsgrad=True)
        optimizer_rho = torch.optim.Adam(model_list[4].parameters(),
                                         lr=learning_rate_rho, amsgrad=True)

        scheduler_disp = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_disp,
            milestones=torch.linspace(0, TO_num_iter, 4).tolist(),
            gamma=0.75
        )
        scheduler_disp_adj = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_disp_adj,
            milestones=torch.linspace(0, TO_num_iter, 4).tolist(),
            gamma=0.75
        )
        scheduler_T = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_T,
            milestones=torch.linspace(0, TO_num_iter, 4).tolist(),
            gamma=0.75
        )
        scheduler_T_adj = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_T_adj,
            milestones=torch.linspace(0, TO_num_iter, 4).tolist(),
            gamma=0.75
        )
        scheduler_rho = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_rho,
            milestones=torch.linspace(0, TO_num_iter, 4).tolist(),
            gamma=0.75
        )

        # ========================
        # Training loop
        # ========================
        epochs_iter = tqdm(range(TO_num_iter),
                           desc="TO Epoch",
                           position=0, leave=True)
        total_time, start_time = 0, time.time()

        # ---- monitoring / best-checkpoint state ------------------------------
        # The finite-strain adjoint can go indefinite near a limit point (the
        # geometric term flips K_T's curvature), which drives the DEM adjoint —
        # and hence the design sensitivities — into a runaway tail.  We log
        # Hmax = max|grad u| and the adjoint curvature 1/2 lambda^T K lambda
        # (= strain_energy_adj), and keep the best *stable* design in memory so
        # the final fields/DXF reflect it instead of a diverged tail.  Training
        # itself always runs the full schedule (no early stopping).
        best_state = None
        best_uout  = -float('inf')
        best_epoch = -1
        warmup_ep  = TO_num_iter // 2                   # only bank converged designs
        H_cap      = MP.get('Hmax_cap', 1.0)            # finite-strain validity cap for banking
        sea_tol    = 1e-6                               # adjoint-curvature negativity tol
        vw_tol     = MP.get('adjoint_vw_tol', 0.15)     # |EW/(2(SE+SP)) - 1| tolerance: adjoint
                                                        #   counted as solved for banking "best".
                                                        #   Primal and adjoint share ONE mesh and
                                                        #   quadrature, so the virtual-work identity
                                                        #   is an exact same-mesh check again (no
                                                        #   coarsening slack; gate back to 0.15).
        # direct-adjoint runtime state
        direct_res = float('nan')                       # last direct-solve rel residual
        MP['lambda_nodal'] = {}                         # solved adjoint per mesh slice
        MP.pop('_adj_coo_idx', None)                    # rebuild COO/BC cache per run
        MP.pop('_gcov_cache', None)                     # rebuild GP cross-cov cache per run
        MP.pop('_linelas_Kunit', None)                  # rebuild fast-tangent cache per run

        def _snapshot_cpu(ml):
            return {
                'model_u_state_dict':     {k: v.detach().cpu().clone() for k, v in ml[0].mean_module_NN_All.state_dict().items()},
                'model_u_adj_state_dict': {k: v.detach().cpu().clone() for k, v in ml[1].mean_module_NN_All.state_dict().items()},
                'model_T_state_dict':     {k: v.detach().cpu().clone() for k, v in ml[2].mean_module_NN_All.state_dict().items()},
                'model_T_adj_state_dict': {k: v.detach().cpu().clone() for k, v in ml[3].mean_module_NN_All.state_dict().items()},
                'model_rho_state_dict':   {k: v.detach().cpu().clone() for k, v in ml[4].mean_module_NN_All.state_dict().items()},
            }

        for epoch in epochs_iter:
            optimizer_disp.zero_grad()
            optimizer_disp_adj.zero_grad()
            optimizer_T.zero_grad()
            optimizer_T_adj.zero_grad()
            optimizer_rho.zero_grad()

            # --- mesh slice ---
            index = index_CP[epoch]
            mesh_key = f"Mesh_{index:02d}"
            GPU_key = "GPU0"

            # assign mesh data to model_u
            model_list[0].collocation_x = mesh_data[GPU_key][mesh_key]["X_node"]
            model_list[0].elem_x        = mesh_data[GPU_key][mesh_key]["X_elem"]
            model_list[0].elem_vol        = mesh_data[GPU_key][mesh_key]["elem_vol"]
            model_list[0].K_out_index       = mesh_data[GPU_key][mesh_key]["K_out_index"]
            model_list[0].K_out_magnitude   = mesh_data[GPU_key][mesh_key]["K_out_magnitude"]
            model_list[0].f_adj_index       = mesh_data[GPU_key][mesh_key]["f_adj_index"]
            model_list[0].f_adj_magnitude   = mesh_data[GPU_key][mesh_key]["f_adj_magnitude"]
            model_list[0].conn          = mesh_data[GPU_key][mesh_key]["conn"]
            model_list[0].B             = mesh_data[GPU_key][mesh_key]["B"]
            model_list[0].B_T             = mesh_data[GPU_key][mesh_key]["B_T"]
            model_list[0].N             = mesh_data[GPU_key][mesh_key]["N"]
            model_list[0].detJ          = mesh_data[GPU_key][mesh_key]["detJ"]

            # Tell the Helmholtz filter which mesh we're on (for caching)
            MP['cur_mesh_key'] = mesh_key

            # bias shift for the temperature NN to match TD:
            if epoch == 0:
                with torch.no_grad():
                    T_mean = calculate_T_mean(model_list)
                    bias_shift = MP['TD'] - T_mean.item()
                    model_list[2].mean_module_NN_All.network.last.bias += bias_shift
                    T_mean_updated = calculate_T_mean(model_list)
                    print(f'Initial mean temperature (K): {T_mean_updated}')

            # --- beta continuation for the density projection -----------------
            # beta0 (=2) until beta_start_ep (default: end of mass ramp,
            # frac_decrease), then doubled every beta_double_every epochs,
            # capped at beta_max (=16): 2 -> 4 -> 8 -> 16.
            if MP.get('use_projection', True) and MP['num_phase'] == 2:
                bs = MP.get('beta_start_ep', frac_decrease)
                if epoch < bs:
                    MP['beta_proj'] = MP.get('beta0', 2.0)
                else:
                    MP['beta_proj'] = MP.get('beta0', 2.0) + ((epoch-bs)/(10000-bs)) * (MP.get('beta_max', 16.0)-MP.get('beta0', 2.0))
                    # MP['beta_proj'] = min(
                    #     MP.get('beta0', 2.0)
                    #     * 2.0 ** ((epoch - bs) // MP.get('beta_double_every', 1000) + 1),
                    #     MP.get('beta_max', 16.0))
            else:
                MP['beta_proj'] = None

            # --- calculate local losses ---
            loss_dict = calculate_TO_loss(model_list)

            # for multi-GPU, assemble the global quantities here:
            obj_global           = loss_dict['obj_func']
            strain_energy_global = loss_dict['strain_energy']
            spring_energy_global = loss_dict['spring_energy']
            external_work_global = loss_dict['external_work']
            strain_energy_adj_global = loss_dict['strain_energy_adj']
            spring_energy_adj_global = loss_dict['spring_energy_adj']
            external_work_adj_global = loss_dict['external_work_adj']
            heat_energy_global = loss_dict['heat_energy']
            source_energy_global = loss_dict['source_energy']
            heat_energy_adj_global = loss_dict['heat_energy_adj']
            source_energy_adj_global = loss_dict['source_energy_adj']
            conv_energy_adj_global = loss_dict['conv_energy_adj']
            mass_global          = loss_dict['mass']
            M0_global            = loss_dict['M0']
            cost_global          = loss_dict['cost']
            cost0_global         = loss_dict['cost0']
            grey_counts_global   = loss_dict['grey_counts']
            count_total_global   = loss_dict['count_total']
            interface_penalty_global = loss_dict['interface_penalty']
            dead_penalty_global = loss_dict['dead_penalty']

            # calculate fraction
            massfrac = mass_global / M0_global
            costfrac = cost_global / cost0_global
            grey_fraction = grey_counts_global / count_total_global

            # set initial volume or cost fraction based on the initial prediction of NN
            if epoch == 0:
                initial_massfrac = massfrac.detach().item()
                initial_costfrac = costfrac.detach().item()
                frac_step_mass = (initial_massfrac - massfrac_f)/frac_decrease
                frac_step_cost = (initial_costfrac - costfrac_f)/frac_decrease
                frac_step_p = (MP['pf'] - MP['p0'])/frac_decrease

                model_list[0].MP['p'] = MP['p0']
                model_list[0].MP['massfrac_star'] = initial_massfrac
                model_list[0].MP['costfrac_star'] = initial_costfrac
                model_list[0].MP['frac_step_p'] = frac_step_p
                model_list[0].MP['frac_step_mass'] = frac_step_mass
                model_list[0].MP['frac_step_cost'] = frac_step_cost
                MP['frac_step_mass'] = frac_step_mass
                MP['frac_step_cost'] = frac_step_cost
                MP['frac_step_p'] = frac_step_p
                MP['massfrac0'] = model_list[0].MP['massfrac_star']
                MP['costfrac0'] = model_list[0].MP['costfrac_star']

            # --- build loss terms ---
            massfrac_star = model_list[0].MP['massfrac_star']
            costfrac_star = model_list[0].MP['costfrac_star']
            loss_obj = obj_global
            loss_dem_disp = strain_energy_global + spring_energy_global - external_work_global
            offset_disp_adj = (1 + delta) / 2 * external_work_adj_global.detach()
            loss_dem_disp_adj = strain_energy_adj_global + spring_energy_adj_global - external_work_adj_global + offset_disp_adj
            loss_dem_t = heat_energy_global - source_energy_global
            loss_dem_t_adj = (heat_energy_adj_global - source_energy_adj_global
                              + conv_energy_adj_global)
            # --- mass / cost constraints -------------------------------------
            # EQUALITY (default): two-sided quadratic wm*((frac/star)-1)^2 for
            # the ENTIRE run — massfrac must TRACK the ramped target star
            # (initial -> massfrac_f), over- AND under-target both penalized.
            # This is what the old `epoch < frac_decrease+100000` guard did
            # implicitly (its else-branch was unreachable in any run shorter
            # than 100k epochs); now explicit and unconditional.
            # 'inequality': the legacy one-sided form after the ramp ends
            # (only frac > star penalized; the design may shed mass freely).
            if MP.get('mass_constraint', 'equality') == 'equality' \
                    or epoch < frac_decrease:
                loss_mConstraint = torch.square((massfrac / massfrac_star) - 1.0)
            else:
                loss_mConstraint = 10*torch.square(torch.clamp((massfrac / massfrac_star) - 1.0, min=0.0)) #changed
            if MP.get('cost_constraint', 'equality') == 'equality' \
                    or epoch < frac_decrease:
                loss_cConstraint = torch.square((costfrac / costfrac_star) - 1.0)
            else:
                loss_cConstraint = torch.square(torch.clamp((costfrac / costfrac_star) - 1.0, min=0.0))
            # --- displacement-adjoint contribution to the loss ---------------
            # 'residual' : minimise the KKT-stationarity residual of the FULL
            #   (indefinite-capable) tangent adjoint energy,
            #        R_adj = 1/2 || d Pi_adj / d theta_lambda ||^2 ,
            #   Pi_adj = 1/2 l^T K l + 1/2 K_out l_out^2 - f_adj . l .
            #   Bounded below by 0; its zero is the exact adjoint K l = f_adj with
            #   the geometric term retained.  Only theta_lambda (model 1) enters
            #   Pi_adj (state / material / gamma are detached), so this updates the
            #   adjoint net alone and leaves the other gradients untouched.
            #   NB: the residual squares the tangent's condition number, so the
            #   adjoint needs enough Adam iterations / a warm start to converge.
            # 'energy'   : legacy DEM energy min (see MP['adjoint_geometric']).
            adjoint_solver = MP.get('adjoint_solver', 'residual')
            adjoint_mode = MP.get('adjoint_mode', 'dem')
            # multi-material COST constraint (P-weighted): off by default;
            # python 0.0 when off so no dead graph edge is kept.
            loss_cc_term = (wp * loss_cConstraint) \
                if MP.get('use_cost_constraint', False) else 0.0
            if adjoint_mode == 'direct':
                # lambda comes from the sparse direct solve — no adjoint-net
                # loss term, no theta_lambda to train.
                adj_res_norm = float('nan')
                loss = (wc * loss_obj + wd * loss_dem_disp
                        + wtemp * (loss_dem_t + loss_dem_t_adj) + wm * loss_mConstraint
                        + loss_cc_term)
            elif adjoint_solver == 'residual':
                Pi_adj = (strain_energy_adj_global + spring_energy_adj_global
                          - external_work_adj_global)
                g_lambda = torch.autograd.grad(
                    Pi_adj, adj_params,
                    create_graph=True, allow_unused=True)
                adj_res_sq = sum((gi * gi).sum() for gi in g_lambda if gi is not None)
                R_adj = 0.5 * adj_res_sq
                adj_res_norm = adj_res_sq.detach().clamp_min(0).sqrt().item()
                loss = (wc * loss_obj + wd * loss_dem_disp + wd_adj * R_adj
                        + wtemp * (loss_dem_t + loss_dem_t_adj) + wm * loss_mConstraint
                        + loss_cc_term)
            else:
                adj_res_norm = float('nan')
                loss = (wc * loss_obj + wd * (loss_dem_disp + loss_dem_disp_adj)
                        + wtemp * (loss_dem_t + loss_dem_t_adj) + wm * loss_mConstraint
                        + loss_cc_term)

            # --- backward ---
            # retain_graph=True kept the ENTIRE epoch graph (primal three-term
            # Hencky energy over every fine element, mech_sens E_g, DEM terms)
            # alive until `loss` was rebound the NEXT epoch — so during each
            # forward the PREVIOUS epoch's multi-GB graph was still resident
            # (~2x peak graph memory).  In direct mode there is exactly ONE
            # backward per epoch and nothing traverses the graph afterwards
            # (the adjoint solve consumes only the detached stash), so the
            # graph is freed here.  Legacy DEM modes keep the old behavior.
            # --- interface penalty: continuation ramp ---------------
            #   Phase 1 (epoch < frac_decrease//4): OFF — let the
            #     optimiser find a good layout first.
            #   Phase 2 (< frac_decrease): weight 0 -> w_interface and
            #     beta_interface beta_if0 -> beta_if_max (sharper
            #     detection as the layout binarises).  NB the beta set
            #     here is consumed by the NEXT epoch's forward (this
            #     epoch's penalty is already computed) — a one-epoch
            #     lag, negligible under Adam step sizes.
            #   Phase 3 (>= frac_decrease): full weight, full beta.
            # START: default = frac_decrease (the END of the mass ramp).
            # On a grey 4-phase field (w ~ 0.25)
            # the sigmoid indicator floor is H = sigmoid(beta_if*(0.25-eta))
            # ~ 0.45 at beta_if=4: the penalty is NOT interface-selective and
            # acts as a GLOBAL solid-phase suppressor at up to w_interface ~ 5e5 --
            # observed to fragment the layout mid-ramp and trigger a
            # sink-driven T runaway (T_min drifting linearly to < 0 K from
            # the epoch the penalty engaged).  Engage it only once the field
            # has binarised; MP['interface_start_ep'] can move the start
            # earlier if desired.
            warmup_if = MP.get('interface_start_ep', None)
            warmup_if = frac_decrease if warmup_if is None else warmup_if
            ramp_if = MP.get('interface_ramp_ep', 2000)
            w_dead = MP.get('w_dead', 0.0)
            if epoch < warmup_if:
                w_interface_eff = 0.0
                w_dead_eff = 0.0
            else:
                t_if = min(1.0, (epoch - warmup_if) / max(1, ramp_if))
                w_interface_eff = w_interface * t_if
                w_dead_eff = w_dead * t_if
                _b0 = MP.get('beta_interface0', 4.0)
                MP['beta_interface'] = _b0 + (
                    MP.get('beta_interface_max', 16.0) - _b0) * t_if
            loss = (loss + w_interface_eff * interface_penalty_global
                    + w_dead_eff * dead_penalty_global)

            loss.backward(retain_graph=(adjoint_mode != 'direct'))

            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(model_list[0].parameters(), nrmThreshold)
                torch.nn.utils.clip_grad_norm_(model_list[1].parameters(), nrmThreshold)
                torch.nn.utils.clip_grad_norm_(model_list[2].parameters(), nrmThreshold)
                torch.nn.utils.clip_grad_norm_(model_list[3].parameters(), nrmThreshold)
                torch.nn.utils.clip_grad_norm_(model_list[4].parameters(), nrmThreshold)

            optimizer_disp.step()
            if adjoint_mode != 'direct':
                optimizer_disp_adj.step()
            optimizer_T.step()
            optimizer_T_adj.step()
            optimizer_rho.step()

            # ---- direct adjoint: refactorize + solve at the fresh state ------
            # (uses this epoch's stash; lambda is consumed from the NEXT epoch
            #  on, a one-epoch lag that is negligible under Adam's step sizes.)
            # FIRST-VISIT RULE: with n_g mesh slices cycling and a modulo-K
            # cadence, gcd(K, n_g)=1 (e.g. 25 vs 51) makes each slice's solve
            # coincide with epoch%K==0 only ~every K*n_g epochs, so most slices
            # consume a stale (or zero) lambda for thousands of epochs and the
            # objective gradient starves early in training.
            # Fix: ALWAYS solve on a slice's first visit (no lambda banked yet),
            # then keep the epoch-modulo-K refresh cadence on top.
            #
            # PRIMAL WARM-UP: no adjoint is solved before MP['adjoint_start_ep'].
            # lambda stays zero for those epochs, so the objective's mechanical
            # sensitivity (mech_sens = lambda^T dR/drho) is identically zero and
            # the design is driven only by the primal DEM + mass constraint,
            # letting u / T converge before the adjoint (and thus rho) engages.
            # The first-visit rule then re-solves each slice on its first visit
            # AT OR AFTER adjoint_start_ep, so no slice consumes a stale zero.
            if adjoint_mode == 'direct' and epoch >= MP.get('adjoint_start_ep', 0):
                # one-time pool release at adjoint onset: the first-visit rule
                # triggers a burst of solves + transfer-cache builds right here,
                # on top of the run's standing peak
                if epoch == MP.get('adjoint_start_ep', 0):
                    torch.cuda.empty_cache()
                    print(f"[Epoch {epoch}] adjoint onset — GPU mem before first "
                          f"solve: alloc {torch.cuda.memory_allocated()/2**30:.2f} GiB, "
                          f"peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
                _mk = MP.get('cur_mesh_key', 'default')
                _first_visit = MP['lambda_nodal'].get(_mk) is None
                _ages = MP.setdefault('_lam_solve_epoch', {})
                _max_age = MP.get('adjoint_max_age', None)   # None -> no age cap
                _stale = (_max_age is not None
                          and (epoch - _ages.get(_mk, -10**9)) > _max_age)
                if _first_visit or _stale \
                        or (epoch % MP.get('adjoint_refactor_every', 25) == 0):
                    # no_grad: lambda is a CONSTANT by construction; without
                    # this, torch_sla's autograd.Function pins its saved ctx
                    # (val/row/col/x, ~10 MB) to every stored lambda -> up to
                    # ~0.5 GB across 51 slices for nothing.
                    with torch.no_grad():
                        lam_new, direct_res = solve_adjoint_direct(model_list, MP)
                    MP['lambda_nodal'][_mk] = lam_new
                    _ages[_mk] = epoch
                    if not math.isfinite(direct_res) or direct_res > 1e-6:
                        print(f"[Epoch {epoch}] WARNING: direct adjoint residual "
                              f"{direct_res:.3e} (expected ~1e-12)")
            timeHistory.setdefault('adj_direct_res', []).append(
                direct_res if adjoint_mode == 'direct' else float('nan'))
            timeHistory.setdefault('adj_recip', []).append(
                MP.get('_adj_recip_ratio', float('nan'))
                if adjoint_mode == 'direct' else float('nan'))
            timeHistory.setdefault('beta_proj', []).append(MP.get('beta_proj') or 0.0)

            scheduler_disp.step()
            if adjoint_mode != 'direct':
                scheduler_disp_adj.step()
            scheduler_T.step()
            scheduler_T_adj.step()
            scheduler_rho.step()

            # save the loss histories
            timeHistory['loss_total'].append(loss.item())
            timeHistory['loss_obj'].append(loss_obj.item())
            timeHistory['loss_dem_disp'].append(loss_dem_disp.item())
            timeHistory['loss_dem_disp_adj'].append(loss_dem_disp_adj.item())
            timeHistory['loss_dem_t'].append(loss_dem_t.item())
            timeHistory['loss_dem_t_adj'].append(loss_dem_t_adj.item())
            timeHistory['loss_mConstraint'].append(loss_mConstraint.item())
            timeHistory['loss_cConstraint'].append(loss_cConstraint.item())
            timeHistory['massfrac'].append(massfrac.item())
            timeHistory['costfrac'].append(costfrac.item())
            timeHistory['strain_energy'].append(strain_energy_global.item())
            timeHistory['spring_energy'].append(spring_energy_global.item())
            timeHistory['heat_energy'].append(heat_energy_global.item())
            timeHistory['source_energy'].append(source_energy_global.item())
            timeHistory['external_work'].append(external_work_global)
            timeHistory['strain_energy_adj'].append(strain_energy_adj_global.item())
            timeHistory['spring_energy_adj'].append(spring_energy_adj_global.item())
            timeHistory['external_work_adj'].append(external_work_adj_global.item())
            timeHistory['heat_energy_adj'].append(heat_energy_adj_global.item())
            timeHistory['source_energy_adj'].append(source_energy_adj_global.item())
            timeHistory.setdefault('conv_energy_adj', []).append(
                conv_energy_adj_global.item())
            _td = MP.get('_tdep_diag', None)
            if _td is not None:
                timeHistory.setdefault('tdep_T_min', []).append(_td['T_min'])
                timeHistory.setdefault('tdep_T_max', []).append(_td['T_max'])
                timeHistory.setdefault('tdep_clamp_frac', []).append(_td['clamp_frac'])
                timeHistory.setdefault('tdep_clamp_frac_solid', []).append(
                    _td.get('clamp_frac_solid', float('nan')))
                timeHistory.setdefault('tdep_kappa_min', []).append(_td.get('kappa_min', float('nan')))
                # warn on the SOLID-masked fraction (the raw one is dominated
                # by benign void-region drift; it is still logged above)
                _cfs = _td.get('clamp_frac_solid', _td['clamp_frac'])
                if _cfs > 0.01 and epoch % 200 == 0:
                    print(f"[Epoch {epoch}] tdep WARNING: T at int points "
                          f"[{_td['T_min']:.0f}, {_td['T_max']:.0f}] K; "
                          f"{100*_cfs:.1f}% of SOLID points outside the "
                          f"property-fit window (clamped; all-point "
                          f"{100*_td['clamp_frac']:.1f}%); "
                          f"kappa_min={_td.get('kappa_min', float('nan')):.2e}, "
                          f"E_min={_td.get('E_min', float('nan')):.2e}, "
                          f"w_if_eff={w_interface_eff:.2e}")
            timeHistory['wc'].append(wc)
            timeHistory['wd'].append(wd)
            timeHistory['wm'].append(wm)
            timeHistory['wp'].append(wp)
            timeHistory['wtemp'].append(wtemp)
            timeHistory['interface_penalty'].append(interface_penalty_global.item()
                if torch.is_tensor(interface_penalty_global) else float(interface_penalty_global))
            timeHistory['w_interface'].append(w_interface_eff)
            timeHistory.setdefault('dead_penalty', []).append(
                dead_penalty_global.item())
            timeHistory.setdefault('w_dead', []).append(w_dead_eff)
            timeHistory['grey'].append(grey_fraction.detach().cpu().tolist())
            timeHistory['p'].append(model_list[0].MP['p'])
            # cumulative GPU peak (GiB) — memory forensics for OOM hunting
            timeHistory.setdefault('gpu_peak_gb', []).append(
                torch.cuda.max_memory_allocated() / 2**30
                if torch.cuda.is_available() else 0.0)

            # ---- stability monitor / best-checkpoint ------------------------
            Hmax_val = loss_dict['Hmax'].item()
            sea_val  = strain_energy_adj_global.item()   # 1/2 lambda^T K lambda (FULL tangent)
            uout_val = loss_dict['u_out'].item()
            loss_val = loss.item()
            timeHistory['Hmax'].append(Hmax_val)
            timeHistory.setdefault('u_out', []).append(uout_val)   # signed output stroke (DEM estimate; carries the known ~20% Adam soft-mode low-bias)
            timeHistory['adj_res'].append(adj_res_norm)  # ||d Pi_adj/d theta_l|| (residual solver)

            # ---- adjoint-quality monitor: virtual-work identity --------------
            # Exact adjoint K_T l = f_adj  =>  l^T K_T l = f_adj . l, i.e.
            #   vw_ratio = EW_adj / (2*(SE_adj + spring_adj))  ==  1.
            # A wrong or unconverged adjoint shows up here immediately (the
            # ratio goes negative or drifts far from 1).  Log it and require it
            # to be near 1 (and EW_adj > 0 while K is SPD) before trusting rho
            # gradients or banking a "best" design.
            _den = 2.0 * (sea_val + spring_energy_adj_global.item())
            vw_ratio = external_work_adj_global.item() / _den if abs(_den) > 1e-30 else float('nan')
            timeHistory.setdefault('adj_vw_ratio', []).append(vw_ratio)
            adj_ok = math.isfinite(vw_ratio) and (abs(vw_ratio - 1.0) <= vw_tol)

            finite  = math.isfinite(loss_val) and math.isfinite(Hmax_val)
            # In 'residual' mode the FULL tangent is used, so 1/2 l^T K l may be
            # negative for the exact adjoint near a limit point — that is NOT
            # ill-health.  Health = finite + bounded deformation + a SOLVED
            # adjoint (virtual-work ratio near 1).  In 'energy'+GN mode K is
            # SPD, so the sign of sea is additionally a health signal.
            healthy = finite and (Hmax_val <= H_cap) and adj_ok
            if adjoint_mode != 'direct' and adjoint_solver == 'energy':
                # GN energy mode: K SPD, so sea's sign is a health signal.  In
                # direct mode the FULL tangent is used and sea < 0 is legitimate
                # near a limit point (vw_ratio + Hmax already gate health).
                healthy = healthy and math.isfinite(sea_val) and (sea_val >= -sea_tol)

            # bank the best-performing STABLE design (max output stroke) once the
            # mass schedule has converged; this is what gets rolled back to at
            # the end.  Banking only — training is never stopped early.
            if healthy and (epoch >= warmup_ep) and (uout_val > best_uout):
                best_uout  = uout_val
                best_epoch = epoch
                best_state = _snapshot_cpu(model_list)

            # --- optional tangent diagnostic (indefiniteness / primal residual) --
            if MP.get('diagnose_tangent', False) and ((epoch + 1) % plotting_interval_TO == 0):
                try:
                    diagnose_tangent(model_list, epoch, n_lanczos=MP.get('n_lanczos', 24))
                except Exception as _e:
                    print(f"[Epoch {epoch}] tangent diagnostic skipped ({_e})")

            # --- checkpoint ---
            if  (epoch+1) % plotting_interval_TO == 0 or epoch == 1:
                if torch.cuda.is_available():
                    print(f"[Epoch {epoch+1}] GPU mem: "
                          f"alloc {torch.cuda.memory_allocated()/2**30:.2f} GiB, "
                          f"peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB, "
                          f"reserved {torch.cuda.memory_reserved()/2**30:.2f} GiB")
                end_time = time.time()
                torch.save(
                    {
                        'model_u_state_dict': model_list[0].mean_module_NN_All.state_dict(),
                        'model_u_adj_state_dict': model_list[1].mean_module_NN_All.state_dict(),
                        'model_T_state_dict': model_list[2].mean_module_NN_All.state_dict(),
                        'model_T_adj_state_dict': model_list[3].mean_module_NN_All.state_dict(),
                        'model_rho_state_dict': model_list[4].mean_module_NN_All.state_dict(),
                    },
                     save_folder + f'Trained_models_{epoch}.pth'
                    )
                total_time = total_time + (end_time - start_time)
                start_time = time.time()

            # --- GPU cooldown: pause 90s every 2000 epochs to prevent thermal throttling ---
            if (epoch + 1) % 2500 == 0 and (epoch + 1) < TO_num_iter:
                # stop the timer so cooldown doesn't count toward training time
                end_time = time.time()
                total_time += (end_time - start_time)
                torch.cuda.synchronize()
                print(f"[Epoch {epoch+1}] Cooling down for 90s ...")
                torch.cuda.empty_cache()   # defragment the pool while idle
                time.sleep(90)
                start_time = time.time()

            model_list[0].MP['massfrac_star'] = max(model_list[0].MP['massfrac_f'], model_list[0].MP['massfrac_star'] - model_list[0].MP['frac_step_mass'])
            model_list[0].MP['costfrac_star'] = max(model_list[0].MP['costfrac_f'], model_list[0].MP['costfrac_star'] - model_list[0].MP['frac_step_cost'])
            model_list[0].MP['p'] = min(model_list[0].MP['pf'], model_list[0].MP['p'] + model_list[0].MP['frac_step_p'])

        end_time = time.time()
        total_time += (end_time - start_time)
        print(f"Run {i} finished in {total_time:.2f}s")

        # -----------------------------
        # Roll back to the best stable design (adjoint-healthy, max stroke) so
        # all downstream fields / DXF reflect it rather than a diverged tail.
        # -----------------------------
        if best_state is not None:
            torch.save(best_state, save_folder + 'Trained_models_best.pth')
            model_list[0].mean_module_NN_All.load_state_dict(best_state['model_u_state_dict'])
            model_list[1].mean_module_NN_All.load_state_dict(best_state['model_u_adj_state_dict'])
            model_list[2].mean_module_NN_All.load_state_dict(best_state['model_T_state_dict'])
            model_list[3].mean_module_NN_All.load_state_dict(best_state['model_T_adj_state_dict'])
            model_list[4].mean_module_NN_All.load_state_dict(best_state['model_rho_state_dict'])
            print(f"Restored best stable design from epoch {best_epoch} "
                  f"(u_out={best_uout:.4g}); wrote Trained_models_best.pth")
        else:
            print("No stable post-warmup design was banked; keeping final-epoch weights.")

        # -----------------------------
        # Save metadata and results
        # -----------------------------
        # NOTE: helmholtz_filter is a nn.Module — exclude from MP save to avoid issues
        MP_save = {k: v for k, v in MP.items()
                   if k not in ('helmholtz_filter', 'cur_mesh_key',
                                'adjoint_fixed_dofs_fn', '_adj_coo_idx',
                                '_adj_stash', 'lambda_nodal',
                                '_lam_solve_epoch', '_gcov_cache',
                                '_linelas_Kunit', '_tdep_diag')}
        torch.save(MP_save, save_folder + "MP.pt")
        with open(save_folder + "NN_config_disp.json", "w") as file:
            json.dump(NN_config_disp, file, indent=4)
        with open(save_folder + "NN_config_rho.json", "w") as file:
            json.dump(NN_config_rho, file, indent=4)

        torch.save(train_data, save_folder + "train_data.pth")

        with open(save_folder + "timeHistory.json", "w") as file:
            json.dump(timeHistory, file)

        result_file_path = f"{save_folder}Results_summary.txt"
        with open(result_file_path, 'w') as f:
            f.write(f"The total training time in second is: {total_time}\n")
            f.write(f"Final value strain energy: {timeHistory['strain_energy'][-1]}\n")
            f.write(f"Final value external work: {timeHistory['external_work'][-1]}\n")


# ========================
# Launch block (single GPU)
# ========================
if __name__ == "__main__":
    ############################### Define Parameters ##############################################
    base_folder = os.path.dirname(os.path.abspath(__file__)) + '/'
    # repo root: Data/ (mesh + BC data) and Results/ (outputs) live here
    random_state = [1, 5, 7, 15, 17]   # the five independent initializations
                                       # reported in the paper (Sec. 3)
    num_CP = 51
    Example = 'EX2D2'
    Case = 'actuator'
    Diff_type = 'CPS4_reduced'
    design_flag = False
    tau = 0.5
    wt = [4.0]           # fine reduced 1-pt rule, consumed by get_data ONLY;
                         # overridden to 2x2 ones(4) after the unigrid build
    T_inf = 293
    T0 = 473
    TD = 873             # design temperature: surface Dirichlet BC (K).
                         # The paper runs TD in {673, 873, 1073}; all three
                         # lie inside the property-fit window (293-1100 K).
    hs = 2.0e-8
    hv = 1.33e-9
    # ---- MULTI-MATERIAL (TiCuFe): phase order = [void, Ti, Cu, Fe/steel] ----
    # Ti REPLACES Al so TD = 1073 K is physically meaningful: Al melts at
    # 933 K (it capped the old fit window at 900 K), while Ti stays
    # alpha-phase all the way to its 1155 K beta-transus (melting 1941 K).
    # Bases are the 293 K property values (multiplicative tdep corrections
    # below).  Ti bases from the pure-alpha-Ti anchors: kappa 21.9 W/mK,
    # alpha_inst 8.6e-6/K, E 115 GPa.  kappa_Ti = 2.19e-5 matches the TPRC
    # 293 K anchor that the temperature-correction fits below are normalised
    # to (property sources and fits: Appendix B of the paper).
    s = [0.0, -4.5e-8, -4.5e-8, -4.5e-8]   # body heat source per phase
    kappa = [1.0e-8, 2.19e-5, 40e-5, 6e-5] # W/(K um): Ti 21.9, Cu 400, steel 60 W/mK
    alpha = [1.2e-5, 0.86e-5, 1.7e-5, 1.2e-5]  # CTE: Ti 8.6, Cu 17, Fe 12 e-6/K
    deltaT = 100
    D = np.array([0, 4.506, 8.96, 7.8])    # density g/cm^3: Ti, Cu, Fe
    D = D / D.max()                        # normalize with the maximum value
    E = [1e-5, 0.115, 0.128, 0.2]          # modulus GPa/1000: Ti 115, Cu 128, Fe 200
    P = [0, 3.0, 2.0, 1.0]                 # cost (Ti the most expensive phase)
    K_out = 2e-3
    f_out = 0.1
    a_T = 1.0
    nu = 0.31
    p0 = 1
    pf = 3
    massfrac_f = 0.25
    costfrac_f = 0.6
    mass_constraint = 'inequality'  # 'equality' (default): two-sided quadratic on
                                  #   massfrac for the WHOLE run — the physical
                                  #   mass tracks the ramped target exactly.
                                  # 'inequality': one-sided (over-target only)
                                  #   after the mass ramp — legacy form.
    cost_constraint = 'equality'  # same switch for costfrac (enters the loss
                                  #   only if use_cost_constraint below).
    use_cost_constraint = False  # multi-material cost constraint wp*loss_cC in
                                 #   the loss (off in the paper runs; costfrac
                                 #   is still logged).  Set True to activate
                                 #   the P = [0, 3, 2, 1] pricing.
    # ---- interface incompatibility penalty (Ti-Steel) ----
    w_interface = 5e6          # penalty weight (volume-normalised penalty, so
                               #   O(1e5-1e6); 0 disables the term entirely)
    beta_interface0 = 4.0      # Heaviside sharpness at ramp start
    beta_interface_max = 16.0  # ... and at full continuation
    eta_interface = 0.30       # presence threshold: w_i below ~ "absent"
    interface_pairs = [(1, 3)] # incompatible phase pairs (1=Ti, 3=Fe:
                               #   TiFe/TiFe2 brittle intermetallics); add
                               #   tuples for more exclusions, [] disables
    w_dead = 0                 # ISLAND KILLER for equality-mass runs: penalises
                               #   solid fraction in regions of ~zero elastic
                               #   strain-energy density (dead material; see
                               #   the P_dead block in calculate_TO_loss).
                               #   0 disables.  With mass_constraint='equality'
                               #   and Cu filler appearing, start at 1e4-1e5
                               #   (volume-normalised; P_dead in [0,1]) and
                               #   ramp with the interface schedule.  Prefer
                               #   FIRST calibrating massfrac_f to the natural
                               #   mass read off an inequality run's history.
    dead_se_frac = 0.05        # "dead" threshold: SE density below this
                               #   fraction of the domain mean scores psi ~ 1
    frac_decrease = 0.5
    b = 8
    thres_static = 0.5
    rho_min = 0.1
    rho_max = 0.9

    # define domain
    thickness = 15.0
    Nelx = 500
    Nely = 250
    xmin, xmax = 0.0, 500.0
    ymin, ymax = 0.0, 250.0

    # =====================================================================
    # SINGLE UNIFORM GRID -- primal == adjoint mesh (no transfers anywhere)
    # =====================================================================
    unigrid = (200, 100) # (Ngx, Ngy): ONE uniform quad mesh over the domain
                         #   box carrying EVERYTHING -- primal DEM (u, T),
                         #   design field, fused sensitivity, direct adjoint --
                         #   with FULL 2x2 quadrature (MP['wt'] -> ones(4)
                         #   after get_data).  200x100 -> 20,000 elements,
                         #   20,301 nodes / 40,602 displacement dofs; square
                         #   hx = hy = 2.5 um cells.
                         #   Nelx x Nely above stays ONLY so get_data can
                         #   source the geometry mask and load/spring points;
                         #   the fine slices are freed after the build.

    # =====================================================================
    # Helmholtz filter parameters
    # =====================================================================
    r_min = 5.0          # Helmholtz filter radius (um); minimum feature
                         # size ~ 2*r_min*sqrt(3) ~ 17.3 um.  The unigrid
                         # cell is 2.5 um at 200x100, so r_min spans 2 cells:
                         # well resolved.
                         # The interface penalty REQUIRES the filter (it
                         # reuses the filter's N2E/E2N neighbour projections
                         # and needs smooth fields for the sigmoid indicator).
    filter_cg_tol = 1e-4
    filter_cg_max_iter = 50

    # define model parameters and training parameters
    init_method = 'kaiming_uniform_'
    dynamic_weight = False
    gradient_clip = True
    omega = 0.25
    learning_rate_disp = 5e-3
    learning_rate_disp_adj = 1e-3
    learning_rate_T = 1e-3
    learning_rate_T_adj = 1e-3
    learning_rate_rho = 5e-5
    nrmThreshold = 5.0
    TO_num_iter = 10000
    plot_num = 10
    delta = 1e-1
    wc = 1e3
    wd = 1e3
    wm = 1e5
    wp = 1e5
    wtemp = 1e3
    # =====================================================================
    # Geometric nonlinearity (finite-strain St. Venant–Kirchhoff)
    # =====================================================================
    # ------------------------------------------------------------------
    # PHYSICS MODEL SELECTION (paper Secs. 2.2-2.3)
    #   baseline      : constitutive = 'linelas', prop_eval = 'fixed_TD'
    #                   (small-strain linear kinematics, properties
    #                    anchored at T = TD)
    #   high-fidelity : constitutive = 'hencky',  prop_eval = 'tdep'
    #                   (quadratic-Hencky finite-strain kinematics,
    #                    properties at the local field temperature T(x))
    # ------------------------------------------------------------------
    constitutive = 'hencky'    # 'hencky' -> finite-strain quadratic-Hencky with
                               #   the three-term Wang interpolation (full-
                               #   tangent direct adjoint, geometric term
                               #   retained).
                               # 'linelas' -> small-strain plane stress: the
                               #   Wang form collapses to psi_L identically
                               #   (gamma inert), the tangent has NO geometric
                               #   part, and the direct adjoint reduces to the
                               #   standard linear thermo-mechanical system
                               #   K lambda = f_adj (FD-verified; expect
                               #   adj_direct_res ~1e-12 and vw_ratio == 1).
    energy_interp = (constitutive != 'linelas')
                               # three-term Wang interpolation gamma(rho):
                               # INERT under 'linelas' (the Wang form collapses
                               # to psi_L; gamma never enters) and REQUIRED
                               # under 'hencky' (void regions must see the
                               # small-strain law or the layout goes grey /
                               # ill-conditioned).
    gamma_beta = 100.0         # Heaviside sharpness for γ(ρ_solid) — near-binary
    gamma_eta  = 0.10          # solid-fraction threshold for γ
    # ----- density projection (beta continuation) ----------------------------
    use_projection    = True   # smooth Heaviside on the filtered solid fraction.
                               # NOTE: the projection path is TWO-PHASE only
                               # (guarded by num_phase == 2 everywhere); with
                               # a 4-phase material set it is INERT (beta_proj
                               # = None, mass on the filtered raw weights) —
                               # exactly the original multi-material behavior.
    beta0             = 2.0    # initial sharpness from epoch 0
    beta_start_ep     = 2000   # ramp start; None -> frac_decrease (end of mass ramp)
    beta_double_every = 1000   # doubling cadence after ramp start
    beta_max          = 10.0   # final sharpness (2 -> 4 -> 8 -> 16)
    eta_proj          = 0.5    # projection threshold
    # ----- adjoint solve ------------------------------------------------------
    adjoint_mode = 'direct'    # 'direct' (default) -> EXACT full-tangent adjoint:
                               #   assemble K_T from batched per-element Hessians
                               #   (vmap of the three-term energy; verified against
                               #   the nested-autograd HVP to machine precision)
                               #   and solve K_T λ = f_adj with torch-sla
                               #   (cuDSS LDL^T on GPU, indefinite-safe; SuperLU
                               #   fallback on CPU).  Geometric term retained.
                               #   Removes the adjoint net + its optimizer + its
                               #   loss term from the loop entirely.
                               # 'dem' -> legacy in-loop DEM adjoint; then
                               #   adjoint_solver/adjoint_geometric below apply.
    adjoint_start_ep = 2000    # PRIMAL WARM-UP: do not solve the adjoint (lambda
                               #   stays 0) until this epoch, so u/T converge
                               #   first.  While lambda=0 the mechanical design
                               #   sensitivity mech_sens = lambda^T dR/drho is
                               #   identically 0, so rho is driven only by the
                               #   primal DEM + mass constraint.  At this epoch
                               #   the first-visit rule re-solves every slice, so
                               #   none consume a stale zero.  Set 0 to disable.
    adjoint_refactor_every = 25  # epochs between refactor+solve; the design and
                               #   state drift little per Adam step, so a lagged
                               #   λ is accurate; lower to 1 for exact-per-epoch.
                               # NOTE (sliced meshes): a slice ALWAYS gets a solve
                               #   on its FIRST visit (avoids the gcd(25, n_g)=1
                               #   staleness that starves objective gradients
                               #   when slices cycle); afterwards refreshes
                               #   land on whichever slice is current at
                               #   epoch%25==0.
    adjoint_max_age = None     # per-slice staleness cap -- redundant on the
                               #   single unigrid mesh: the modulo cadence alone
                               #   bounds lambda's age at adjoint_refactor_every
                               #   (the gcd staleness pathology needed >1 slice).
    adjoint_direct_method = 'ldlt'  # cuDSS factorization: 'ldlt' (symmetric
                               #   indefinite) or 'lu'.  CPU path ignores this.
    adjoint_hess_chunk = 4096  # elements per vmap(hessian) batch in the K_T
                               #   assembly.  Chunking the ASSEMBLY is exact
                               #   (element tangents are independent) and bounds
                               #   the forward-over-reverse autodiff transient
                               #   at O(chunk) instead of O(N_elem) — the term
                               #   that actually scales when the grid
                               #   grows (the 26k-dof LDL^T factor is only
                               #   ~10-30 MB; 10,800/2048 = 6 chunks at 180x60).
                               #   None -> single batch (legacy).
    adjoint_solve_device = 'gpu'  # 'gpu' -> cuDSS LDL^T on device (default);
                               #   'cpu' -> scipy SuperLU on the host: removes
                               #   the factorization + workspace from VRAM
                               #   entirely.  ~0.1-0.3 s per solve at 26k dofs,
                               #   amortized over the refresh cadence.  Use if
                               #   the memory prints show the cuDSS transient
                               #   riding the ceiling at adjoint onset.
    adjoint_solver = 'energy'  # consulted only when adjoint_mode='dem'.
                               # 'energy' + adjoint_geometric=False (Gauss-
                               #   Newton, SPD, bounded-below DEM min) is a
                               #   CONVERGENT approximate adjoint solver.
                               # 'residual' (parameter-space KKT norm) squares
                               #   the tangent's condition number and converges
                               #   far too slowly to be used in-loop; kept for
                               #   offline refinement at frozen states only.
    w_adj_residual = None      # weight on the KKT-residual term; None -> reuse wd
    adjoint_geometric = False  # Consulted ONLY when adjoint_solver=='energy':
                               #   False -> Gauss-Newton material-only tangent:
                               #            SPD (bounded energy), APPROXIMATE
                               #            adjoint (drops the geometric term; the
                               #            geometric term adds no rho-term to the
                               #            sensitivity, so the only bias is via
                               #            lambda -- modest at Hmax ~ 0.2).
                               #   True  -> full Newton tangent (consistent, can
                               #            run away near a limit point).
    diagnose_tangent_flag = False   # at each checkpoint, Lanczos-estimate lambda_min/max
                               #   of the param-space adjoint Hessian + primal
                               #   residual (is the tangent really indefinite at a
                               #   converged primal, or off-equilibrium?)
    n_lanczos = 24
    Hmax_cap = 1.0             # finite-strain validity cap on max|grad u| for banking "best"
    # ----- temperature-dependent thermal properties ---------------------------
    # kappa_i(T) = kappa[i] * poly(kappa_T_coef[i], T - prop_T_ref)   (SIMP per phase)
    # alpha_i(T) = alpha[i] * poly(alpha_T_coef[i], T - prop_T_ref)   (INSTANTANEOUS CTE)
    # Eigenstrain uses the exact integral  eps_th = int_{T_inf}^{T} alpha(tau) dtau ;
    # the T-adjoint gains (i) alpha_inst(T*) in the coupling source and (ii) the
    # kappa'(T) tangent-transpose term.  coef=[1.0] on both phases reproduces the
    # constant-property code bit-for-bit in value (only shapes broadcast wider);
    # tdep_props_on=False removes the code path entirely.
    tdep_props_on = True       # keep True for BOTH physics models: 'tdep'
                               # evaluates the fits at the local T(x), while
                               # 'fixed_TD' consumes them once to anchor
                               # kappa/E at TD and build the secant CTE.
    prop_eval = 'tdep'     # 'tdep' -> properties at the LOCAL field T(x)
                               #   (full tdep: spatial variation + the dk/dT
                               #   and dE/dT adjoint pathways).
                               # 'fixed_TD' -> the COUNTERPART model: kappa/E
                               #   frozen at T = TD, alpha = per-phase SECANT
                               #   abar(TD) with eps_th = abar*(T - T_inf);
                               #   all d(prop)/dT adjoint terms vanish and
                               #   the T-adjoint collapses to the classic
                               #   constant-property structure.  Agrees with
                               #   'tdep' exactly wherever T(x) = TD, so the
                               #   pair isolates spatial-property-variation
                               #   + adjoint-pathway effects.
    prop_T_ref = T_inf         # reference temperature of the polynomial fits (K)
    # MULTI-MATERIAL literature fits [void, Ti, Cu, Fe/steel], constrained
    # least squares over 293-1100 K anchors (nine points, 293/400/.../1100),
    # f(T_ref)=1 exact (multiplicative form => the bases above are the 293 K
    # values).  SOURCE / VERIFICATION LEDGER:
    #   kappa_Ti : TPRC recommended values (Ho, Powell & Liley, J. Phys.
    #     Chem. Ref. Data 1 (1972); digitized reproduction at efunda.com):
    #     300:21.9, 400:20.4, 500:19.7, 600:19.4, 800:19.7, 1000:20.7,
    #     1200:22.0 W/mK.  Base 2.19e-5 = TPRC 293 K.
    #   kappa_Cu : Ho-Powell-Liley recommended series (401...366 at
    #     300-800 K, near-linear); 900-1100 continues the same trend.
    #   kappa_Fe : Incropera & DeWitt, Fundamentals of Heat & Mass
    #     Transfer, Table A.1 plain carbon steel (60.5/56.7/48.0/39.2/31.3
    #     at 300/400/600/800/1000 K); other points interpolated.
    #   alpha    : cross-checked via handbook mean CTEs integrated from the
    #     fits: Cu +0.2%/+0.5% (293-373/573 K); Ti +2.5% vs 9.7e-6
    #     (293-1088 K CP-Ti mean); Fe -5% vs ~14.7e-6 (293-873 K carbon-
    #     steel mean).  The Fe deviation sits INSIDE the documented spread:
    #     published Fe expansion datasets disagree by 10-15% above 600 K
    #     (Fe LTEC dilatometry, 130-1180 K) -- an irreducible few-% band
    #     that eps_th_Fe inherits (noted in the paper).  Primary tables:
    #     Touloukian et al., TPRC v12 (1975).
    #   E_Ti     : Fisher & Renken, Phys. Rev. 135 (1964) A482 (alpha-Ti
    #     single-crystal constants 4-1156 K, polycrystal aggregate); slope
    #     corroborated by published Ti-6Al-4V impulse-excitation data
    #     (118 -> 72 GPa over 293-1173 K, near-linear).
    #   E_Cu     : Chang & Himmel, J. Appl. Phys. 37 (1966) 3567: Cu
    #     elastic constants decrease LINEARLY 300-800 K, with Ag evidence
    #     of linearity to ~0.8 T_m; above 800 K the measured series is
    #     continued at its terminal slope (-0.07 GPa/K).
    #   E_Fe     : engineering carbon-steel series to 800 K; accelerating
    #     decline toward T_C = 1043 K per the magnetoelastic softening of
    #     alpha-Fe (Dever, J. Appl. Phys. 43 (1972) 3293).
    #
    #   Ti (pure alpha-Ti; beta-transus 1155 K > window top, melt 1941 K):
    #     kappa: 21.9/20.4/19.7/19.4/19.55/19.7/20.2/20.7/21.35 W/mK (TPRC;
    #       700/900/1100 interpolated) -- NON-monotone (minimum ~600 K,
    #       gentle electronic rise); quartic, |err| 0.31%; factor 0.888 at
    #       673 K, 0.966 at 1073 K.
    #     alpha_inst: 8.6/9.0/9.4/9.7/10.0/10.3/10.6/10.9/11.1 e-6/K
    #       (quadratic, |err| 0.4%; factor 1.155 at 673 K, 1.285 at 1073 K).
    #     E: 115/108/102/96/89/82/75/67/58 GPa -- strong softening toward the
    #       transus (cubic, |err| 0.4%; factor 0.791 at 673 K, 0.526 at 1073).
    #
    #   Cu (OFHC; melts 1358 K):
    #     kappa: 401/393/386/379/373/366/359/352/346 W/mK (linear,
    #       |err| 0.2%; factor 0.935 at 673 K).
    #     alpha_inst: 16.7/17.6/18.3/18.9/19.6/20.3/21.1/22.0/23.0 e-6/K
    #       (cubic, |err| 0.2%; factor 1.163 at 673 K).
    #     E: 128/124/119/114/108/101/94/87/80 GPa -- measured series to
    #       800 K, terminal-slope linear continuation above (Chang-Himmel);
    #       cubic, |err| 0.28%; factor 0.855 at 673 K, 0.639 at 1073 K.
    #
    #   Fe (plain carbon steel; Curie 1043 K is INSIDE the window -- the
    #   smooth fits average through the kappa flattening / alpha cusp there;
    #   noted in the paper):
    #     kappa: 60.5/56.7/52.6/48.0/43.6/39.2/35.2/31.3/28.5 W/mK (cubic,
    #       |err| 0.4%; factor 0.741 at 673 K -- the strongest kappa(T)
    #       drop of the set: 0.482 at 1073 K).
    #     alpha_inst: 11.8/12.7/13.4/13.9/14.5/15.0/15.5/15.9/15.6 e-6/K
    #       (quartic through the Curie rollover, |err| 0.4%; factor 1.213
    #       at 673 K).
    #     E: 200/193/185/176/166/154/141/126/108 GPa (cubic, |err| 0.3%;
    #       factor 0.844 at 673 K; 0.566 at 1073 K).
    kappa_T_coef = [[1.0],                                          # void: constant
                    [1.0, -8.32894e-4,  2.02144e-6, -1.87281e-9,  7.42533e-13],  # Ti (quartic; TPRC-verified)
                    [1.0, -1.72205e-4],                             # Cu (linear)
                    [1.0, -5.38540e-4, -5.82423e-7,  5.41279e-10]]  # Fe (cubic)
    alpha_T_coef = [[1.0],                                          # void: constant
                    [1.0,  4.50136e-4, -1.09250e-7],                # Ti
                    [1.0,  5.18040e-4, -3.84006e-7,  4.00361e-10],  # Cu
                    [1.0,  9.45641e-4, -2.26320e-6,  4.44551e-9, -3.06905e-12]]  # Fe (quartic)
    E_T_coef     = [[1.0],                                          # void: constant
                    [1.0, -5.60051e-4,  1.04476e-7, -2.11509e-10],  # Ti (cubic)
                    [1.0, -2.46040e-4, -4.35741e-7,  2.03154e-10],  # Cu (linear-extended per Chang-Himmel)
                    [1.0, -3.20222e-4, -1.71611e-7, -1.68928e-10]]  # Fe (cubic)
    prop_T_valid = (293.0, 1100.0)  # fit validity window (K): TD = 1073 is
                               # IN-window (Ti alpha-phase to 1155 K; Cu
                               # melts 1358 K; steel Curie 1043 K smoothed
                               # through).  Properties use constant
                               # extrapolation outside (clamped dT, zero
                               # d/dT).  All twelve factors stay positive on
                               # [293, 1100]; beyond it the unguarded E and
                               # kappa fits eventually go negative -- the
                               # same silent topology failure mode as the Ni
                               # cubic kappa, prevented by the clamp.
    basis = 'PGCAN'
    basis_rho = 'PGCAN'
    quant_correlation_class = 'Rough_RBF'
    activation = 'gelu'
    if basis == 'PGCAN':
        n_features = 64
        n_cells = 12
        res = [24, 48]
        n_neurons = int(n_features / 2)
        n_layers = 3
        NN_arch = [n_neurons] * n_layers
        kernel_size = (2, 2)
    else:
        n_features = None
        n_cells = None
        res = None
        n_neurons = 64
        n_layers = 6
        NN_arch = [n_neurons] * n_layers
        kernel_size = None

    ############################### Pre-processing ##############################################
    num_phase = len(D)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32              # float32 + TF32 for speed
    D_tensor = torch.tensor(D, dtype=dtype, device=device)
    E_tensor = torch.tensor(E, dtype=dtype, device=device)
    P_tensor = torch.tensor(P, dtype=dtype, device=device)
    alpha_tensor = torch.tensor(alpha, dtype=dtype, device=device)
    kappa_tensor = torch.tensor(kappa, dtype=dtype, device=device)
    s_tensor = torch.tensor(s, dtype=dtype, device=device)
    wt_tensor = torch.tensor(wt, dtype=dtype, device=device)
    domain = {'x': [xmin, xmax], 'y': [ymin, ymax], 'thickness':thickness}
    frac_decrease = int(frac_decrease * TO_num_iter)
    plotting_interval_TO = TO_num_iter // plot_num
    frac_step = None

    MP = {'num_CP': num_CP, 'Example': Example, 'tau':tau,
          'base_folder': base_folder, 'domain': domain, 'design_flag':design_flag,
          'thres_static': thres_static, 'Diff_type': Diff_type,
          'num_phase': num_phase, 'Nelx': Nelx, 'Nely': Nely,
          'T_inf':T_inf,'T0':T0,'TD':TD,'hs':hs,'hv':hv,'s':s_tensor,
          'D': D_tensor, 'E': E_tensor, 'P': P_tensor, 'wt':wt_tensor,'nu': nu,'K_out':K_out,'f_out':f_out,
          'p': p0, 'p0': p0, 'pf': pf, 'b': b,'alpha':alpha_tensor,'kappa':kappa_tensor,'deltaT':deltaT,
          'rho_min': rho_min, 'rho_max': rho_max,'a_T':a_T,
          'massfrac_star': massfrac_f, 'massfrac0': massfrac_f,
          'massfrac_f': massfrac_f, 'frac_decrease': frac_decrease,
          'costfrac_star': costfrac_f, 'costfrac0': costfrac_f,
          'costfrac_f': costfrac_f,
          'use_cost_constraint': use_cost_constraint,
          'beta_interface': beta_interface0,
          'beta_interface0': beta_interface0,
          'beta_interface_max': beta_interface_max,
          'eta_interface': eta_interface,
          'interface_pairs': interface_pairs,
          'w_dead': w_dead, 'dead_se_frac': dead_se_frac,
          'mass_constraint': mass_constraint,
          'cost_constraint': cost_constraint,
          'frac_step_mass': frac_step, 'frac_step_cost': frac_step,
          'frac_step_p': frac_step,
          'constitutive': constitutive,
          'energy_interp': energy_interp, 'gamma_beta': gamma_beta,
          'gamma_eta': gamma_eta, 'adjoint_geometric': adjoint_geometric,
          'Hmax_cap': Hmax_cap,
          'adjoint_solver': adjoint_solver, 'w_adj_residual': w_adj_residual,
          'adjoint_mode': adjoint_mode,
          'adjoint_start_ep': adjoint_start_ep,
          'adjoint_refactor_every': adjoint_refactor_every,
          'adjoint_max_age': adjoint_max_age,
          'adjoint_direct_method': adjoint_direct_method,
          'adjoint_hess_chunk': adjoint_hess_chunk,
          'adjoint_solve_device': adjoint_solve_device,
          'unigrid': unigrid,
          'use_projection': use_projection, 'beta0': beta0,
          'beta_double_every': beta_double_every, 'beta_max': beta_max,
          'eta_proj': eta_proj,
          **({'beta_start_ep': beta_start_ep} if beta_start_ep is not None else {}),
          'diagnose_tangent': diagnose_tangent_flag, 'n_lanczos': n_lanczos,
          'prop_eval': prop_eval,
          'tdep_props': ({'kappa': kappa_T_coef, 'alpha': alpha_T_coef,
                          'E': E_T_coef, 'T_valid': prop_T_valid,
                          'T_ref': prop_T_ref} if tdep_props_on else None)}

    # =====================================================================
    # Adjoint Dirichlet dofs — EXACT geometric BCs on every mesh slice
    # ---------------------------------------------------------------------
    # EX2D2: left edge (x=xmin) and top edge (y=ymax) CLAMPED (lambda_u =
    # lambda_v = 0); bottom edge (y=ymin) ROLLER (lambda_v = 0, lambda_u
    # free).  Right edge and any interior notch faces free.  The adjoint
    # satisfies the SAME homogeneous kinematic BCs as the primal (paper
    # Eq. 17a), so this replaces the snap-to-conditioning-points fallback
    # with the exact primal-constrained space on every slice resolution —
    # eliminating over/under-pinning as a failure mode (see the load-kill
    # guard and reciprocity check in solve_adjoint_direct).
    # =====================================================================
    def _ex2d2_adjoint_fixed_dofs(X):
        x, y = X[:, 0], X[:, 1]
        tolx = 1e-4 * (x.max() - x.min()).clamp_min(1e-30)
        toly = 1e-4 * (y.max() - y.min()).clamp_min(1e-30)
        idx = torch.arange(X.shape[0], device=X.device)
        clamped = idx[(x <= x.min() + tolx) | (y >= y.max() - toly)]  # left + top
        roller  = idx[y <= y.min() + toly]                            # bottom
        return torch.cat([2 * clamped, 2 * clamped + 1,
                          2 * roller + 1]).unique()

    # =====================================================================
    # Build Helmholtz filter and attach to MP
    # =====================================================================
    helmholtz_filter = build_helmholtz_filter(
        r_min=r_min,
        device=device,
        dtype=dtype,
        cg_tol=filter_cg_tol,
        cg_max_iter=filter_cg_max_iter,
    )
    MP['helmholtz_filter'] = helmholtz_filter   # ON — required by the
                                                # interface penalty
    MP['helmholtz_on'] = True   # persisted flag (the filter object is
                                # excluded from MP.pt); viz rebuilds the
                                # filter from r_min when this is True
    MP['r_min'] = r_min
    MP['adjoint_fixed_dofs_fn'] = _ex2d2_adjoint_fixed_dofs

    NN_config_disp = {'init_method': init_method, 'random_state': random_state, 'state': [],
                      'Example': Example, 'Case': Case,
                      'dynamic_weight': dynamic_weight, 'gradient_clip': gradient_clip,
                      'learning_rate_disp': learning_rate_disp,'learning_rate_disp_adj': learning_rate_disp_adj,
                      'learning_rate_rho': learning_rate_rho, 'learning_rate_T':learning_rate_T, 'learning_rate_T_adj':learning_rate_T_adj,
                      'nrmThreshold': nrmThreshold, 'omega': omega,
                      'TO_num_iter': TO_num_iter, 'plot_num': plot_num,
                      'plotting_interval_TO': plotting_interval_TO,
                      'delta': delta, 'wc': wc, 'wd': wd, 'wm': wm, 'wp': wp,'wtemp':wtemp, 'w_interface': w_interface,
                      'kernel_size': kernel_size,
                      'basis': basis, 'quant_correlation_class': quant_correlation_class,
                      'activation': activation,
                      'n_features': n_features, 'n_cells': n_cells,
                      'res': res, 'NN_arch': NN_arch, 'save_folder': []}
    NN_config_rho = copy.deepcopy(NN_config_disp)
    NN_config_rho['basis'] = basis_rho
    if basis_rho == 'PGCAN':
        NN_config_rho['n_features'] = 64
        NN_config_rho['n_cells'] = 12
        NN_config_rho['res'] = [24, 48]
        n_neurons = int(NN_config_rho['n_features'] / 2)
        n_layers = 3
        NN_config_rho['NN_arch'] = [n_neurons] * n_layers
        NN_config_rho['kernel_size'] = (2, 2)
    else:
        NN_config_rho['n_features'] = None
        NN_config_rho['n_cells'] = None
        NN_config_rho['res'] = None
        n_neurons = 64
        n_layers = 6
        NN_config_rho['NN_arch'] = [n_neurons] * n_layers

    ############################### Generate Data ##############################################
    Training, mesh_data = get_data(MP)
    u_X_train = Training['u_X_train']
    v_X_train = Training['v_X_train']
    T_X_train = Training['T_X_train']
    u_train = Training['u_train']
    v_train = Training['v_train']
    T_train = Training['T_train']
    vd1_X_train = Training['vd1_X_train']
    vd1_train = Training['vd1_train']
    vd2_X_train = Training['vd2_X_train']
    vd2_train = Training['vd2_train']
    vt_X_train = Training['vt_X_train']
    vt_train = Training['vt_train']

    train_data = {
                'disp': {
                    'X': [u_X_train, v_X_train],
                    'y': [u_train, v_train],
                },
                'disp_adj': {
                    'X': [vd1_X_train, vd2_X_train],
                    'y': [vd1_train, vd2_train],
                },
                'T': {
                    'X': [T_X_train],
                    'y': [T_train],
                },
                'T_adj': {
                    'X': [vt_X_train],
                    'y': [vt_train],
                },
                'phase': {
                    'X': [Training[f'phase{i}_X_train'] for i in range(num_phase)],
                    'y': [Training[f'phase{i}_train']   for i in range(num_phase)],
                },
            }

    # --- collapse EVERYTHING onto the single uniform grid ---------------------
    mesh_data = _build_unigrid_mesh_data(MP, mesh_data, device, dtype)
    # FULL 2x2 quadrature on the unigrid (replaces the fine reduced 1-pt rule;
    # a 1-pt tangent on this grid would be hourglass-singular)
    MP['wt'] = torch.ones(4, dtype=dtype, device=device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()   # the fine slices are unreachable now -- free them

    index_CP = np.ones(TO_num_iter, dtype=int)   # single mesh: Mesh_01 every epoch

    try:
        run_worker(MP, NN_config_disp, NN_config_rho, train_data,
                   index_CP, mesh_data)
    except torch.cuda.OutOfMemoryError:
        # forensics: WHAT is resident when the pool overflows.  The summary
        # separates active vs reserved vs fragmented; large "allocated" with
        # this traceback pre-epoch-adjoint_start_ep means the standing GP/DEM
        # load doesn't fit (check nvidia-smi for other processes); post-onset
        # points at the adjoint path.
        if torch.cuda.is_available():
            print(torch.cuda.memory_summary())
        raise

