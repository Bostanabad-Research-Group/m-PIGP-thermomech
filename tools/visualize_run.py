#!/usr/bin/env python3
"""Post-processing for finished optimization runs (actuator EX2D2 / gripper
EX2D1): figures, CSV exports, and the input folders for the evaluation
notebooks.

The training scripts write NO images during a run -- only model checkpoints
(Trained_models_<epoch>.pth every TO_num_iter/plot_num epochs, plus
Trained_models_best.pth), MP.pt, train_data.pth, NN_config_*.json and
timeHistory.json.  This tool loads a finished run folder, rebuilds the unigrid
mesh and the five LMGP models via the TRAINING module (imported by path -- no
physics is re-implemented here; the GP field extraction is copied
line-for-line from calculate_TO_loss), and produces for EVERY checkpoint:

  figures : material map (argmax), per-phase weight fields, deformed shape
            (true scale), temperature field; plus run-level history plots
            (u_out, mass/cost fraction, adjoint health when logged)
  CSVs    : topo_material_id.csv (categorical design), topo_rho_solid.csv,
            topo_w_<phase>.csv  -- COMSOL-style (x, y, value) at the element
            centers -- and a compressed fields_<Ngx>x<Ngy>.npz bundle

and, once per run (default: from the 'best' checkpoint):

  evaluator export : eval_export/Run_<seed>_<constitutive>_<TD>/
                         topo_material_id.csv
            at the TRAINING resolution, named exactly as the cross-evaluation
            notebooks (evaluation/evaluate_*.ipynb) expect.  Zip the
            eval_export folders of all runs you want evaluated and upload the
            zip to the notebook's data cell.

Fields are continuous GP/NN functions, so they can be re-queried on a grid
finer than the training one (--viz-grid) for sharper figures; the evaluator
export always uses the training grid regardless.

Usage
-----
python tools/visualize_run.py --run_folder Results/EX2D2/actuator/Run_7_<ts>
python tools/visualize_run.py --run_folder ... --viz-grid 500 250   # HD figs

The training script is auto-detected from MP['Example'] (EX2D1 ->
run_gripper.py, EX2D2 -> run_actuator.py, resolved relative to this file);
override with --train-script if you have renamed things.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib as mpl
matplotlib_backend = os.environ.get('MPLBACKEND', 'Agg')
mpl.use(matplotlib_backend)
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, to_rgb
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator

REPO_ROOT = Path(__file__).resolve().parent.parent
# The training scripts import their own packages as `models.*` / `utils.*`,
# so the repo root must be importable no matter where this tool is launched
# from (running `python tools/visualize_run.py` puts tools/ on sys.path,
# not the repo root).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# =====================================================================
# House style
# =====================================================================
FIGSIZE_HIST = (14.56, 4.0)
LINEWIDTH = 2.5
FONT_SIZE = 16
OFFSET_FONT_SIZE = 14
TICK_SIZE = 4
SUBPLOT_PARAMS_HIST = dict(left=0.07, right=0.99, top=0.97, bottom=0.18)

PHASE_NAMES = ['void', 'Ti', 'Cu', 'Steel']
PHASE_COLORS = ['#f2f2f2', '#8da4b8', '#b87333', '#4a5560']

TRAIN_SCRIPT_FOR_EXAMPLE = {'EX2D1': 'run_gripper.py', 'EX2D2': 'run_actuator.py'}


def configure_rcparams(font_size: int = FONT_SIZE) -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "font.size": font_size,
        "axes.linewidth": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": TICK_SIZE,
        "ytick.major.size": TICK_SIZE,
        "axes.formatter.use_mathtext": True,
    })


# =====================================================================
# Uniform-lattice raster backend (imshow instead of a PolyCollection of
# every element: faster, seam-free; falls back automatically when the
# element set is not a full-lattice bijection, e.g. notched geometry)
# =====================================================================
_GRID: dict = {}


def setup_raster_grid(MP, elem_x, Ng, force_poly=False):
    Ngx, Ngy = Ng
    (x0, x1), (y0, y1) = MP['domain']['x'], MP['domain']['y']
    hx, hy = (x1 - x0) / Ngx, (y1 - y0) / Ngy
    ex = elem_x[:, :2].detach().cpu().numpy()
    i = np.clip(((ex[:, 0] - x0) / hx).astype(np.int64), 0, Ngx - 1)
    j = np.clip(((ex[:, 1] - y0) / hy).astype(np.int64), 0, Ngy - 1)
    flat = j * Ngx + i
    ok = (not force_poly) and (np.unique(flat).size == flat.size)
    _GRID.clear()
    _GRID.update(idx=flat, Ngx=Ngx, Ngy=Ngy, n_cell=Ngx * Ngy,
                 extent=[x0, x1, y0, y1], ok=bool(ok))
    print(f"[viz] raster backend: {'imshow' if ok else 'PolyCollection'} "
          f"({flat.size} elements on a {Ngx}x{Ngy} lattice)")
    return _GRID


def _sel_mask(n, sel):
    if sel is None:
        return np.ones(n, dtype=bool)
    sel = np.asarray(sel)
    if sel.dtype != bool:
        m = np.zeros(n, dtype=bool)
        m[sel] = True
        return m
    return sel


def _solid_poly(quads, facecolors):
    return PolyCollection(quads, facecolors=facecolors,
                          edgecolors=facecolors, linewidths=0.3)


def _draw_scalar(ax, quads, vals, sel=None, cmap='viridis', norm=None,
                 zorder=1):
    vals = np.asarray(vals, dtype=float)
    m = _sel_mask(vals.size, sel)
    if _GRID.get('ok'):
        img = np.full(_GRID['n_cell'], np.nan)
        img[_GRID['idx'][m]] = vals[m]
        img = np.ma.masked_invalid(img.reshape(_GRID['Ngy'], _GRID['Ngx']))
        return ax.imshow(img, origin='lower', extent=_GRID['extent'],
                         cmap=cmap, norm=norm, interpolation='nearest',
                         aspect='equal', zorder=zorder)
    pc = PolyCollection(np.asarray(quads)[m], array=vals[m], cmap=cmap,
                        norm=norm, edgecolors='face', linewidths=0.3,
                        zorder=zorder)
    ax.add_collection(pc)
    return pc


def _draw_rgb(ax, quads, rgb, sel=None, zorder=1):
    rgb = np.asarray(rgb, dtype=float)
    n_tot = len(quads) if quads is not None else _GRID['idx'].size
    m = _sel_mask(n_tot, sel)
    if _GRID.get('ok'):
        img = np.zeros((_GRID['n_cell'], 4))
        img[_GRID['idx'][m], :3] = rgb
        img[_GRID['idx'][m], 3] = 1.0
        return ax.imshow(img.reshape(_GRID['Ngy'], _GRID['Ngx'], 4),
                         origin='lower', extent=_GRID['extent'],
                         interpolation='nearest', aspect='equal',
                         zorder=zorder)
    ax.add_collection(_solid_poly(np.asarray(quads)[m],
                                  [tuple(c) for c in rgb]))
    return None


def _raster_image(vals, sel=None, fill=np.nan):
    if not _GRID.get('ok'):
        return None
    vals = np.asarray(vals, dtype=float)
    m = _sel_mask(vals.size, sel)
    img = np.full(_GRID['n_cell'], fill, dtype=float)
    img[_GRID['idx'][m]] = vals[m]
    return img.reshape(_GRID['Ngy'], _GRID['Ngx'])


def _cell_centers():
    x0, x1, y0, y1 = _GRID['extent']
    Ngx, Ngy = _GRID['Ngx'], _GRID['Ngy']
    hx, hy = (x1 - x0) / Ngx, (y1 - y0) / Ngy
    xs = x0 + hx * (np.arange(Ngx) + 0.5)
    ys = y0 + hy * (np.arange(Ngy) + 0.5)
    return np.meshgrid(xs, ys)


# =====================================================================
# Run-folder loading + model / mesh reconstruction
# =====================================================================

def _normalize_mp(MP):
    """MP.pt files written by the old multi-GPU code store per-device LISTS
    of property tensors (e.g. MP['D'] = [tensor]); the single-GPU code stores
    plain tensors.  Normalize to the flat form so both load."""
    for k in ('D', 'E', 'P', 'alpha', 'kappa', 's', 'wt'):
        v = MP.get(k)
        if isinstance(v, list):
            MP[k] = v[0]
    return MP


def _torch_load(path, dev):
    """torch.load across torch versions (>=2.6 defaults weights_only=True,
    which rejects the MP dict / train_data payloads)."""
    try:
        return torch.load(path, map_location=dev, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=dev)


def import_train_module(path: Path):
    """Import the training script by path (its __main__ block is guarded, so
    this only defines functions/classes -- the run is NOT re-executed)."""
    spec = importlib.util.spec_from_file_location("train_mod", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_train_script(MP, override):
    if override:
        p = Path(override)
        if not p.exists():
            sys.exit(f"--train-script {p} does not exist")
        return p
    ex = MP.get('Example')
    name = TRAIN_SCRIPT_FOR_EXAMPLE.get(ex)
    if name is None:
        sys.exit(f"cannot infer the training script for Example={ex!r}; "
                 f"pass --train-script")
    p = REPO_ROOT / name
    if not p.exists():
        sys.exit(f"inferred training script {p} not found; pass --train-script")
    print(f"[viz] training module: {p.name} (inferred from Example={ex})")
    return p


def find_all_checkpoints(run: Path) -> list[Path]:
    cks = list(run.glob('Trained_models_*.pth'))

    def sort_key(p):
        tail = p.stem.split('_')[-1]
        return int(tail) if tail.isdigit() else float('inf')  # 'best' last

    cks.sort(key=sort_key)
    if not cks:
        sys.exit("no Trained_models_*.pth found in run folder")
    return cks


def _unigrid_cell_centers(MP, dev, dtype=torch.float32):
    Ngx, Ngy = MP['unigrid']
    (x0, x1), (y0, y1) = MP['domain']['x'], MP['domain']['y']
    hx, hy = (x1 - x0) / Ngx, (y1 - y0) / Ngy
    i = torch.arange(Ngx, device=dev, dtype=dtype)
    j = torch.arange(Ngy, device=dev, dtype=dtype)
    J, I = torch.meshgrid(j, i, indexing='ij')
    return torch.stack([x0 + hx * (I.reshape(-1) + 0.5),
                        y0 + hy * (J.reshape(-1) + 0.5)], dim=1)


def _slice_has_holes(fine):
    """A full-rectangle structured slice has n_unique_x * n_unique_y ==
    n_elem; any notch or hole breaks the equality."""
    ex = fine['X_elem']
    nx = torch.unique(ex[:, 0]).numel()
    ny = torch.unique(ex[:, 1]).numel()
    return nx * ny != ex.shape[0], nx, ny


def _prepare_mask_source(mesh_fine, MP, mode, dev, dtype=torch.float32):
    """Choose what _build_unigrid_mesh_data votes with.  That function keeps a
    unigrid cell only if it contains at least one fine element center; if the
    requested viz grid is finer than the source slice, cells fall between the
    centers and are silently deleted.  modes: 'auto' (finest slice; drop the
    mask entirely when it has no holes), 'finest', 'first', 'none'."""
    slices = mesh_fine['GPU0']
    keys = sorted(slices.keys())
    n_el = {k: slices[k]['X_elem'].shape[0] for k in keys}
    finest = max(keys, key=lambda k: n_el[k])
    chosen_key = keys[0] if mode == 'first' else finest
    fine = dict(slices[chosen_key])

    holes, nx, ny = _slice_has_holes(fine)
    Ngx, Ngy = MP['unigrid']
    coarser = (nx < Ngx) or (ny < Ngy)
    print(f"[viz] mask source '{chosen_key}': {nx}x{ny} lattice, "
          f"{'HAS holes/notches' if holes else 'no holes (full rectangle)'}")

    drop_mask = (mode == 'none') or (mode == 'auto' and not holes)
    if drop_mask:
        fine['X_elem'] = _unigrid_cell_centers(MP, dev, dtype)
    elif coarser:
        print(f"[viz] *** WARNING: mask source is {nx}x{ny} but the viz grid "
              f"is {Ngx}x{Ngy}; cells with no source center inside them WILL "
              f"be dropped.  Use --mask none if the holes are outside the "
              f"region you care about.")
    return {'GPU0': {'Mesh_01': fine}}


def build_models_and_mesh(mod, MP, train_data, nn_disp, nn_rho, dev,
                          viz_grid=None, filter_cg_tol=1e-6,
                          filter_cg_max_iter=2000, mask='auto'):
    """Rebuild the 5 LMGP models and the unigrid mesh exactly as run_worker
    does, except that the resolution may be overridden by viz_grid.  get_data
    and _build_unigrid_mesh_data come from the training module, so the mesh is
    built by the SAME code the run used -- only the cell size changes."""
    train_grid = tuple(MP['unigrid'])
    if viz_grid is not None and tuple(viz_grid) != train_grid:
        print(f"[viz] unigrid override: trained on "
              f"{train_grid[0]}x{train_grid[1]}, re-querying on "
              f"{viz_grid[0]}x{viz_grid[1]}")
        MP['unigrid'] = (int(viz_grid[0]), int(viz_grid[1]))

    if MP.get('helmholtz_on', False):
        MP['helmholtz_filter'] = mod.build_helmholtz_filter(
            r_min=MP['r_min'], device=dev, dtype=torch.float32,
            cg_tol=filter_cg_tol, cg_max_iter=filter_cg_max_iter)
    else:
        MP['helmholtz_filter'] = None
    MP['cur_mesh_key'] = 'Mesh_viz_%dx%d' % tuple(MP['unigrid'])
    Training, mesh_fine = mod.get_data(MP)
    mesh_src = _prepare_mask_source(mesh_fine, MP, mask, dev, torch.float32)
    mesh_data = mod._build_unigrid_mesh_data(MP, mesh_src, dev, torch.float32)
    MP['wt'] = torch.ones(4, dtype=torch.float32, device=dev)
    del Training, mesh_fine, mesh_src
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    LMGP = mod.LMGP
    num_phase = MP['num_phase']
    model_list = [
        LMGP(train_data['disp']['X'], train_data['disp']['y'], nn_disp,
             name_output="disp", MP=MP, num_output=2).to_device(dev),
        LMGP(train_data['disp_adj']['X'], train_data['disp_adj']['y'], nn_disp,
             name_output="adj", MP=MP, num_output=2).to_device(dev),
        LMGP(train_data['T']['X'], train_data['T']['y'], nn_disp,
             name_output="T", MP=MP, num_output=1).to_device(dev),
        LMGP(train_data['T_adj']['X'], train_data['T_adj']['y'], nn_disp,
             name_output="T_adj", MP=MP, num_output=1).to_device(dev),
        LMGP(train_data['phase']['X'], train_data['phase']['y'], nn_rho,
             name_output="rho", MP=MP, num_output=num_phase).to_device(dev),
    ]
    m = mesh_data['GPU0']['Mesh_01']
    m0 = model_list[0]
    m0.collocation_x = m['X_node'];    m0.elem_x = m['X_elem']
    m0.elem_vol = m['elem_vol'];       m0.conn = m['conn']
    m0.B = m['B'];   m0.B_T = m['B_T'];   m0.N = m['N'];   m0.detJ = m['detJ']
    m0.K_out_index = m['K_out_index'];   m0.K_out_magnitude = m['K_out_magnitude']
    m0.f_adj_index = m['f_adj_index'];   m0.f_adj_magnitude = m['f_adj_magnitude']
    return model_list


def load_checkpoint(model_list, ckpt_path, dev):
    ck = _torch_load(ckpt_path, dev)
    keys = ['model_u_state_dict', 'model_u_adj_state_dict', 'model_T_state_dict',
            'model_T_adj_state_dict', 'model_rho_state_dict']
    for m, k in zip(model_list, keys):
        m.mean_module_NN_All.load_state_dict(ck[k])
    print(f"[viz] loaded {ckpt_path.name}")


# =====================================================================
# Field prediction (line-for-line the GP extraction of calculate_TO_loss,
# with the mean and cross-covariance evaluations chunked over query points
# -- chunking is exact: the columns of g = k(X_train, X_query) are
# independent)
# =====================================================================

def _mean_chunked(model, Xq, chunk):
    if Xq.shape[0] <= chunk:
        return model.mean_module_NN_All(Xq)
    return torch.cat([model.mean_module_NN_All(Xq[s:s + chunk])
                      for s in range(0, Xq.shape[0], chunk)], dim=0)


def _gp_correction_chunked(model, out_idx, Xq, offset_vec, chunk):
    kern = model.independent_kernels[out_idx]
    Xtr = model.train_inputs_per_output[out_idx]
    if Xq.shape[0] <= chunk:
        return kern(Xtr, Xq).evaluate().t() @ offset_vec
    out = []
    for s in range(0, Xq.shape[0], chunk):
        g = kern(Xtr, Xq[s:s + chunk]).evaluate()
        out.append(g.t() @ offset_vec)
        del g
    return torch.cat(out, dim=0)


@torch.no_grad()
def predict_fields(model_list, MP, chunk=16384):
    """u, v, T at the nodes and the (filtered, clamped) phase weights at the
    element centers of the current mesh."""
    from gpytorch.settings import cholesky_jitter
    X = model_list[0].collocation_x.clone()
    elem_x = model_list[0].elem_x.clone()
    conn = model_list[0].conn
    num_phase = MP['num_phase']
    for m in model_list:
        m.train()

    with cholesky_jitter(1e-5):
        def chol(m):
            if m.chol_decomp is None:
                m.chol_decomp = []
                for i in range(m.num_output):
                    K_i = m.independent_kernels[i](
                        m.train_inputs_per_output[i]).evaluate()
                    m.chol_decomp.append(torch.linalg.cholesky(
                        K_i + 1e-5 * torch.eye(K_i.shape[0], device=K_i.device)))
        chol(model_list[0]); chol(model_list[2])
        if MP['design_flag']:
            chol(model_list[4])

    def offset(m, i):
        return torch.cholesky_solve(
            m.train_target_per_output[i].unsqueeze(-1)
            - m.mean_module_NN_All(m.train_inputs_per_output[i])[:, i].unsqueeze(-1),
            m.chol_decomp[i])

    m_col_disp = _mean_chunked(model_list[0], X, chunk)
    m_col_T = _mean_chunked(model_list[2], X, chunk)

    u = m_col_disp[:, 0:1] + _gp_correction_chunked(
        model_list[0], 0, X, offset(model_list[0], 0), chunk)
    v = m_col_disp[:, 1:2] + _gp_correction_chunked(
        model_list[0], 1, X, offset(model_list[0], 1), chunk)
    Node_T = m_col_T[:, 0:1] + _gp_correction_chunked(
        model_list[2], 0, X, offset(model_list[2], 0), chunk)
    Node_disp = torch.cat([u, v], dim=1)
    del m_col_disp, m_col_T

    m_elem_phase = _mean_chunked(model_list[4], elem_x, chunk)
    if MP['design_flag']:
        phase_weights = torch.cat(
            [m_elem_phase[:, i:i + 1] + _gp_correction_chunked(
                model_list[4], i, elem_x, offset(model_list[4], i), chunk)
             for i in range(model_list[4].num_output)], dim=1)
    else:
        phase_weights = m_elem_phase

    phase_weights_raw = phase_weights
    hf = MP.get('helmholtz_filter')
    if hf is not None:
        phase_weights = hf(phase_weights, conn, X,
                           mesh_key=MP.get('cur_mesh_key', 'default'))
    phase_weights = phase_weights.clamp(0.0, 1.0)
    beta_p = MP.get('beta_proj', None)
    if (beta_p is not None) and (num_phase == 2):
        eta_p = MP.get('eta_proj', 0.5)
        ws = phase_weights[:, 1:2].clamp(0.0, 1.0)
        den = math.tanh(beta_p * eta_p) + math.tanh(beta_p * (1.0 - eta_p))
        wsb = (math.tanh(beta_p * eta_p) + torch.tanh(beta_p * (ws - eta_p))) / den
        phase_weights = torch.cat([1.0 - wsb, wsb], dim=1)

    return Node_disp, Node_T, phase_weights, phase_weights_raw


# =====================================================================
# Field figures (true scale, dashed domain box)
# =====================================================================

def _quads(nodes, conn):
    return nodes[conn].cpu().numpy()


def _domain_box(ax, MP, **kw):
    (x0, x1), (y0, y1) = MP['domain']['x'], MP['domain']['y']
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                           linestyle='--', linewidth=1.0, edgecolor='k',
                           zorder=5, **kw))


def _field_fig(MP):
    (x0, x1), (y0, y1) = MP['domain']['x'], MP['domain']['y']
    W, Hd = (x1 - x0), (y1 - y0)
    fw = 10.0
    fh = fw * Hd / W + 1.3
    fig, ax = plt.subplots(figsize=(fw, fh))
    ax.set_aspect('equal')
    fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.10)
    return fig, ax


def _set_field_lims(ax, MP, pad=True):
    (x0, x1), (y0, y1) = MP['domain']['x'], MP['domain']['y']
    W, Hd = (x1 - x0), (y1 - y0)
    if pad:
        ax.set_xlim(x0 - 0.06 * W, x1 + 0.06 * W)
        ax.set_ylim(y0 - 0.10 * Hd, y1 + 0.16 * Hd)
    else:
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)


def fig_material_map(out, MP, nodes, conn, pw, eta, dpi):
    num_phase = pw.shape[1]
    names = (PHASE_NAMES if num_phase == len(PHASE_NAMES)
             else [f'phase {i}' for i in range(num_phase)])
    cols = (PHASE_COLORS if num_phase == len(PHASE_COLORS)
            else plt.cm.tab10(np.linspace(0, 1, num_phase)))
    rho_solid = pw[:, 1:].sum(dim=1)
    mat = pw[:, 1:].argmax(dim=1) + 1
    cat = torch.where(rho_solid >= eta, mat,
                      torch.zeros_like(mat)).cpu().numpy()
    fig, ax = _field_fig(MP)
    rgb_table = np.array([to_rgb(c) for c in cols])
    _draw_rgb(ax, _quads(nodes, conn), rgb_table[cat])
    _domain_box(ax, MP)
    D, P = MP['D'].cpu(), MP['P'].cpu()
    frac = [(cat == i).mean() for i in range(num_phase)]
    pwc = pw.cpu()
    massfrac = float((pwc * D).sum(dim=1).mean())
    costfrac = float((pwc * P).sum(dim=1).mean() / P.max())
    handles = [Patch(facecolor=cols[i], edgecolor='k', linewidth=0.5,
                     label=f'{names[i]}  ({100*frac[i]:.1f}%)')
               for i in range(num_phase)]
    ax.legend(handles=handles, loc='upper right', fontsize=FONT_SIZE - 4,
              framealpha=0.95, ncol=min(num_phase, 4))
    ax.set_title(f'material map (argmax, $\\eta$={eta:g});  '
                 f'massfrac = {massfrac:.3f},  costfrac = {costfrac:.3f}',
                 fontsize=FONT_SIZE - 2)
    _set_field_lims(ax, MP)
    fig.savefig(out, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return massfrac, costfrac, frac


def fig_phase_fields(out, MP, nodes, conn, pw, dpi):
    num_phase = pw.shape[1]
    names = (PHASE_NAMES if num_phase == len(PHASE_NAMES)
             else [f'phase {i}' for i in range(num_phase)])
    (x0, x1), (y0, y1) = MP['domain']['x'], MP['domain']['y']
    W, Hd = (x1 - x0), (y1 - y0)
    fw = 4.2 * num_phase
    fh = 4.2 * Hd / W + 1.2
    fig, axs = plt.subplots(1, num_phase, figsize=(fw, fh))
    fig.subplots_adjust(left=0.03, right=0.90, top=0.86, bottom=0.06,
                        wspace=0.08)
    quads = _quads(nodes, conn)
    norm01 = Normalize(0.0, 1.0)
    for i, ax in enumerate(np.atleast_1d(axs)):
        vals = pw[:, i].clamp(0, 1).cpu().numpy()
        pc = _draw_scalar(ax, quads, vals, cmap='viridis', norm=norm01)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        _domain_box(ax, MP)
        ax.set_title(names[i], fontsize=FONT_SIZE - 2)
        _set_field_lims(ax, MP, pad=False)
    cax = fig.add_axes([0.92, 0.10, 0.015, 0.72])
    cbar = fig.colorbar(pc, cax=cax)
    cbar.set_label(r'$w_i$', fontsize=FONT_SIZE - 2)
    fig.savefig(out, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def fig_topology_deformed(out, MP, nodes, conn, pw, Node_disp,
                          K_out_index, eta, dpi):
    """Deformed vs undeformed solid structure (true scale; needs polygons)."""
    rho_solid = pw[:, 1:].sum(dim=1)
    solid = (rho_solid >= eta).cpu().numpy()
    Xd = nodes + Node_disp
    fig, ax = _field_fig(MP)
    ax.add_collection(PolyCollection(_quads(nodes, conn)[solid],
                                     facecolors='tab:blue', alpha=0.35,
                                     edgecolors='none'))
    ax.add_collection(PolyCollection(_quads(Xd, conn)[solid],
                                     facecolors='tab:red', alpha=0.35,
                                     edgecolors='none'))
    _domain_box(ax, MP)
    port = int(K_out_index[0])
    dx, dy = float(Node_disp[port, 0]), float(Node_disp[port, 1])
    ax.plot([nodes[port, 0].item()], [nodes[port, 1].item()], 'o',
            color='tab:blue', ms=5)
    ax.plot([Xd[port, 0].item()], [Xd[port, 1].item()], 'o',
            color='tab:red', ms=5)
    ax.annotate(rf'$\Delta x_\mathrm{{port}} = {dx:+.2f}$'
                + '\n' + rf'$\Delta y_\mathrm{{port}} = {dy:+.2f}$',
                xy=(Xd[port, 0].item(), Xd[port, 1].item()),
                xytext=(12, 12), textcoords='offset points',
                fontsize=FONT_SIZE - 3,
                bbox=dict(boxstyle='round', fc='white', ec='k', lw=0.6))
    handles = [Patch(facecolor='tab:blue', alpha=0.35, label='undeformed'),
               Patch(facecolor='tab:red', alpha=0.35,
                     label='deformed (true scale)')]
    ax.legend(handles=handles, loc='upper right', fontsize=FONT_SIZE - 4)
    _set_field_lims(ax, MP)
    fig.savefig(out, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return dx, dy


def fig_T_field(out, MP, nodes, conn, pw, Node_T, eta, dpi):
    Tn = Node_T[:, 0]
    Te = Tn[conn].mean(dim=1).cpu().numpy()
    fig, ax = _field_fig(MP)
    pc = _draw_scalar(ax, _quads(nodes, conn), Te, cmap='inferno')
    rho_solid = pw[:, 1:].sum(dim=1).cpu().numpy()
    img = _raster_image(rho_solid, fill=0.0)
    if img is not None:
        Xc, Yc = _cell_centers()
        ax.contour(Xc, Yc, img, levels=[eta], colors='k', linewidths=0.6,
                   alpha=0.8, zorder=4)
    else:
        solid = rho_solid >= eta
        ax.add_collection(PolyCollection(_quads(nodes, conn)[solid],
                                         facecolors='none', edgecolors='k',
                                         linewidths=0.05, alpha=0.4))
    _domain_box(ax, MP)
    cb = fig.colorbar(pc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label('T (K)', fontsize=FONT_SIZE - 2)
    td = MP.get('tdep_props', None)
    win = td.get('T_valid') if td else None
    title = f'T field: [{Te.min():.0f}, {Te.max():.0f}] K'
    if win:
        inside = (Te.min() >= win[0]) and (Te.max() <= win[1])
        title += (f';  tdep window [{win[0]:.0f}, {win[1]:.0f}] K '
                  + ('(inside)' if inside else '(*** OUTSIDE ***)'))
    ax.set_title(title, fontsize=FONT_SIZE - 2)
    _set_field_lims(ax, MP)
    fig.savefig(out, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return float(Te.min()), float(Te.max())


# =====================================================================
# CSV / npz export
# =====================================================================

def write_topology_csvs(outdir, MP, elem_x, pw, eta):
    """COMSOL-style (x, y, value) CSVs at the element centers.
    topo_material_id.csv is the categorical design the evaluation notebooks
    consume: 0 void, then material index by argmax where rho_solid >= eta."""
    xy = elem_x[:, :2].detach().cpu().numpy()
    names = (PHASE_NAMES if pw.shape[1] == len(PHASE_NAMES)
             else [f'phase{i}' for i in range(pw.shape[1])])

    def _write(path, vals, col):
        arr = np.column_stack([xy, np.asarray(vals, dtype=float)])
        np.savetxt(path, arr, fmt='%.8e',
                   header='COMSOL Spreadsheet interpolation data\n'
                          f'columns: x y {col}',
                   comments='% ')

    rho_solid = pw[:, 1:].sum(dim=1).cpu().numpy()
    _write(Path(outdir) / 'topo_rho_solid.csv', rho_solid, 'rho')
    for i in range(1, pw.shape[1]):
        _write(Path(outdir) / f'topo_w_{names[i]}.csv',
               pw[:, i].clamp(0, 1).cpu().numpy(), f'w_{names[i]}')
    mat = (pw[:, 1:].argmax(dim=1) + 1).cpu().numpy()
    cat = np.where(rho_solid >= eta, mat, 0).astype(float)
    _write(Path(outdir) / 'topo_material_id.csv', cat, 'mat_id')


def write_fields_npz(path, MP, elem_x, pw, pw_raw, Node_T, conn, eta, grid):
    rho_solid = pw[:, 1:].sum(dim=1)
    mat = pw[:, 1:].argmax(dim=1) + 1
    cat = torch.where(rho_solid >= eta, mat, torch.zeros_like(mat))
    T_e = Node_T[conn].reshape(conn.shape[0], -1).mean(dim=1)
    np.savez_compressed(
        path,
        elem_x=elem_x[:, :2].detach().cpu().numpy(),
        w=pw.cpu().numpy(),
        w_raw=pw_raw.detach().cpu().numpy(),
        rho_solid=rho_solid.cpu().numpy(),
        mat_id=cat.cpu().numpy().astype(np.int16),
        T_elem=T_e.cpu().numpy(),
        eta=np.array(eta),
        unigrid=np.array(grid),
        domain=np.array([MP['domain']['x'], MP['domain']['y']]))


# =====================================================================
# History figures
# =====================================================================

def _round_up_nice(value):
    if value <= 0:
        return 1
    mag = 10 ** int(math.floor(math.log10(value)))
    for mult in (1, 2, 2.5, 5, 10):
        if mult * mag >= value:
            return int(mult * mag)
    return int(10 * mag)


def _hist_axes(n_epoch):
    fig, ax = plt.subplots(figsize=FIGSIZE_HIST)
    fig.subplots_adjust(**SUBPLOT_PARAMS_HIST)
    x_max = _round_up_nice(n_epoch)
    ticks = [t for t in (2000, 4000, 6000, 8000, 10000) if t < x_max] + [x_max]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xlim(-x_max * 0.025, x_max * 1.025)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(3, 3),
                        useMathText=True)
    ax.xaxis.get_offset_text().set_fontsize(OFFSET_FONT_SIZE)
    ax.set_xlabel('epoch')
    return fig, ax


def hist_plot(out, series, labels, ylabel, dpi, hlines=(), ylim=None,
              drop_first=0, logy=False):
    n = max(len(s) for s in series)
    fig, ax = _hist_axes(n)
    for i, (s, lab) in enumerate(zip(series, labels)):
        y = np.asarray(s, dtype=float)
        x = np.arange(len(y))
        if drop_first:
            x, y = x[drop_first:], y[drop_first:]
        ax.plot(x, y, color=f'C{i}', linewidth=LINEWIDTH, label=lab)
    for hy, hlab in hlines:
        ax.axhline(hy, color='k', linestyle='--', linewidth=1.2,
                   label=hlab, zorder=0)
    if logy:
        ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    if len(series) > 1 or hlines:
        ax.legend(fontsize=FONT_SIZE - 3, framealpha=0.95)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


# =====================================================================
# Evaluator export
# =====================================================================

def _run_seed(run: Path, nn_disp):
    state = nn_disp.get('state', None)
    if isinstance(state, int):
        return state
    m = re.match(r'Run_(\d+)_', run.name)
    if m:
        return int(m.group(1))
    sys.exit(f"cannot determine the seed for {run.name}; expected an integer "
             f"'state' in NN_config_disp.json or a Run_<seed>_* folder name")


def export_for_evaluator(run, mod, dev, args, nn_disp, nn_rho, train_data):
    """Write eval_export/Run_<seed>_<constitutive>_<TD>/topo_material_id.csv
    at the TRAINING resolution -- the exact input the cross-evaluation
    notebooks (evaluation/evaluate_*.ipynb) glob for and parse.  Always uses
    a fresh MP from disk so a --viz-grid override cannot leak in."""
    MP = _normalize_mp(_torch_load(run / 'MP.pt', dev))
    MP['base_folder'] = str(REPO_ROOT) + '/'
    train_grid = tuple(MP['unigrid'])

    ck_want = args.eval_checkpoint
    cks = find_all_checkpoints(run)
    if ck_want == 'best':
        best = run / 'Trained_models_best.pth'
        ckpt = best if best.exists() else cks[-1]
        if not best.exists():
            print(f"[eval-export] no Trained_models_best.pth; using "
                  f"{ckpt.name}")
    elif ck_want == 'last':
        ckpt = cks[-1]
    else:
        ckpt = run / f'Trained_models_{ck_want}.pth'
        if not ckpt.exists():
            sys.exit(f"--eval-checkpoint {ck_want}: {ckpt.name} not found")

    seed = _run_seed(run, nn_disp)
    const = MP.get('constitutive', 'hencky')
    TD = int(round(float(MP['TD'])))
    name = f"Run_{seed}_{const}_{TD}"
    outdir = run / 'eval_export' / name
    outdir.mkdir(parents=True, exist_ok=True)

    model_list = build_models_and_mesh(
        mod, MP, train_data, nn_disp, nn_rho, dev, viz_grid=train_grid,
        filter_cg_tol=args.filter_cg_tol,
        filter_cg_max_iter=args.filter_cg_iter, mask=args.mask)
    load_checkpoint(model_list, ckpt, dev)
    _, _, pw, _ = predict_fields(model_list, MP, chunk=args.chunk)
    write_topology_csvs(outdir, MP, model_list[0].elem_x, pw, args.eta)

    n_solid = int((pw[:, 1:].sum(dim=1) >= args.eta).sum())
    print(f"[eval-export] {outdir.relative_to(run)}  "
          f"(from {ckpt.name}; {train_grid[0]}x{train_grid[1]} grid, "
          f"{n_solid} solid elements)")
    print(f"[eval-export] to evaluate: zip the eval_export/Run_* folders of "
          f"all runs and upload the zip in the notebook's data cell.")
    del model_list, pw
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =====================================================================
# main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run_folder', required=True,
                    help='a finished Results/<Example>/<Case>/Run_* folder')
    ap.add_argument('--train-script', default=None,
                    help='training script to import (default: inferred from '
                         "MP['Example'] -> run_gripper.py / run_actuator.py)")
    ap.add_argument('--device',
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--dpi', type=int, default=600)
    ap.add_argument('--eta', type=float, default=0.5,
                    help='solid/material threshold for categorical views')
    ap.add_argument('--out', default=None,
                    help='output folder (default: <run_folder>/viz_<Ngx>x<Ngy>)')
    ap.add_argument('--viz-grid', nargs=2, type=int, default=None,
                    metavar=('NGX', 'NGY'),
                    help='resolution at which the fields are re-queried for '
                         'the FIGURES (default: the training grid). The '
                         'evaluator export always uses the training grid.')
    ap.add_argument('--mask', choices=['auto', 'finest', 'first', 'none'],
                    default='auto',
                    help='how the unigrid geometry mask is sourced (see '
                         '_prepare_mask_source)')
    ap.add_argument('--chunk', type=int, default=16384,
                    help='query points per GP/NN evaluation chunk')
    ap.add_argument('--force-poly', action='store_true',
                    help='disable the imshow raster backend')
    ap.add_argument('--filter-cg-tol', type=float, default=1e-6)
    ap.add_argument('--filter-cg-iter', type=int, default=2000)
    ap.add_argument('--eval-checkpoint', default='best',
                    help="checkpoint for the evaluator export: 'best', "
                         "'last', or an epoch number (default: best)")
    ap.add_argument('--no-eval-export', action='store_true',
                    help='skip the evaluator export')
    ap.add_argument('--no-figures', action='store_true',
                    help='skip figures/CSVs per checkpoint (evaluator export '
                         'only)')
    args = ap.parse_args()

    configure_rcparams()
    run = Path(args.run_folder)
    dev = torch.device(args.device)

    MP = _normalize_mp(_torch_load(run / 'MP.pt', dev))
    MP['base_folder'] = str(REPO_ROOT) + '/'   # Data/ lives at the repo root
    train_grid = tuple(MP['unigrid'])
    Ng = tuple(args.viz_grid) if args.viz_grid else train_grid

    mod = import_train_module(resolve_train_script(MP, args.train_script))
    train_data = _torch_load(run / 'train_data.pth', dev)
    nn_disp = json.load(open(run / 'NN_config_disp.json'))
    nn_rho = json.load(open(run / 'NN_config_rho.json'))

    if not args.no_figures:
        outdir = Path(args.out) if args.out else run / f'viz_{Ng[0]}x{Ng[1]}'
        outdir.mkdir(parents=True, exist_ok=True)

        model_list = build_models_and_mesh(
            mod, MP, train_data, nn_disp, nn_rho, dev, viz_grid=Ng,
            filter_cg_tol=args.filter_cg_tol,
            filter_cg_max_iter=args.filter_cg_iter, mask=args.mask)
        nodes = model_list[0].collocation_x
        conn = model_list[0].conn
        K_out_index = model_list[0].K_out_index
        elem_x = model_list[0].elem_x
        setup_raster_grid(MP, elem_x, MP['unigrid'],
                          force_poly=args.force_poly)

        # ---- run-level history plots ------------------------------------
        th_path = run / 'timeHistory.json'
        if th_path.exists():
            th = json.load(open(th_path))
            if th.get('u_out'):
                hist_plot(outdir / 'hist_u_out.png', [th['u_out']], [None],
                          r'$u_\mathrm{out}$', args.dpi)
            if th.get('massfrac'):
                hist_plot(outdir / 'hist_massfrac.png', [th['massfrac']],
                          ['massfrac'], 'mass fraction', args.dpi,
                          hlines=[(MP['massfrac_f'], 'target')])
            if th.get('costfrac'):
                hist_plot(outdir / 'hist_costfrac.png', [th['costfrac']],
                          ['costfrac'], 'cost fraction', args.dpi,
                          hlines=[(MP['costfrac_f'], 'target')])
            if th.get('adj_vw_ratio'):
                hist_plot(outdir / 'hist_vw_ratio.png', [th['adj_vw_ratio']],
                          ['vw ratio'], 'adjoint health', args.dpi,
                          hlines=[(1.0, 'exact')], ylim=(0.0, 1.5),
                          drop_first=int(MP.get('adjoint_start_ep', 0)))

        # ---- per-checkpoint figures + CSVs ------------------------------
        for ckpt_path in find_all_checkpoints(run):
            ck_out = outdir / ckpt_path.stem
            ck_out.mkdir(parents=True, exist_ok=True)
            load_checkpoint(model_list, ckpt_path, dev)
            Node_disp, Node_T, pw, pw_raw = predict_fields(
                model_list, MP, chunk=args.chunk)

            massfrac, costfrac, frac = fig_material_map(
                ck_out / 'fig_material_map.png', MP, nodes, conn, pw,
                args.eta, args.dpi)
            fig_phase_fields(ck_out / 'fig_phase_fields.png', MP, nodes,
                             conn, pw, args.dpi)
            dx, dy = fig_topology_deformed(
                ck_out / 'fig_topology_deformed.png', MP, nodes, conn, pw,
                Node_disp, K_out_index, args.eta, args.dpi)
            Tmin, Tmax = fig_T_field(ck_out / 'fig_T_field.png', MP, nodes,
                                     conn, pw, Node_T, args.eta, args.dpi)
            write_topology_csvs(ck_out, MP, elem_x, pw, args.eta)
            write_fields_npz(ck_out / f'fields_{Ng[0]}x{Ng[1]}.npz', MP,
                             elem_x, pw, pw_raw, Node_T, conn, args.eta, Ng)

            names = (PHASE_NAMES if pw.shape[1] == len(PHASE_NAMES)
                     else [f'phase {i}' for i in range(pw.shape[1])])
            with open(ck_out / 'summary.txt', 'w') as f:
                f.write(f"checkpoint: {ckpt_path.name}\n")
                f.write(f"viz grid: {Ng[0]}x{Ng[1]} ({conn.shape[0]} elements "
                        f"kept, {nodes.shape[0]} nodes); trained on "
                        f"{train_grid[0]}x{train_grid[1]}\n")
                f.write(f"u_out (port): dx = {dx:+.4f}, dy = {dy:+.4f}\n")
                f.write(f"massfrac = {massfrac:.4f} "
                        f"(target {MP['massfrac_f']})\n")
                f.write(f"costfrac = {costfrac:.4f} "
                        f"(target {MP['costfrac_f']})\n")
                for i, nm in enumerate(names):
                    f.write(f"  {nm:>5s}: {100*frac[i]:.2f}% of elements\n")
                f.write(f"T range on elements: [{Tmin:.1f}, {Tmax:.1f}] K\n")
            print(f"[viz] wrote {ck_out}")

            del Node_disp, Node_T, pw, pw_raw
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del model_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not args.no_eval_export:
        export_for_evaluator(run, mod, dev, args, nn_disp, nn_rho, train_data)

    print("[viz] done.")


if __name__ == '__main__':
    main()
