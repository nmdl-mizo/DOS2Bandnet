#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
- If mp/*/ebs_* directories are specified directly with --dirs, save each directory's inputs (DOS, ED) and
  (1) GT band, (2) LDM-predicted band in check_one-data style + four themes.
- The themes are reused as-is by importing
  /home/yrjin/_code/vasp_related/dos_plot.py, /home/yrjin/_code/structure_related/diffrac_1d_plot.py,
  and /home/yrjin/_code/vasp_related/ebs_plot.py.

Example:
  python plot_selected_from_split.py exp-abc12345 --dirs mp-12345/ebs_100 mp-99999/ebs_110 \
         --steps 50 --guidance 1.0 --view_range -2 4 --zoom_range 0 3.5 \
         --theme-color blue --hatch // --hide-y-numbers

As before, {split}.txt + --idx mode also works.
"""
import os, sys, re, argparse, numpy as np, torch
from scipy.ndimage import gaussian_filter1d
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import AutoMinorLocator

# === Load train.py API ===
HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.append(HERE)
from dos2bandnet.train_core import (
    TrainConfig, BandDataset, build_models, device_auto, make_schedule,
    sample_ddim, resolve_ckpt, _ensure_simple_encoders,
)
from dos2bandnet.model import ddim_timesteps, q_sample, EMA

# --- add theme modules from user path ---
USER_PATHS = ["/home/yrjin/_code/vasp_related", "/home/yrjin/_code/structure_related"]
for _p in USER_PATHS:
    if _p not in sys.path: sys.path.append(_p)

# import theme modules (reuse functions/palettes)
from dos_plot import plot_dos as dos_plot_curve, PALETTES as DOS_PALETTES  # :contentReference[oaicite:3]{index=3}
from diffrac_1d_plot import plot_1d as ed_plot_curve, PALETTES as ED_PALETTES  # :contentReference[oaicite:4]{index=4}
from ebs_plot import build_colormap as build_arpes_cmap  # :contentReference[oaicite:5]{index=5}


# --------------------------- paths/utils ---------------------------
def _cm2inch(w_cm, h_cm):
    return (float(w_cm) / 2.54, float(h_cm) / 2.54)


# --- k-path number → symbol mapping ---
KPATH_MAP = {
    "100": "X",
    "010": "Y",
    "001": "Z",
    "110": "S",
    "101": "U",
    "011": "T",
    "111": "R",
    "000": "Γ",  # if needed
}


def kpath_to_symbol(label: str) -> str:
    s = str(label).strip()
    return KPATH_MAP.get(s, s)


def resolve_base_run_name(run_name: str) -> str:
    if "/finetune" in run_name:
        return run_name.split("/finetune", 1)[0]
    return run_name


def resolve_latest_finetune_run(out_root: str, run_name: str) -> str | None:
    base_dir = os.path.join(out_root, run_name)
    if not os.path.isdir(base_dir):
        return None
    candidates = []
    for entry in os.listdir(base_dir):
        m = re.match(r"^finetune(\d+)$", entry)
        if m:
            candidates.append((int(m.group(1)), entry))
    if not candidates:
        return None
    _, latest = max(candidates, key=lambda item: item[0])
    return f"{run_name}/{latest}"


def _sigma_pts_from_fwhm(fwhm, grid):
    """Convert FWHM (physical unit) → sigma[samples]. grid is the x-axis (energy/effective q)."""
    if fwhm is None or fwhm <= 0 or grid is None or len(grid) < 2:
        return None
    dx = float(np.mean(np.diff(grid)))
    if dx <= 0:
        return None
    return float(fwhm) / (2.3548200450309493 * dx)  # 2*sqrt(2*ln2)


def _gauss_1d(arr, sigma_pts):
    if sigma_pts is None or sigma_pts <= 0:
        return arr
    return gaussian_filter1d(arr, float(sigma_pts), mode="nearest")


def resolve_run_name(arg: str, prefix: str = "exp") -> str:
    if os.path.isdir(os.path.join("runs", arg)): return arg
    if re.fullmatch(r"[a-z0-9]{8}", arg): return f"{prefix}-{arg}"
    return arg


def read_list(path: str):
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith('#')]


def ensure_out_dir(out_root: str, run_name: str) -> str:
    d = os.path.join(out_root, run_name, "previews_selected")
    os.makedirs(d, exist_ok=True)
    return d


def _style_axes(ax, fontsize=10, tickpad=5):
    ax.xaxis.set_minor_locator(AutoMinorLocator());
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis="both", which="major", direction="in", length=6, width=1.2, top=True, right=True,
                   labelsize=fontsize, pad=tickpad)
    ax.tick_params(axis="both", which="minor", direction="in", length=3, width=1.0, top=True, right=True)
    for s in ["left", "right", "top", "bottom"]:
        ax.spines[s].set_visible(True);
        ax.spines[s].set_linewidth(1.6)


def electron_wavelength_angstrom(kV: float) -> float:
    h = 6.62607015e-34;
    m = 9.10938356e-31;
    e = 1.602176634e-19;
    c = 2.99792458e8
    V = float(kV) * 1e3;
    p = (2 * m * e * V * (1.0 + (e * V) / (2 * m * c ** 2))) ** 0.5;
    return (h / p) * 1e10


def load_kaxis_from_klables(klables_path):
    try:
        arr = np.loadtxt(klables_path, dtype=np.string_, skiprows=1, usecols=(0, 1))
        labels = [x.decode('utf-8', 'ignore') for x in arr[:-1, 0].tolist()]
        labels = [("Γ" if (s.upper() in ("GAMMA", "G")) else s) for s in labels]
        ticks = [float(x) for x in arr[:-1, 1].tolist()]
        return ticks, labels
    except Exception:
        return None, None


def load_krange_from_ebs(ebs_path):
    try:
        d = np.loadtxt(ebs_path);
        k = d[:, 0].astype(float);
        return float(k.min()), float(k.max())
    except Exception:
        return None, None


def load_inputs(band_dir, dos_range=(-10.0, 5.0), band_range=(-5.0, 5.0),
                two_theta_range=(0.05, 4.3), accel_kV=200.0):
    f_dif = os.path.join(band_dir, "diffrac.npy")
    f_dos = os.path.join(band_dir, "dos.npy")
    f_bnd = os.path.join(band_dir, "ebs.npy")
    f_kl = os.path.join(band_dir, "KLABELS")
    f_ebs = os.path.join(band_dir, "EBS.dat")

    Iq = np.load(f_dif)  # (nq,)
    DOS = np.load(f_dos)  # (nE,) or (sites,nE)
    Bgt = np.load(f_bnd)  # (nK,nE)

    Emin_dos, Emax_dos = map(float, dos_range)
    nE = DOS.shape[-1];
    E_grid = np.linspace(Emin_dos, Emax_dos, nE)

    Emin_band, Emax_band = map(float, band_range)
    nK = Bgt.shape[0]
    kmin, kmax = load_krange_from_ebs(f_ebs);
    if kmin is None: kmin, kmax = 0.0, float(nK - 1)

    k_ticks, k_labels = load_kaxis_from_klables(f_kl)

    # q-grid from two-theta
    try:
        lam = electron_wavelength_angstrom(accel_kV)
        tt_min = np.radians(two_theta_range[0]);
        tt_max = np.radians(two_theta_range[1])
        q_min = (2.0 / lam) * np.sin(tt_min / 2.0);
        q_max = (2.0 / lam) * np.sin(tt_max / 2.0)
        q = np.linspace(q_min, q_max, Iq.size);
        q_label = r"$q$ (Å$^{-1}$)";
        q_pitch = np.nan  # unknown without POSCAR/dir
    except Exception:
        q = np.arange(Iq.size, dtype=float);
        q_label = "q (index)";
        q_pitch = np.nan

    info = {"E_grid": E_grid, "Emin_band": Emin_band, "Emax_band": Emax_band,
            "kmin": kmin, "kmax": kmax, "k_ticks": k_ticks, "k_labels": k_labels,
            "q": q, "q_label": q_label, "q_pitch": q_pitch}
    return Iq, DOS, Bgt, info


def parse_material_and_direction(dir_path: str):
    d2 = os.path.basename(os.path.abspath(dir_path))
    d1 = os.path.basename(os.path.dirname(os.path.abspath(dir_path)))
    mat = d1
    m = re.search(r"ebs[-_]?(.+)$", d2, flags=re.IGNORECASE)
    if m:
        direc = m.group(1)
    else:
        direc = d2.split('_', 1)[-1] if ('_' in d2) else d2

    def sanitize(s):
        return re.sub(r"[^A-Za-z0-9_\-\[\]\(\)]+", "", s)

    return sanitize(mat), sanitize(direc)


def build_k_ticks_labels(info, start_label: str, end_label: str):
    kmin, kmax = info["kmin"], info["kmax"]
    t0 = info.get("k_ticks", None);
    l0 = info.get("k_labels", None)
    if t0 is not None and l0 is not None and len(t0) == len(l0) and len(t0) > 0:
        ticks, labels = list(t0), list(l0)
        if abs(ticks[0] - kmin) < 1e-6:
            labels[0] = start_label
        else:
            ticks = [kmin] + ticks; labels = [start_label] + labels
        if abs(ticks[-1] - kmax) < 1e-6:
            labels[-1] = end_label
        else:
            ticks.append(kmax); labels.append(end_label)
        return ticks, labels
    return [kmin, kmax], [start_label, end_label]


def _latent_to_2d(zchw: torch.Tensor, how: str = "l2") -> np.ndarray:
    """Aggregate (1,C,H,W) -> (H,W)"""
    if zchw.dim() == 4: zchw = zchw[0]
    C, H, W = zchw.shape
    if how == "l2":
        x = torch.sqrt(torch.mean(zchw ** 2, dim=0))
    elif how == "mean":
        x = torch.mean(zchw, dim=0)
    elif how == "absmean":
        x = torch.mean(torch.abs(zchw), dim=0)
    elif how == "ch0":
        x = zchw[0]
    else:
        x = torch.sqrt(torch.mean(zchw ** 2, dim=0))
    return x.detach().cpu().numpy()


# --------------------------- Plot (theme applied) ---------------------------
def plot_three_panel_thematic(Iq, DOS, Bmap, info, view_range=(-5.0, 5.0), zoom_range=(0.0, 3.5),
                              save_path=None, title_band=None, k0_label="Γ", kend_label=None,
                              mat_name=None, direction=None, hatch="//",
                              use_hatch=True, hide_y_numbers=False,
                              dos_theme='red', ed_theme='blue', size_cm=None,
                              dos_gauss_points=None, dos_fwhm=None, ed_gauss_points=None, ed_fwhm=None):
    E_grid = info["E_grid"];
    Emin_band = info["Emin_band"];
    Emax_band = info["Emax_band"]
    kmin = info["kmin"];
    kmax = info["kmax"];
    q = info["q"];
    q_label = info["q_label"];
    q_pitch = info["q_pitch"]

    # energy grid
    E_band_grid = np.linspace(Emin_band, Emax_band, Bmap.shape[1])

    # view/zoom normalization
    e0, e1 = float(view_range[0]), float(view_range[1])
    e_lo = max(min(e0, e1), min(Emin_band, Emax_band));
    e_hi = min(max(e0, e1), max(Emin_band, Emax_band))
    m_main = (E_band_grid >= e_lo) & (E_band_grid <= e_hi)
    if not np.any(m_main): m_main = slice(None)
    vmin_main = float(np.nanmin(Bmap[:, m_main]));
    vmax_main = float(np.nanmax(Bmap[:, m_main]))
    norm_main = Normalize(vmin=vmin_main, vmax=vmax_main)

    z0, z1 = float(zoom_range[0]), float(zoom_range[1])
    z_lo = max(min(z0, z1), min(Emin_band, Emax_band));
    z_hi = min(max(z0, z1), max(Emin_band, Emax_band))
    m_zoom = (E_band_grid >= z_lo) & (E_band_grid <= z_hi)
    if not np.any(m_zoom): m_zoom = slice(None)
    vmin_zoom = float(np.nanmin(Bmap[:, m_zoom]));
    vmax_zoom = float(np.nanmax(Bmap[:, m_zoom]))
    norm_zoom = Normalize(vmin=vmin_zoom, vmax=vmax_zoom)

    # Figure
    figsize = _cm2inch(*size_cm) if size_cm else (11, 4)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.07, hspace=0.02)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 0.8], height_ratios=[1, 1, 1])

    # (1,1) DOS — use dos_plot.py theme function
    ax_dos = fig.add_subplot(gs[0, 0])
    color = DOS_PALETTES.get(dos_theme, DOS_PALETTES["red"])

    # calculate σ_pts (priority: --dos-fwhm -> --dos-gauss-points)
    sigma_dos = _sigma_pts_from_fwhm(dos_fwhm, E_grid)
    if sigma_dos is None:
        sigma_dos = dos_gauss_points

    # if multichannel, broaden each channel separately and then average
    if DOS.ndim > 1:
        D_proc = np.vstack([_gauss_1d(ch, sigma_dos) for ch in DOS])
        y = np.nanmean(D_proc, axis=0)
    else:
        y = _gauss_1d(DOS, sigma_dos)

    # first set xlim/ylim and then call plot_dos on top of it (including hatch)
    ax_dos.set_xlim(view_range)
    yref = y[(E_grid >= view_range[0]) & (E_grid <= view_range[1])] if y.size else y
    if yref.size:
        y0 = 0.0 if float(np.nanmin(yref)) >= 0 else float(np.nanmin(yref))
        ax_dos.set_ylim(y0, float(np.nanmax(yref)) * 1.05)
    dos_plot_curve(ax_dos, E_grid, (y if y.ndim == 1 else y),
                   view_range=view_range, swap_xy=False, hatch=hatch, use_hatch=use_hatch,
                   color=color, xlabel="Energy (eV)", ylabel="DOS (a.u.)",
                   fontsize=10, tickpad=5, normalize=False,
                   hide_y_numbers=hide_y_numbers, spectrum_only=False)  # :contentReference[oaicite:6]{index=6}

    # (2,1) ED — use diffrac_1d_plot.py theme function
    ax_dif = fig.add_subplot(gs[1, 0])
    ed_color = ED_PALETTES.get(ed_theme, ED_PALETTES["blue"])

    # calculate σ_pts (priority: --ed-fwhm -> --ed-gauss-points) ; q is already a uniform grid in info
    sigma_ed = _sigma_pts_from_fwhm(ed_fwhm, q)
    if sigma_ed is None:
        sigma_ed = ed_gauss_points

    Iq_plot = _gauss_1d(Iq, sigma_ed)

    ed_plot_curve(ax_dif, q, Iq_plot, q_pitch=q_pitch, color=ed_color,
                  hatch=hatch, use_hatch=use_hatch,
                  fontsize=10, tickpad=5, hide_y_numbers=hide_y_numbers,
                  spectrum_only=False)  # :contentReference[oaicite:7]{index=7}
    ax_dif.set_xlabel(q_label, fontsize=12, labelpad=1)
    ax_dif.grid(True, which="major", axis="y", alpha=0.2, linestyle="-", linewidth=0.8)

    # common k ticks
    ax_band = fig.add_subplot(gs[0:2, 1])
    newcmp = build_arpes_cmap()  # keep user custom colormap  :contentReference[oaicite:8]{index=8}
    im_main = ax_band.imshow(Bmap.T, origin="lower", aspect="auto",
                             extent=(kmin, kmax, Emin_band, Emax_band),
                             cmap=newcmp, interpolation="nearest", norm=norm_main)
    if kend_label is None and (direction is not None): kend_label = direction
    kticks, klabels = build_k_ticks_labels(info, start_label=k0_label, end_label=kend_label or "end")
    ax_band.set_xticks(kticks);
    ax_band.set_xticklabels(klabels)
    xlab = "k-path";
    if direction is not None: xlab = f"k-path  {k0_label} → {direction}"
    ax_band.set_xlabel(xlab, fontsize=12, labelpad=2)
    ax_band.set_ylabel("Energy (eV)", fontsize=12, labelpad=2)
    if title_band:
        if mat_name and direction:
            ax_band.set_title(f"{title_band}  [{mat_name} | {direction}]", fontsize=10)
        else:
            ax_band.set_title(title_band, fontsize=10)
    ax_band.set_ylim(view_range);
    ax_band.margins(x=0.0, y=0.0)
    _style_axes(ax_band, fontsize=10, tickpad=5)
    cbar = fig.colorbar(im_main, ax=ax_band, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=10, pad=4);
    cbar.set_label("Norm. intensity (view-range)", fontsize=10, labelpad=8)

    # (1,3) Band zoom
    ax_zoom = fig.add_subplot(gs[2, 1])
    im_zoom = ax_zoom.imshow(Bmap.T, origin="lower", aspect="auto",
                             extent=(kmin, kmax, Emin_band, Emax_band),
                             cmap=newcmp, interpolation="nearest", norm=norm_zoom)
    ax_zoom.set_xticks(kticks);
    ax_zoom.set_xticklabels(klabels)
    ax_zoom.set_xlabel("k-path", fontsize=10, labelpad=2)
    ax_zoom.set_ylabel("Energy (eV)", fontsize=10, labelpad=2)
    title_zoom = f"{z_lo:.2f}–{z_hi:.2f} eV";
    if direction is not None: title_zoom += f"  [{direction}]"
    ax_zoom.set_title(title_zoom, fontsize=10)
    ax_zoom.set_ylim((z_lo, z_hi));
    ax_zoom.margins(x=0.0, y=0.0)
    _style_axes(ax_zoom, fontsize=10, tickpad=4)
    cbar_z = fig.colorbar(im_zoom, ax=ax_zoom, fraction=0.046, pad=0.02)
    cbar_z.ax.tick_params(labelsize=10, pad=3);
    cbar_z.set_label("Norm. intensity (zoom)", fontsize=10, labelpad=6)

    if save_path:
        fig.savefig(save_path, dpi=600);
        plt.close(fig)
    else:
        plt.show()


# REPLACE the whole function
def plot_three_panel_diff(
        Iq, DOS, DIFF, info, view_range=(-2.0, 4.0), zoom_range=(0.0, 3.5),
        save_path=None, k0_label="Γ", kend_label=None,
        mat_name=None, direction=None, hatch="//", use_hatch=True,
        hide_y_numbers=False, dos_theme="red", ed_theme="blue", size_cm=None
):
    import numpy as np
    from matplotlib.colors import Normalize
    import matplotlib.pyplot as plt

    E_grid = info["E_grid"];
    Emin_band = info["Emin_band"];
    Emax_band = info["Emax_band"]
    kmin = info["kmin"];
    kmax = info["kmax"];
    q = info["q"];
    q_label = info["q_label"];
    q_pitch = info["q_pitch"]

    # energy axis
    E_band_grid = np.linspace(Emin_band, Emax_band, DIFF.shape[1])

    # view/zoom mask
    e0, e1 = float(view_range[0]), float(view_range[1])
    e_lo = max(min(e0, e1), min(Emin_band, Emax_band));
    e_hi = min(max(e0, e1), max(Emin_band, Emax_band))
    m_main = (E_band_grid >= e_lo) & (E_band_grid <= e_hi)
    if not np.any(m_main): m_main = slice(None)

    z0, z1 = float(zoom_range[0]), float(zoom_range[1])
    z_lo = max(min(z0, z1), min(Emin_band, Emax_band));
    z_hi = min(max(z0, z1), max(Emin_band, Emax_band))
    m_zoom = (E_band_grid >= z_lo) & (E_band_grid <= z_hi)
    if not np.any(m_zoom): m_zoom = slice(None)

    # normalization (separate vmin~vmax for main/zoom)
    vmin_main = 0;
    vmax_main = 1
    vmin_zoom = float(np.nanmin(DIFF[:, m_zoom]));
    vmax_zoom = float(np.nanmax(DIFF[:, m_zoom]))
    norm_main = Normalize(vmin=vmin_main, vmax=vmax_main)
    norm_zoom = Normalize(vmin=vmin_zoom, vmax=vmax_zoom)

    # === Figure ===
    figsize = _cm2inch(*size_cm) if size_cm else (11, 4)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.07, hspace=0.02)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 0.8], height_ratios=[1, 1, 1])

    # DOS (fixed RED)
    ax_dos = fig.add_subplot(gs[0, 0])
    color = DOS_PALETTES.get(dos_theme, DOS_PALETTES["red"])
    y = np.nanmean(DOS, axis=0) if DOS.ndim > 1 else DOS
    ax_dos.set_xlim(view_range)
    yref = y[(E_grid >= view_range[0]) & (E_grid <= view_range[1])] if y.size else y
    if yref.size:
        y0 = 0.0 if float(np.nanmin(yref)) >= 0 else float(np.nanmin(yref))
        ax_dos.set_ylim(y0, float(np.nanmax(yref)) * 1.05)
    dos_plot_curve(ax_dos, E_grid, y, view_range=view_range, swap_xy=False,
                   hatch=hatch, use_hatch=use_hatch, color=color,
                   xlabel="Energy (eV)", ylabel="DOS (a.u.)",
                   fontsize=10, tickpad=5, normalize=False,
                   hide_y_numbers=hide_y_numbers, spectrum_only=False)

    # ED (fixed BLUE)
    ax_dif = fig.add_subplot(gs[1, 0])
    ed_color = ED_PALETTES.get(ed_theme, ED_PALETTES["blue"])
    ed_plot_curve(ax_dif, q, Iq, q_pitch=q_pitch, color=ed_color,
                  hatch=hatch, use_hatch=use_hatch, fontsize=10, tickpad=5,
                  hide_y_numbers=hide_y_numbers, spectrum_only=False)
    ax_dif.set_xlabel(q_label, fontsize=12, labelpad=1)
    ax_dif.grid(True, which="major", axis="y", alpha=0.2, linestyle="-", linewidth=0.8)

    # Band diff panels — use ebs_plot colormap
    newcmp = build_arpes_cmap()  # same cmap as ebs
    # main
    ax_main = fig.add_subplot(gs[0:2, 1])
    im_main = ax_main.imshow(
        DIFF.T, origin="lower", aspect="auto",
        extent=(kmin, kmax, Emin_band, Emax_band),
        cmap=newcmp, interpolation="nearest", norm=norm_main
    )
    if kend_label is None and (direction is not None): kend_label = direction
    kticks, klabels = build_k_ticks_labels(info, start_label=k0_label, end_label=kend_label or "end")
    ax_main.set_xticks(kticks);
    ax_main.set_xticklabels(klabels)
    xlab = "k-path";
    if direction is not None: xlab = f"k-path  {k0_label} → {direction}"
    ax_main.set_xlabel(xlab, fontsize=12, labelpad=2)
    ax_main.set_ylabel("Energy (eV)", fontsize=12, labelpad=2)
    ax_main.set_title("Δ band (GT − Pred)", fontsize=10)
    ax_main.set_ylim(view_range);
    ax_main.margins(x=0.0, y=0.0)
    _style_axes(ax_main, fontsize=10, tickpad=5)
    cbar = fig.colorbar(im_main, ax=ax_main, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=10, pad=4);
    cbar.set_label("Δ intensity (view-range)", fontsize=10, labelpad=8)

    # zoom
    ax_zoom = fig.add_subplot(gs[2, 1])
    im_zoom = ax_zoom.imshow(
        DIFF.T, origin="lower", aspect="auto",
        extent=(kmin, kmax, Emin_band, Emax_band),
        cmap=newcmp, interpolation="nearest", norm=norm_zoom
    )
    ax_zoom.set_xticks(kticks);
    ax_zoom.set_xticklabels(klabels)
    ax_zoom.set_xlabel("k-path", fontsize=10, labelpad=2)
    ax_zoom.set_ylabel("Energy (eV)", fontsize=12, labelpad=2)
    ax_zoom.set_title(f"Δ zoom  {z_lo:.2f}–{z_hi:.2f} eV", fontsize=10)
    ax_zoom.set_ylim((z_lo, z_hi));
    ax_zoom.margins(x=0.0, y=0.0)
    _style_axes(ax_zoom, fontsize=10, tickpad=4)
    cbar_z = fig.colorbar(im_zoom, ax=ax_zoom, fraction=0.046, pad=0.02)
    cbar_z.ax.tick_params(labelsize=10, pad=3);
    cbar_z.set_label("Δ intensity (zoom)", fontsize=10, labelpad=6)

    if save_path:
        fig.savefig(save_path, dpi=600);
        plt.close(fig)
    else:
        plt.show()


@torch.no_grad()
def collect_ddim_traces(
        ldm, sched, dos, ed, device,
        band_shape, downsample, z_ch,
        steps=50, eta=0.0, guidance=1.0, latent_scale=1.0, predict_type="v",
        trace_steps=(0, 10, 20, 30, 40, 50),
):
    """
    For specified step(i) in the DDIM loop,
      - xt_dec: the band obtained by decoding the current z_t (x_t)
      - x0_dec: the band obtained by decoding the current step's x̂0 (denoised)
    are collected and returned, respectively.
    """
    B = dos.size(0);
    assert B == 1, "trace assumes batch=1"
    nk, nE = band_shape
    H = nk // downsample;
    W = nE // downsample
    z_t = torch.randn(B, z_ch, H, W, device=device)

    # organize indices to collect
    want = set(int(x) for x in trace_steps if x >= 0)
    want.discard(0)  # i=0 will be saved as the state right after the first step, not before entering the loop
    want = sorted(want)
    last_i = steps  # label the 'final' as steps

    t_list = ddim_timesteps(len(sched.betas), steps)
    xt_snaps = {}  # i -> np.ndarray (nK,nE)
    x0_snaps = {}  # i -> np.ndarray (nK,nE)

    for i, t_scalar in enumerate(t_list):  # i: 0..steps-1
        t = torch.full((B,), t_scalar, device=device, dtype=torch.long)

        # predict v or eps
        if guidance != 1.0:
            v_u = ldm(z_t, t, dos, ed, uncond=True)
            v_c = ldm(z_t, t, dos, ed, uncond=False)
            v_hat = v_u + guidance * (v_c - v_u)
        else:
            v_hat = ldm(z_t, t, dos, ed, uncond=False)

        alpha_t = sched.sqrt_alphas_cumprod[t_scalar]
        sigma_t = sched.sqrt_one_minus_alphas_cumprod[t_scalar]
        if predict_type.lower() == "v":
            x0_hat = alpha_t * z_t - sigma_t * v_hat
            eps_hat = sigma_t * z_t + alpha_t * v_hat
        else:
            eps_hat = v_hat
            x0_hat = (z_t - sigma_t * eps_hat) / (alpha_t + 1e-12)

        # save snapshot ("right after" step i)
        if i in want:
            # decode x_t
            xt_dec = ldm.vae.decode((z_t / latent_scale)).clamp(-1, 1)
            xt_np = ((xt_dec + 1.0) / 2.0)[0, 0].detach().cpu().numpy()
            xt_snaps[i] = xt_np
            # decode x̂0
            x0_dec = ldm.vae.decode((x0_hat / latent_scale)).clamp(-1, 1)
            x0_np = ((x0_dec + 1.0) / 2.0)[0, 0].detach().cpu().numpy()
            x0_snaps[i] = x0_np

        # if last, end with x̂0
        if i == len(t_list) - 1:
            z_t = x0_hat
            break

        # DDIM update
        t_prev = t_list[i + 1]
        abar_t = sched.alphas_cumprod[t_scalar]
        abar_prev = sched.alphas_cumprod[t_prev]
        sigma_ddim = eta * torch.sqrt((1 - abar_prev) / (1 - abar_t)) * torch.sqrt(1 - abar_t / abar_prev)
        noise = torch.randn_like(z_t) if eta > 0 else 0.0
        z_t = torch.sqrt(abar_prev) * x0_hat + torch.sqrt(
            1 - abar_prev - sigma_ddim ** 2) * eps_hat + sigma_ddim * noise

    # if final(i=steps) snapshot is requested
    if last_i in want or (len(trace_steps) and trace_steps[-1] == last_i):
        final_dec = ldm.vae.decode((z_t / latent_scale)).clamp(-1, 1)
        final_np = ((final_dec + 1.0) / 2.0)[0, 0].detach().cpu().numpy()
        xt_snaps[last_i] = final_np
        x0_snaps[last_i] = final_np  # at the final state, x_t==x̂0

    return xt_snaps, x0_snaps


def plot_ebs_grid(snaps: dict, info, view_range, save_path, title=None, size_cm=None):
    """
    snaps: {i: (nK,nE) ndarray}  (i is the step index)
    """
    if not snaps: return
    newcmp = build_arpes_cmap()
    kmin, kmax = info["kmin"], info["kmax"]
    Emin, Emax = info["Emin_band"], info["Emax_band"]
    nk = len(snaps)
    cols = len(sorted(snaps.keys()))
    figsize = _cm2inch(*size_cm) if size_cm else (max(8, 4 * cols), 3.8)

    # common vmin/vmax within the view-range
    keys = sorted(snaps.keys())
    e_grid = np.linspace(Emin, Emax, list(snaps.values())[0].shape[1])
    m = (e_grid >= min(view_range)) & (e_grid <= max(view_range))
    vmin = min(np.nanmin(snaps[k][:, m]) for k in keys)
    vmax = max(np.nanmax(snaps[k][:, m]) for k in keys)

    fig, axes = plt.subplots(1, cols, figsize=figsize, constrained_layout=True)
    if cols == 1: axes = [axes]

    for ax, i in zip(axes, keys):
        B = snaps[i]
        im = ax.imshow(
            B.T, origin="lower", aspect="auto",
            extent=(kmin, kmax, Emin, Emax),
            cmap=newcmp, interpolation="nearest",
            vmin=vmin, vmax=vmax
        )
        ax.set_axis_off()
        # ax.set_ylim(view_range)
        # ax.set_xlabel("k-path", fontsize=10)
        # ax.set_ylabel("Energy (eV)", fontsize=10)
        # ax.set_title(f"t={i}", fontsize=10)
        # _style_axes(ax, fontsize=9, tickpad=3)

    # common colorbar
    # cax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    # fig.colorbar(axes[0].images[0], cax=cax)
    # if title: fig.suptitle(title, fontsize=11, y=1.02)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _latent_to_2d(zchw: torch.Tensor, how: str = "l2") -> np.ndarray:
    """
    zchw: (B=1, C, H, W) or (C, H, W). Returns: (H, W) ndarray
    """
    import torch
    if zchw.dim() == 4:
        zchw = zchw[0]
    C, H, W = zchw.shape
    if how == "l2":
        x = torch.sqrt(torch.mean(zchw ** 2, dim=0))
    elif how == "mean":
        x = torch.mean(zchw, dim=0)
    elif how == "absmean":
        x = torch.mean(torch.abs(zchw), dim=0)
    elif how == "ch0":
        x = zchw[0]
    else:
        x = torch.sqrt(torch.mean(zchw ** 2, dim=0))
    return x.detach().cpu().numpy()


@torch.no_grad()
def collect_ddim_latent_traces(
        ldm, sched, dos, ed, device,
        band_shape, downsample, z_ch,
        steps=50, eta=0.0, guidance=1.0, latent_scale=1.0, predict_type="v",
        trace_steps=(0, 10, 20, 30, 40, 50),
):
    """
    For specified t ∈ [0..steps] in the DDIM reverse process,
      - zt_snaps[t] : current latent z_t  (C,H,W)
      - z0_snaps[t] : current step's ẑ0   (C,H,W)  (x̂0 in latent)
    are returned. (t=steps is the initial noise, t=0 is the final state)
    """
    B = dos.size(0);
    assert B == 1
    nk, nE = band_shape
    H = nk // downsample;
    W = nE // downsample

    want_t = sorted(set(int(x) for x in trace_steps if 0 <= int(x) <= int(steps)))

    # initial noise z_T
    z_t = torch.randn(B, z_ch, H, W, device=device)

    # DDIM timestep indices (large→small)
    t_idx_list = ddim_timesteps(len(sched.betas), steps)

    zt_snaps, z0_snaps = {}, {}

    def abar(ts):
        return sched.alphas_cumprod[ts]

    def c0(ts):
        return torch.sqrt(abar(ts))

    def c1(ts):
        return torch.sqrt(1.0 - abar(ts))

    # if t=steps saving is requested (initial noise)
    if steps in want_t:
        zt_snaps[steps] = z_t.clone()  # (1,C,H,W)

    for i, t_scalar in enumerate(t_idx_list):  # i: 0..steps-1
        t = torch.full((B,), t_scalar, device=device, dtype=torch.long)

        # classifier-free guidance
        if guidance != 1.0:
            pred_u = ldm(z_t, t, dos, ed, uncond=True)
            pred_c = ldm(z_t, t, dos, ed, uncond=False)
            pred = pred_u + guidance * (pred_c - pred_u)
        else:
            pred = ldm(z_t, t, dos, ed, uncond=False)

        ct0, ct1 = c0(t_scalar), c1(t_scalar)

        # === parameterization conversion ===
        pt = predict_type.lower()
        if pt == "v":
            x0_hat = ct0 * z_t - ct1 * pred
            eps_hat = ct1 * z_t + ct0 * pred
        elif pt in ("eps", "epsilon"):
            eps_hat = pred
            x0_hat = (z_t - ct1 * eps_hat) / (ct0 + 1e-12)
        elif pt in ("x0", "x_start"):
            x0_hat = pred
            eps_hat = (z_t - ct0 * x0_hat) / (ct1 + 1e-12)
        else:
            raise ValueError(f"Unknown predict_type={predict_type}")

        # current 'physical t' (0..steps)
        t_curr = steps - i

        # save snapshot
        if t_curr in want_t:
            zt_snaps[t_curr] = z_t.clone()  # (1,C,H,W)
            z0_snaps[t_curr] = x0_hat.clone()  # (1,C,H,W)

        # if last, end with ẑ0
        if i == len(t_idx_list) - 1:
            z_t = x0_hat
            break

        # DDIM update
        t_prev = t_idx_list[i + 1]
        abar_t, abar_prev = abar(t_scalar), abar(t_prev)
        sigma_ddim = eta * torch.sqrt((1 - abar_prev) / (1 - abar_t)) * torch.sqrt(1 - abar_t / abar_prev)
        noise = torch.randn_like(z_t) if eta > 0 else 0.0
        z_t = torch.sqrt(abar_prev) * x0_hat + torch.sqrt(
            1 - abar_prev - sigma_ddim ** 2) * eps_hat + sigma_ddim * noise

    # force save t=0
    if 0 in want_t and 0 not in zt_snaps:
        zt_snaps[0] = z_t.clone()
        z0_snaps[0] = z_t.clone()

    return zt_snaps, z0_snaps


def plot_latent_grid(snaps: dict, info, view_range, save_path, title=None, size_cm=None,
                     aggregate="ch0"):
    import numpy as np
    import matplotlib.pyplot as plt

    if not snaps: return
    kmin, kmax = info["kmin"], info["kmax"]
    Emin, Emax = info["Emin_band"], info["Emax_band"]

    keys = sorted(snaps.keys())
    mats = [_latent_to_2d(snaps[t], aggregate) for t in keys]

    # common vmin/vmax
    vmin = min(float(np.nanmin(m)) for m in mats)
    vmax = max(float(np.nanmax(m)) for m in mats)

    cols = len(keys)
    figsize = _cm2inch(*size_cm) if size_cm else (max(8, 2.2 * cols), 3.8)

    # ❗ do not use add_axes, also disable constrained_layout so fig.colorbar reserves the margin
    fig, axes = plt.subplots(1, cols, figsize=figsize,
                             gridspec_kw={"wspace": 0.02}, constrained_layout=False)
    if cols == 1:
        axes = [axes]

    ims = []
    for ax, t, M in zip(axes, keys, mats):
        im = ax.imshow(
            M.T, origin="lower", aspect="auto",
            extent=(kmin, kmax, Emin, Emax),
            cmap="twilight_shifted", interpolation="nearest",
            vmin=vmin, vmax=vmax
        )
        ims.append(im)
        ax.set_axis_off()
        # if needed, only view the visible energy range:
        # ax.set_ylim(view_range)

    # ✅ automatically reserve colorbar space by passing all data axes
    cb = fig.colorbar(ims[0], ax=axes, location="right", fraction=0.035, pad=0.02)
    # cb.set_label("Intensity")  # if a label is needed

    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def _encode_band_to_latent(vae, band_np: np.ndarray, device, latent_scale: float):
    x = torch.from_numpy(band_np).float().to(device)
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    x = (x.clamp(0, 1) * 2.0) - 1.0

    # ✅ unpack encode return value + deterministic encoding
    z0, _ = vae.encode(x, sample_posterior=False)  # z0: (1,C,H',W')
    z0 = z0 * float(latent_scale)
    return z0


@torch.no_grad()
def collect_forward_latent_traces(vae, band_np, sched, device,
                                  steps=50, trace_steps=(0, 10, 20, 30, 40, 50),
                                  seed=123, latent_scale=1.0):
    """
    z0 = VAE.encode(x0); z_t = sqrt(abar_t)*z0 + sqrt(1-abar_t)*eps
    trace_steps: 0..steps (t=0 is z0, t=steps is the noisiest)
    """
    z0 = _encode_band_to_latent(vae, band_np, device, latent_scale)  # (1,C,H,W)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    eps = torch.randn(z0.shape, dtype=z0.dtype, device=z0.device, generator=gen)

    # DDIM schedule t_scalar mapping: t_curr(0..steps) ↔ t_scalar
    t_idx_list = ddim_timesteps(len(sched.betas), steps)  # length=steps, large→small
    tmap = {0: None}  # t=0 is z0 itself
    for i, t_scalar in enumerate(t_idx_list):
        t_curr = steps - i  # i=0 → t_curr=steps (most noisy)
        tmap[t_curr] = int(t_scalar)

    snaps = {}
    for t in sorted(set(int(x) for x in trace_steps if 0 <= int(x) <= int(steps))):
        if t == 0:
            snaps[t] = z0.clone()
        else:
            ts = tmap.get(t, None)
            if ts is None: continue
            abar_t = sched.alphas_cumprod[ts]
            c0 = torch.sqrt(abar_t);
            c1 = torch.sqrt(1.0 - abar_t)
            z_t = c0 * z0 + c1 * eps
            snaps[t] = z_t.clone()
    return snaps


@torch.no_grad()
def decode_latent_snaps(vae, snaps: dict, latent_scale: float) -> dict:
    out = {}
    for t, z in snaps.items():
        x = vae.decode((z / float(latent_scale))).clamp(-1, 1)
        out[t] = ((x + 1.0) / 2.0)[0, 0].detach().cpu().numpy()
    return out


# --------------------------- Pred ---------------------------
def predict_one(dir_path: str, cfg, ldm, vae, sched, device, steps=50, guidance=1.0):
    ds = BandDataset([dir_path], band_shape=cfg.BAND_SHAPE, dos_len=cfg.DOS_LEN, ed_len=cfg.ED_LEN)
    item = ds[0]
    dos = item["dos"].unsqueeze(0).to(device);
    ed = item["ed"].unsqueeze(0).to(device)
    x0 = item["band"].unsqueeze(0).to(device)

    with torch.no_grad():
        cond_model = str(getattr(cfg, "cond_model", "diffusion")).lower()
        if cond_model == "simple":
            _ensure_simple_encoders(ldm, cfg, device)
            pred = ldm(dos, ed, uncond=False).clamp(-1, 1)
            pred01 = (pred + 1.0) / 2.0
        else:
            pred01 = sample_ddim(ldm, dos, ed, sched, device,
                                 band_shape=cfg.BAND_SHAPE, downsample=cfg.downsample, z_ch=cfg.z_ch,
                                 steps=max(1, int(steps)), eta=getattr(cfg, "preview_eta", 0.0), guidance=float(guidance),
                                 latent_scale=float(cfg.latent_scale), predict_type=cfg.predict_type)
        gt01 = (x0.clamp(-1, 1) + 1.0) / 2.0
        mae = float(torch.mean(torch.abs(pred01 - gt01)).item())

    pred = pred01[0, 0].detach().cpu().numpy();
    gt = gt01[0, 0].detach().cpu().numpy()
    return pred, gt, mae, dos, ed


# --------------------------- Main ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("id_or_name", help="runs/<run_name> or 8-character hash")
    ap.add_argument("--out_root", default="runs");
    ap.add_argument("--prefix", default="exp")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "new"])
    # new mode: specify directories directly
    ap.add_argument("--dirs", nargs="+", help="List of mp/*/ebs_* directories to specify directly (space-separated)")
    # old mode: {split}.txt + --idx
    ap.add_argument("--idx", nargs="+", type=int, help="Selected indices (space-separated)")
    # plot parameters
    ap.add_argument("--dos_range", type=float, nargs=2, default=[-10.0, 5.0])
    ap.add_argument("--band_range", type=float, nargs=2, default=[-5.0, 5.0])
    ap.add_argument("--view_range", type=float, nargs=2, default=[-2.0, 4.0])
    ap.add_argument("--two_theta_range", type=float, nargs=2, default=[0.05, 4.3])
    ap.add_argument("--accel_kV", type=float, default=200.0)
    ap.add_argument("--zoom_range", type=float, nargs=2, default=[0.0, 3.5])
    ap.add_argument("--k0_000", action="store_true", help="Set k-path start label to '000' (default: Γ)")
    # theme options
    ap.add_argument("--theme-color", choices=("blue", "red"), default="blue")
    ap.add_argument("--hatch", default="//");
    ap.add_argument("--no-hatch", action="store_true")
    ap.add_argument("--hide-y-numbers", action="store_true")
    # LDM options
    ap.add_argument("--ldm_ckpt", default=None);
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=1.0);
    ap.add_argument("--use_ema", action="store_true")
    # add to argparse settings
    ap.add_argument("--size-cm", type=float, nargs=2, metavar=("W", "H"),
                    help="Figure size in centimeters (width height)")
    # DOS/ED Gaussian options
    ap.add_argument("--dos-gauss-points", type=float, default=None,
                    help="DOS Gaussian sigma in sample points")
    ap.add_argument("--dos-fwhm", type=float, default=None,
                    help="DOS Gaussian FWHM in eV (converted to sigma points)")
    ap.add_argument("--ed-gauss-points", type=float, default=None,
                    help="ED Gaussian sigma in sample points")
    ap.add_argument("--ed-fwhm", type=float, default=None,
                    help="ED Gaussian FWHM in Å^-1 (converted to sigma points)")
    # Trace options (save intermediate results during sampling/denoising)
    ap.add_argument("--trace", action="store_true",
                    help="Save 2x3 grid plots for DDIM reverse sampling and x0_hat(denoising) intermediate stages(t)")
    ap.add_argument("--trace-steps", type=int, nargs="+",
                    default=[0, 10, 20, 30, 40, 50],
                    help="Step indices to display (e.g., 0 10 20 30 40 50). If steps=S, automatically mapped on the 0..S scale")
    ap.add_argument("--latent-aggregate", choices=["l2", "mean", "absmean", "ch0"],
                    default="ch0", help="How to compress z_t (C,H,W) → (H,W)")
    ap.add_argument("--forward", action="store_true",
                    help="Save forward(q-sample) latent z_t snapshots")
    ap.add_argument("--forward-from", choices=["gt", "pred"], default="gt",
                    help="Forward starting x0: GT(ebs.npy) or Pred")
    ap.add_argument("--forward-steps", type=int, nargs="+",
                    default=[0, 10, 20, 30, 40, 50],
                    help="Forward t indices to save (0..steps)")
    ap.add_argument("--forward-seed", type=int, default=123,
                    help="Forward noise epsilon seed (reproducibility)")
    ap.add_argument("--forward-decode", action="store_true",
                    help="Also save the EBS grid obtained by decoding forward z_t")
    ap.add_argument("--tsize-cm", type=float, nargs=2, metavar=("W", "H"),
                    help="Figure size in centimeters (width height)")

    args = ap.parse_args()

    run_name = resolve_run_name(args.id_or_name, prefix=args.prefix)
    base_run_name = resolve_base_run_name(run_name)
    finetune_run = resolve_latest_finetune_run(args.out_root, base_run_name)
    ldm_run_name = finetune_run or run_name
    run_root = os.path.join(args.out_root, run_name)

    # prepare target directories
    dirs = []
    if args.dirs:
        for d in args.dirs:
            if os.path.isdir(d):
                dirs.append(d)
            else:
                print(f"[WARN] No such directory: {d} → skipping")
        if not dirs: raise SystemExit("[ERR] No valid directories.")
    else:
        ds_txt = os.path.join(run_root, "datasets", f"{args.split}.txt")
        if not os.path.isfile(ds_txt):
            raise SystemExit(f"[ERR] datasets file does not exist: {ds_txt}")
        lines = read_list(ds_txt)
        if not lines: raise SystemExit(f"[ERR] {ds_txt} is empty.")
        if not args.idx: raise SystemExit("[ERR] One of --dirs or --idx is required.")
        for i in args.idx:
            if 0 <= i < len(lines):
                dirs.append(lines[i])
            else:
                print(f"[WARN] Index out of range: {i} (0~{len(lines) - 1}) → skipping")
        if not dirs: raise SystemExit("[ERR] No valid indices.")

    # load model
    device = device_auto()
    ldm_ckpt = args.ldm_ckpt or resolve_ckpt(args.out_root, ldm_run_name, kind="ldm")
    print(ldm_ckpt)
    ck = torch.load(ldm_ckpt, map_location=device)
    meta = ck.get('meta', None)
    if meta is None: raise SystemExit("[ERR] LDM ckpt has no meta (datasets/run_root/vae_ckpt).")
    cfg = TrainConfig(**meta.get('config', {}))
    cond_model = str(getattr(cfg, "cond_model", "diffusion")).lower()

    vae, ldm, sched = build_models(cfg, device)
    if cond_model != "simple":
        vae_ckpt = resolve_ckpt(args.out_root, base_run_name, kind="vae")
        ck_v = torch.load(vae_ckpt, map_location=device)
        vae.load_state_dict(ck_v['vae_state'] if (isinstance(ck_v, dict) and 'vae_state' in ck_v) else ck_v, strict=False)

    ldm.load_state_dict(ck['ldm_state'], strict=False)
    if args.use_ema and ck.get('ema', None) is not None:
        e = EMA(ldm);
        e.shadow = ck['ema'];
        e.copy_to(ldm)
    if vae is not None:
        vae.eval()
    ldm.eval()

    if cond_model == "simple" and (args.trace or args.forward):
        raise SystemExit("[ERR] --trace/--forward options are supported only in diffusion mode.")

    out_dir = ensure_out_dir(args.out_root, run_name)
    k0_label = "000" if args.k0_000 else "Γ"
    theme_color = args.theme_color;
    use_hatch = (not args.no_hatch)

    for d in dirs:
        mat_name, direction = parse_material_and_direction(d)
        disp_dir = kpath_to_symbol(direction)
        # load input/GT
        Iq, DOS, Bgt, info = load_inputs(
            d, dos_range=tuple(args.dos_range), band_range=tuple(args.band_range),
            two_theta_range=tuple(args.two_theta_range), accel_kV=float(args.accel_kV)
        )

        # save GT
        save_gt = os.path.join(out_dir, f"GT_{args.split}_{mat_name}_{direction}.png")
        plot_three_panel_thematic(
            Iq, DOS, Bgt, info,
            view_range=tuple(args.view_range), zoom_range=tuple(args.zoom_range),
            save_path=save_gt, title_band="Band map (ground truth)",
            k0_label=k0_label, kend_label=disp_dir,  # ← here
            mat_name=mat_name, direction=disp_dir,  # ← and here
            hatch=args.hatch, use_hatch=use_hatch,
            hide_y_numbers=args.hide_y_numbers, size_cm=args.size_cm,
            dos_gauss_points=args.dos_gauss_points, dos_fwhm=args.dos_fwhm,
            ed_gauss_points=args.ed_gauss_points, ed_fwhm=args.ed_fwhm
        )

        # generate & save Pred
        pred, gt01, mae, dos, ed = predict_one(d, cfg, ldm, vae, sched, device, steps=args.steps,
                                               guidance=args.guidance)

        save_pr = os.path.join(out_dir, f"PRED_{args.split}_{mat_name}_{direction}_L1={mae:.4f}.png")
        plot_three_panel_thematic(
            Iq, DOS, pred, info,
            view_range=tuple(args.view_range), zoom_range=tuple(args.zoom_range),
            save_path=save_pr, title_band=f"Predicted band (L1={mae:.4f})",
            k0_label=k0_label, kend_label=disp_dir,
            mat_name=mat_name, direction=disp_dir,
            hatch=args.hatch, use_hatch=use_hatch,
            hide_y_numbers=args.hide_y_numbers, size_cm=args.size_cm,
            dos_gauss_points=args.dos_gauss_points, dos_fwhm=args.dos_fwhm,
            ed_gauss_points=args.ed_gauss_points, ed_fwhm=args.ed_fwhm
        )

        # NEW: also save GT − PRED difference
        diff = np.abs(Bgt - pred)
        save_df = os.path.join(out_dir, f"DIFF_{args.split}_{mat_name}_{direction}_L1={mae:.4f}.png")
        plot_three_panel_diff(
            Iq, DOS, diff, info,
            view_range=tuple(args.view_range), zoom_range=tuple(args.zoom_range),
            save_path=save_df, k0_label=k0_label, kend_label=disp_dir,
            mat_name=mat_name, direction=disp_dir,
            hatch=args.hatch, use_hatch=(not args.no_hatch),
            hide_y_numbers=args.hide_y_numbers,
            dos_theme="red", ed_theme="blue", size_cm=args.size_cm
        )

        print(
            f"[OK] dir={d} → GT:{os.path.relpath(save_gt)} | Pred:{os.path.relpath(save_pr)} | Diff:{os.path.relpath(save_df)}")

        # ---- DDIM trace (x_t / x̂0) ----
        if args.trace:
            # clean trace indices (0..steps, last=steps)
            trace_idx = sorted(set(max(0, min(args.steps, int(t))) for t in args.trace_steps))
            xt_snaps, x0_snaps = collect_ddim_traces(
                ldm, sched, torch.from_numpy(DOS).float().unsqueeze(0).unsqueeze(0).to(device) if False else dos,
                # dos/ed are already tensors above in predict_one
                ed, device,
                band_shape=cfg.BAND_SHAPE, downsample=cfg.downsample, z_ch=cfg.z_ch,
                steps=args.steps, eta=getattr(cfg, "preview_eta", 0.0), guidance=float(args.guidance),
                latent_scale=float(cfg.latent_scale), predict_type=cfg.predict_type,
                trace_steps=trace_idx
            )

            zt_snaps, z0_snaps = collect_ddim_latent_traces(
                ldm, sched, dos, ed, device,
                band_shape=cfg.BAND_SHAPE, downsample=cfg.downsample, z_ch=cfg.z_ch,
                steps=args.steps, eta=getattr(cfg, "preview_eta", 0.0), guidance=float(args.guidance),
                latent_scale=float(cfg.latent_scale), predict_type=cfg.predict_type,
                trace_steps=trace_idx
            )

            # save (EBS-only grid)
            base = f"{args.split}_{mat_name}_{direction}"
            save_xt = os.path.join(out_dir, f"TRACE_sampling_xt_{base}.png")
            save_x0 = os.path.join(out_dir, f"TRACE_denoise_x0hat_{base}.png")
            plot_ebs_grid(xt_snaps, info, view_range=tuple(args.view_range), save_path=save_xt,
                          title="DDIM sampling: decoded x_t", size_cm=args.tsize_cm)
            plot_ebs_grid(x0_snaps, info, view_range=tuple(args.view_range), save_path=save_x0,
                          title="DDIM denoising: decoded x̂0", size_cm=args.tsize_cm)

            print(f"[TRACE] saved: {os.path.relpath(save_xt)} | {os.path.relpath(save_x0)}")

            save_z_t = os.path.join(out_dir, f"TRACE_sampling_zt_{base}.png")
            save_z0 = os.path.join(out_dir, f"TRACE_denoise_z0hat_{base}.png")

            plot_latent_grid(zt_snaps, info, view_range=tuple(args.view_range),
                             save_path=save_z_t, title="DDIM sampling: latent z_t",
                             size_cm=args.tsize_cm, aggregate=args.latent_aggregate)
            plot_latent_grid(z0_snaps, info, view_range=tuple(args.view_range),
                             save_path=save_z0, title="DDIM denoising: latent ẑ0",
                             size_cm=args.tsize_cm, aggregate=args.latent_aggregate)

            print(f"[TRACE] saved: {os.path.relpath(save_z_t)} | {os.path.relpath(save_z0)}")

        # ---- Forward q-sample (latent z_t) ----
        if args.forward:
            base_kind = args.forward_from.upper()  # "GT" or "PRED"
            base_map = Bgt if (args.forward_from == "gt") else pred  # (nK,nE) in [0,1]

            zt_fwd = collect_forward_latent_traces(
                vae, base_map, sched, device,
                steps=args.steps, trace_steps=args.forward_steps,
                seed=args.forward_seed, latent_scale=float(cfg.latent_scale)
            )

            base = f"{args.split}_{mat_name}_{direction}_{base_kind}"
            save_zt = os.path.join(out_dir, f"TRACE_forward_zt_{base}.png")
            plot_latent_grid(zt_fwd, info, view_range=tuple(args.view_range),
                             save_path=save_zt, title=f"Forward q-sample (latent z_t) from {base_kind}",
                             size_cm=args.tsize_cm, aggregate=args.latent_aggregate)

            if args.forward_decode:
                xt_fwd = decode_latent_snaps(vae, zt_fwd, float(cfg.latent_scale))
                save_xt = os.path.join(out_dir, f"TRACE_forward_xt_{base}.png")
                plot_ebs_grid(xt_fwd, info, view_range=tuple(args.view_range),
                              save_path=save_xt, title=f"Forward q-sample decoded x_t from {base_kind}",
                              size_cm=args.tsize_cm)
                print(f"[FORWARD] saved: {os.path.relpath(save_zt)} | {os.path.relpath(save_xt)}")
            else:
                print(f"[FORWARD] saved: {os.path.relpath(save_zt)}")


if __name__ == "__main__":
    main()
