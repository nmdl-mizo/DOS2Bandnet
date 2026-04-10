#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing utilities for DOS2Bandnet.

This module unifies:
  - PDOS: projected density of states handling from VASP outputs
  - PBand: band structure + fat band handling from VASP outputs
  - ElectronDiffraction1D: 1D electron diffraction pattern along a given zone axis
"""

from __future__ import annotations

import sys, os
import pickle
from typing import Iterable, List, Optional, Sequence, Tuple, Union
from functools import reduce

from math import gcd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp.outputs import Vasprun, Locpot
from pymatgen.electronic_structure.core import OrbitalType, Spin
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core import Structure

class PDOS:
    def __init__(self, dir_path="./"):
        self.dir_path = dir_path
        self.projected = None

    def read_vasprun(self, filename='vasprun.xml', align_efermi=True):
        # Load from pickle if available
        pklfile = os.path.join(self.dir_path, "dos.pkl")
        if os.path.isfile(pklfile):
            print(f"Found existing {os.path.abspath(pklfile)}. Loading data...")
            with open(pklfile, 'rb') as f:
                self.structure, self.dos = pickle.load(f)

        # 2) Otherwise read vasprun.xml, compute, then cache
        else:
            vasprun_file = os.path.join(self.dir_path, filename)
            self.vasprun = Vasprun(vasprun_file)
            self.structure = self.vasprun.final_structure
            self.dos = self.vasprun.complete_dos
            with open(pklfile, 'wb') as f:
                pickle.dump((self.structure, self.dos), f)
            print(f"Saved data as {os.path.abspath(pklfile)}")

        if align_efermi is True:
            self.energies = self.dos.energies - self.dos.efermi
            self.efermi = 0
        else:
            self.energies = self.dos.energies
            self.efermi = self.dos.efermi

        self.densities = self.dos.densities[Spin.up]
        self.selected_indices = range(len(self.structure))

    def projection(self, orbitals=None):
        """
        - If orbitals is "auto", infer from each element block; you can also pass ["p", "s"], etc.
        """
        spd_dos_list, symbol_list, orbital_list = [], [], []
        # Collect DOS entries corresponding to selected site indices
        for i,site in enumerate(self.structure):
            # Select orbital channel
            if i in self.selected_indices:
                # print(i, site)
                if orbitals is None:
                    orbital = "all"
                    spd_dos = self.dos.get_site_dos(site)
                elif orbitals == 'auto':
                    orbital = {"s": OrbitalType.s, "p": OrbitalType.p, "d": OrbitalType.d}[site.specie.block.lower()]
                    spd_dos = self.dos.get_site_spd_dos(site)[orbital]
                else:
                    orbital = {"s": OrbitalType.s, "p": OrbitalType.p, "d": OrbitalType.d}[orbitals[i]]
                    spd_dos = self.dos.get_site_spd_dos(site)[orbital]

                spd_dos_list.append(spd_dos.densities[Spin.up])
                symbol_list.append(site.specie.symbol)
                orbital_list.append(orbital)

        self.densities = np.array(spd_dos_list)
        self.energies = np.repeat(self.energies[np.newaxis, :], len(self.selected_indices), axis=0)
        self.projected = True
        self.symbols, self.orbitals = symbol_list, orbital_list
        print(f"Projected DOS for {len(self.selected_indices)} sites")

    def interpolate_dos(self, emin, emax, n):
        """
        Linearly interpolate DOS data on n grid points in the energy range [emin, emax].

        The result is stored back into self.energies and self.densities.
        """
        interp_energy = np.linspace(emin, emax, n)
        interp_density = []

        if self.projected:
            for eng, dens in zip(self.energies, self.densities):
                interp_dens = np.interp(interp_energy, eng, dens)
                interp_density.append(interp_dens)
            self.densities = np.array(interp_density)
            self.energies = np.repeat(interp_energy[np.newaxis, :], len(self.selected_indices), axis=0)
        else:
            interp_dens = np.interp(interp_energy, self.energies, self.densities)
            self.densities = interp_dens
            self.energies = interp_energy


    def gaussian_filter(self, sigma=0.5):
        """
        Apply Gaussian filtering to the interpolated DOS data.
        The default sigma is 0.5.

        The filtered result is stored back into self.densities.
        """

        gauss_density = []

        if self.projected:
            for dens in self.densities:
                gauss_dens = gaussian_filter1d(dens, sigma)
                gauss_density.append(gauss_dens)
            self.densities = np.array(gauss_density)
        else:
            gauss_dens = gaussian_filter1d(self.densities, sigma)
            self.densities = gauss_dens

# =========================
# Band-map generation: EBS.dat → (nK, nE)
# =========================
def build_band_map_from_ebs(ebs_path,
                            emin, emax, nE,
                            nK,
                            sigma_E=0.2,
                            normalize=True):
    """
    Build a 2D intensity map from EBS.dat (columns: k, E[eV], w).
    - Automatically scales aspect ratio using k-span (kmax-kmin) and E-span (emax-emin):
        sigma_E = sigma_k * (E_span / K_span)
    - Output: map with shape (nK, nE)
    """
    if not os.path.isfile(ebs_path):
        raise FileNotFoundError(f"Not found: {ebs_path}")
    data = np.loadtxt(ebs_path)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("EBS.dat must have shape (N,3) with columns (k, E, w).")

    kpt = data[:, 0].astype(float)
    eng = data[:, 1].astype(float)
    wgt = data[:, 2].astype(float)

    kmin, kmax = float(np.min(kpt)), float(np.max(kpt))
    K_span = max(1e-12, kmax - kmin)          # Guard against division by zero
    E_span = max(1e-12, float(emax - emin))
    sigma_k = float(sigma_E) * (K_span / E_span) * 2

    k_grid = np.linspace(kmin, kmax, int(nK))
    E_grid = np.linspace(float(emin), float(emax), int(nE))

    inv2_k = 1.0 / (2.0 * sigma_k * sigma_k)
    inv2_E = 1.0 / (2.0 * sigma_E * sigma_E)

    # Accumulate intensity
    I = np.zeros((int(nK), int(nE)), dtype=float)
    # Simple and safe implementation: accumulate outer products point-wise
    for x0, y0, a in zip(kpt, eng, wgt):
        if a <= 0:
            continue
        Gk = np.exp(- (k_grid - x0) ** 2 * inv2_k)  # (nK,)
        Ge = np.exp(- (E_grid - y0) ** 2 * inv2_E)  # (nE,)
        I += a * np.outer(Gk, Ge)

    # Normalize (max=1)
    I = np.nan_to_num(I, nan=0.0, posinf=0.0, neginf=0.0)
    if normalize:
        m = float(np.max(I)) if I.size else 0.0
        if m > 0:
            I = I / m
    return I

class ElectronDiffraction1D:
    """
    1D single-crystal electron-diffraction profile generator along a zone axis [u v w].

    Key points
    - q unit: without 2π (Å^-1)  ⇒  |G| = q_pitch
    - compute(): reflection selection → Gaussian broadening → stores self.q_grid and self.intensity
    - normalize(): max/percentile/integral-based normalization
    - find_effective_first_peak(): on a max=1 normalized spectrum, finds and stores the first peak (highest point of first contiguous region) where I>=thresh (default 0.01)
    - plot(): clean style + dashed guide at q=|G| + compact summary metrics in title
    """

    # ---------- Utilities ----------
    @staticmethod
    def _igcd3(a, b, c):
        return reduce(gcd, (abs(int(a)), abs(int(b)), abs(int(c))))

    @staticmethod
    def _reduce_direction(u, v, w):
        if (u, v, w) == (0, 0, 0):
            raise ValueError("[u v w] = (0,0,0)")
        g = ElectronDiffraction1D._igcd3(u, v, w)
        u0, v0, w0 = u // g, v // g, w // g
        # Normalize sign so the first non-zero component is positive
        for s in (u0, v0, w0):
            if s != 0:
                if s < 0:
                    u0, v0, w0 = -u0, -v0, -w0
                break
        # return (u0, v0, w0)
        return (u, v, w)

    @staticmethod
    def electron_wavelength_angstrom(kV: float) -> float:
        # Relativistic electron de Broglie wavelength (Å)
        h  = 6.62607015e-34
        m  = 9.10938356e-31
        e  = 1.602176634e-19
        c  = 2.99792458e8
        V  = float(kV) * 1e3
        p  = np.sqrt(2*m*e*V*(1.0 + (e*V)/(2*m*c**2)))
        return (h / p) * 1e10

    @staticmethod
    def gaussian(x, mu, fwhm):
        sigma = fwhm / (2*np.sqrt(2*np.log(2)))
        z = (x - mu) / (sigma + 1e-18)
        return np.exp(-0.5 * z*z)

    @staticmethod
    def f_electron(q, Z, model="screened_lorentz", alpha=0.55):
        """
        Approximate electron scattering factor. q uses the non-2π unit (Å^-1).
        - screened_lorentz: f ~ Z / (1 + (q/q0)^2)^2, q0 = alpha * Z^(1/3)
        - gaussian:        f ~ Z * exp(-(q/qd)^2),     qd = 2*alpha * Z^(1/3)
        """
        Z = float(Z)
        if model == "screened_lorentz":
            q0 = alpha * (Z ** (1.0/3.0))
            return Z / (1.0 + (q/q0)**2)**2
        elif model == "gaussian":
            qd = 2.0 * alpha * (Z ** (1.0/3.0))
            return Z * np.exp(-(q/qd)**2)
        else:
            return Z

    # ---------- Construction / setup ----------
    def __init__(self,
                 poscar_path="POSCAR",
                 uvw=(1, 0, 0),
                 accel_kV=200.0,
                 form_factor_model="screened_lorentz",
                 alpha_ff=0.55,
                 B_iso_global=0.6,
                 B_by_element=None,
                 use_Z_weight=True):
        # Load structure
        self.structure = Structure.from_file(poscar_path)
        # Direction
        self.u, self.v, self.w = self._reduce_direction(*uvw)
        # Electron wavelength
        self.accel_kV = float(accel_kV)
        self.lambda_A = self.electron_wavelength_angstrom(self.accel_kV)
        # Compute reciprocal lattice and q_pitch (non-2π unit)
        self._update_reciprocal_and_pitch()

        # Scattering / thermal vibration parameters
        self.form_factor_model = str(form_factor_model)
        self.alpha_ff = float(alpha_ff)
        self.B_iso_global = float(B_iso_global)
        self.B_by_element = dict(B_by_element) if B_by_element else {}
        self.use_Z_weight = bool(use_Z_weight)

        # Output buffers
        self.q_grid = None           # (nq,)
        self.intensity = None        # (nq,)
        self.peaks_q = None          # Reflection positions (n*q_pitch that passed selection rules)
        self.peaks_I = None          # Reflection intensities

        # Analysis results (requested addition)
        self.first_peak_q_true = None     # 'True first reflection' from structure factors (minimum q>0)
        self.first_over_G      = None     # 1st/|G|
        self.eff_first_peak_q  = None     # Effective first peak from normalized spectrum
        self.eff_first_over_G  = None     # (eff 1st)/|G|
        self.m_eff             = None     # round(eff 1st / |G|)

    def _update_reciprocal_and_pitch(self):
        # reciprocal_lattice.matrix includes 2π (Å^-1)
        b1, b2, b3 = self.structure.lattice.reciprocal_lattice.matrix
        G_dir_2pi = self.u*b1 + self.v*b2 + self.w*b3
        Gmag_2pi  = np.linalg.norm(G_dir_2pi)
        if Gmag_2pi == 0:
            raise ValueError("The G vector for the given [u v w] direction is zero.")
        # Pitch in non-2π units (1/d_[uvw])
        self.q_pitch = Gmag_2pi / (2*np.pi)       # |G|
        self.bz_edge_q = self.q_pitch             # (Note: in 1D, the FBZ boundary is |G|/2)

    def set_direction(self, uvw):
        self.u, self.v, self.w = self._reduce_direction(*uvw)
        self._update_reciprocal_and_pitch()
        return self

    def set_structure(self, poscar_path):
        self.structure = Structure.from_file(poscar_path)
        self._update_reciprocal_and_pitch()
        return self

    # ---------- Structure factor ----------
    def structure_factor_intensity(self, h: int, k: int, l: int, q: float) -> float:
        """
        I = | Σ_j f_e(q,Z_j) * exp(2πi(hx+ky+lz)) * exp(-B_j (q/2)^2) |^2
        - Handles mixed occupancy: Z_eff = Σ (Z*occ), B_eff = Σ (B*occ)
        """
        amp = 0.0 + 0.0j
        for site in self.structure.sites:
            x, y, z = site.frac_coords
            # Effective Z/B
            Z_eff = 0.0
            B_eff = 0.0
            for sp, occ in site.species.items():
                Z_eff += (sp.Z if self.use_Z_weight else 1.0) * occ
                B_eff += (self.B_by_element.get(sp.symbol, self.B_iso_global)) * occ
            f_j = self.f_electron(q, Z_eff, model=self.form_factor_model, alpha=self.alpha_ff)
            DW  = np.exp(-B_eff * (0.5*q)**2)  # s = q/2
            amp += f_j * DW * np.exp(2j*np.pi*(h*x + k*y + l*z))
        return float((amp.real*amp.real + amp.imag*amp.imag))

    # ---------- Computation ----------
    def compute(self,
                two_theta_range=(0.01, 4.3),   # deg
                q_range=None,                  # (qmin,qmax); takes priority if provided
                n_q=None, dq=0.005,            # Grid definition: n_q or dq
                fwhm_q=0.08,                   # Broadening width (Å^-1)
                normalize_to=None,             # Scale so max equals normalize_to
                collect_peaks=True):
        """
        Compute 1D diffraction-intensity profile and store in self.q_grid / self.intensity.
        """
        # ---- Determine q range ----
        if q_range is not None:
            q_min, q_max = map(float, q_range)
        else:
            lam = self.lambda_A
            tt_min, tt_max = np.radians(two_theta_range[0]), np.radians(two_theta_range[1])
            q_min = (2.0/lam) * np.sin(tt_min/2.0)
            q_max = (2.0/lam) * np.sin(tt_max/2.0)

        if n_q is not None:
            q_grid = np.linspace(q_min, q_max, int(n_q))
        else:
            dq = float(dq)
            q_grid = np.arange(q_min, q_max + 0.5*dq, dq)

        # ---- Select reflections (integer multiples of [u v w])
        n_max = int(np.floor(q_max / self.q_pitch))
        qs_sel, I_sel = [], []
        for n in range(1, n_max + 1):
            h, k, l = n*self.u, n*self.v, n*self.w
            q_n = n * self.q_pitch
            I_n = self.structure_factor_intensity(h, k, l, q=q_n)
            if I_n > 1e-16:
                qs_sel.append(q_n); I_sel.append(I_n)
            # qs_sel.append(q_n); I_sel.append(I_n)

        qs_sel = np.array(qs_sel, float) if qs_sel else np.empty((0,), float)
        I_sel  = np.array(I_sel,  float) if I_sel  else np.empty((0,), float)
        if collect_peaks:
            self.peaks_q, self.peaks_I = qs_sel, I_sel

        # Record 'true first reflection' (if present)
        self.first_peak_q_true = float(qs_sel[0]) if qs_sel.size else np.nan
        self.first_over_G = (self.first_peak_q_true / self.q_pitch) if qs_sel.size else np.nan

        # ---- Accumulate Gaussian broadening
        Iq = np.zeros_like(q_grid, dtype=float)
        if qs_sel.size:
            for q0, I0 in zip(qs_sel, I_sel):
                if (q0 < q_min - 5*fwhm_q) or (q0 > q_max + 5*fwhm_q):
                    continue
                Iq += I0 * self.gaussian(q_grid, q0, fwhm_q)

        # ---- Normalization (optional)
        if normalize_to is not None and Iq.max() > 0:
            Iq = float(normalize_to) * (Iq / Iq.max())

        # Store outputs
        self.q_grid = q_grid
        self.intensity = Iq

        # Also update effective first peak (default thresh=0.01)
        self.find_effective_first_peak(thresh=0.015)

        return self

    # ---------- Normalization ----------
    def normalize(self, mode="max", target=1.0, p=99, eps=1e-12, inplace=True):
        """
        self.intensity 정규화
          - mode='max'  : max=target
          - mode='pXX'  : XX percentile (e.g., 'p99')
          - mode='l1'   : integral (∑ I dq)=target
          - mode='none' : skip
        """
        if self.intensity is None or self.q_grid is None:
            raise RuntimeError("Call this after compute().")

        I = np.asarray(self.intensity, float).copy()
        if mode == "none":
            if inplace: return self
            return I

        if mode == "max":
            s = float(np.nanmax(I))
        elif mode.lower().startswith("p"):
            try:
                pp = float(mode[1:])
            except Exception:
                pp = float(p)
            s = float(np.nanpercentile(I, pp))
        elif mode == "l1":
            dq = np.mean(np.diff(self.q_grid)) if self.q_grid.size > 1 else 1.0
            s = float(I.sum() * dq)
        else:
            raise ValueError("mode must be 'max', 'pXX', 'l1', or 'none'.")

        if not np.isfinite(s) or s <= eps:
            s = 1.0
        I = (target / s) * I

        if inplace:
            self.intensity = I
            # Re-evaluate effective first peak after normalization changes
            self.find_effective_first_peak(thresh=0.01)
            return self
        else:
            return I

    # ---------- Analysis: effective first peak ----------
    def find_effective_first_peak(self, thresh: float = 0.01):
        """
        Based on ED normalized internally to max=1, define/store the effective first peak as the maximum point in the first contiguous region where I_norm >= thresh.
        """
        if self.intensity is None or self.q_grid is None:
            self.eff_first_peak_q = np.nan
            self.eff_first_over_G = np.nan
            self.m_eff = None
            return self

        q = np.asarray(self.q_grid, float)
        I = np.asarray(self.intensity, float)
        if I.size == 0 or np.nanmax(I) <= 0:
            self.eff_first_peak_q = np.nan
            self.eff_first_over_G = np.nan
            self.m_eff = None
            return self

        In = I / np.nanmax(I)
        mask = In >= float(thresh)
        if not np.any(mask):
            self.eff_first_peak_q = np.nan
        else:
            i0 = int(np.argmax(mask))  # First True index
            j = i0
            n = mask.size
            while (j + 1 < n) and mask[j + 1]:
                j += 1
            seg = slice(i0, j + 1)
            imax = int(seg.start + np.argmax(In[seg]))
            self.eff_first_peak_q = float(q[imax])

        # Ratio / integer multiple
        if np.isfinite(self.eff_first_peak_q):
            self.eff_first_over_G = float(self.eff_first_peak_q / self.q_pitch)
            # Use rounding for integer multiple (robust to small broadening offsets)
            self.m_eff = int(np.round(self.eff_first_over_G))
        else:
            self.eff_first_over_G = np.nan
            self.m_eff = None

        return self