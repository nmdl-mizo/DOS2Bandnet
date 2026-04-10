#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset-level preprocessing script for DOS2Bandnet.

Usage (example)
-----------
python ~/_code/DOS2BandNet/scripts/preprocessing.py \
    --base-dir /path/to/data \
    --mp-prefix mp \
    --band-prefix band \
    --tasks diffraction \
    --nproc 1

Features
----
- For mp* directories under base_dir:
  - For each band* directory inside each mp_dir
    - POSCAR + kpts.txt → ElectronDiffraction1D → diffrac.npy
    - vasprun.xml → PDOS → dos.npy
    - vasprun.xml → PBand → ebs.npy
"""

import os, sys, shutil
import argparse
from functools import partial
from math import gcd
from fractions import Fraction
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

import numpy as np

from dos2bandnet.preprocessing import PDOS, build_band_map_from_ebs, ElectronDiffraction1D

# ----------------------------------------------------------------------
# Utilities: directory search / FBZ → [u,v,w]
# ----------------------------------------------------------------------

def get_dirs(prefix: str, d_base: str = ".") -> List[str]:
    """Return a list of full paths of directories under d_base that start with prefix."""
    out = []
    for name in os.listdir(d_base):
        path = os.path.join(d_base, name)
        if os.path.isdir(path) and name.startswith(prefix):
            out.append(path)
    return sorted(out)


def fbz_to_uvw(vec, max_den=24, tol=1e-4):
    """Convert FBZ coordinates (fractional form) to the minimum integer [u v w] with the same direction"""
    v = []
    for x in vec:
        if abs(x - round(x)) < tol: v.append(float(round(x)))
        elif abs(x) < tol:          v.append(0.0)
        else:                       v.append(float(x))
    fracs = [Fraction(x).limit_denominator(max_den) for x in v]
    denoms = [f.denominator for f in fracs]
    lcm = 1
    for d in denoms:
        if d != 0:
            lcm = lcm * d // gcd(lcm, d)
    ints = [int(round(f.numerator * (lcm // f.denominator))) for f in fracs]
    if all(a == 0 for a in ints):
        raise ValueError("The input vector is not distinguishable from (0,0,0).")
    g = 0
    for a in ints:
        g = gcd(g, abs(a))
    if g > 1:
        ints = [a // g for a in ints]
    # for a in ints:
    #     if a != 0:
    #         if a < 0:
    #             ints = [-x for x in ints]
    #         break
    return tuple(ints)

# ----------------------------------------------------------------------
# Process a single band directory
# ----------------------------------------------------------------------

def _process_one_band_dir(
    band_dir: str,
    dos_range: Tuple[float, float],
    dos_grid: int,
    tt_range: Tuple[float, float],
    dif_grid: int,
    fwhm_q: float,
    nk: int,
    nE: int,
    sigma_E_ebs: float,
    etc_root: str,
    run_diffraction: bool,
    run_dos: bool,
    run_ebs: bool,
) -> Tuple[int, int, int]:
    """
    Generate diffrac.npy, dos.npy, and ebs.npy for one band_* directory.

    Return value: (dif_done, dos_done, ebs_done) - each is 0 or 1
    """
    dif_done = dos_done = ebs_done = 0

    config_path = os.path.join(band_dir, "pre_config.txt")
    config_lines = [
        f"band_dir={band_dir}",
        f"run_diffraction={run_diffraction}",
        f"run_dos={run_dos}",
        f"run_ebs={run_ebs}",
        f"dos_emin={dos_range[0]}",
        f"dos_emax={dos_range[1]}",
        f"dos_grid={dos_grid}",
        f"tt_min={tt_range[0]}",
        f"tt_max={tt_range[1]}",
        f"dif_grid={dif_grid}",
        f"fwhm_q={fwhm_q}",
        f"nk={nk}",
        f"nE={nE}",
        f"sigma_E_ebs={sigma_E_ebs}",
    ]
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("\n".join(config_lines) + "\n")
    except OSError as e:
        print(f"  [WARN] failed to write config file in {band_dir}: {e}")

    # ---------------- Diffraction ----------------
    dif_path = os.path.join(band_dir, "diffrac.npy")
    # if not os.path.isfile(dif_path):
    if run_diffraction:
        poscar = os.path.join(band_dir, "POSCAR")
        kpt_file = os.path.join(band_dir, "kpts.txt")

        if os.path.isfile(poscar) and os.path.isfile(kpt_file):
            with open(kpt_file, "r") as f:
                vals = [float(line.strip()) for line in f if line.strip()]
            if len(vals) >= 3:
                uvw = fbz_to_uvw(vals[:3])
                print(f"[{band_dir}] k-vector -> uvw = {uvw}")

                # Check whether all of u,v,w are one of -1,0,1
                # if not all(c in (-1, 0, 1) for c in uvw):
                #     # If the condition is not satisfied, move to etc/
                #     dst = os.path.join(etc_root, os.path.basename(band_dir))
                #     print(f"  invalid uvw {uvw}, move to {dst}")
                #     os.makedirs(os.path.dirname(dst), exist_ok=True)
                #     shutil.move(band_dir, dst)
                #     return (0, 0, 0)

                # Create ElectronDiffraction1D object and compute
                try:
                    d = ElectronDiffraction1D(
                        poscar_path=poscar,
                        uvw=uvw,
                        accel_kV=200.0,
                        form_factor_model="screened_lorentz",
                        alpha_ff=0.55,
                        B_iso_global=0.6,
                        use_Z_weight=True,
                    )
                    tt_min, tt_max = tt_range
                    d.compute(
                        two_theta_range=(tt_min, tt_max),
                        n_q=dif_grid,
                        fwhm_q=fwhm_q,
                        normalize_to=1.0,
                    )
                    I = np.asarray(d.intensity, dtype=float)
                    I = np.nan_to_num(I, nan=0.0, posinf=0.0, neginf=0.0)

                    np.save(dif_path, I)
                    dif_done = 1
                except Exception as e:
                    print(f"  [WARN] diffraction failed at {band_dir}: {e}")
            else:
                print(f"  [WARN] kpts.txt in {band_dir} has fewer than 3 numbers; skip diffraction.")
        else:
            print(f"  [WARN] POSCAR or kpts.txt missing in {band_dir}; skip diffraction.")
    else:
        print(f"[{band_dir}] diffraction disabled, skip")
    # else:
    #     print(f"[{band_dir}] diffrac.npy exists, skip diffraction.")

    # ---------------- DOS ----------------
    dos_path = os.path.join(band_dir, "dos.npy")
    # if not os.path.isfile(dos_path):
    if run_dos:
        try:
            pdos = PDOS(dir_path=band_dir)
            pdos.read_vasprun()  # vasprun.xml → DOS
            emin, emax = dos_range
            pdos.interpolate_dos(emin, emax, dos_grid)
            # shape: (n_sites, n_E) or (n_E,)
            I = np.asarray(pdos.densities, dtype=float)
            if I.size and (mI := float(np.max(I))) > 0:
                I = I / mI
            np.save(dos_path, I)
            dos_done = 1
        except Exception as e:
            print(f"  [WARN] DOS preprocessing failed at {band_dir}: {e}")
    else:
        print(f"[{band_dir}] DOS disabled, skip.")
    # else:
    #     print(f"[{band_dir}] dos.npy exists, skip DOS.")

    # ---------------- EBS (band structure) ----------------
    ebs_path = os.path.join(band_dir, "ebs.npy")
    # if not os.path.isfile(ebs_path):
    if run_ebs:
        try:
            M = build_band_map_from_ebs(
                ebs_path=os.path.join(band_dir, "EBS.dat"),
                emin=dos_range[0], emax=dos_range[1], nE=nE,
                nK=nk,
                sigma_E=sigma_E_ebs,
                normalize=True
            )
            M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
            if M.size and (mM := float(np.max(M))) > 0:
                M = M / mM

            np.save(ebs_path, M)
            ebs_done = 1
        except Exception as e:
            print(f"  [WARN] EBS preprocessing failed at {band_dir}: {e}")
    else:
        print(f"[{band_dir}] EBS disabled, skip.")
    # else:
    #     print(f"[{band_dir}] ebs.npy exists, skip band structure.")

    return (dif_done, dos_done, ebs_done)

# ----------------------------------------------------------------------
# mp directory unit worker
# ----------------------------------------------------------------------


def _worker(
    mp_dir: str,
    band_prefix: str,
    dos_range: Tuple[float, float],
    dos_grid: int,
    tt_range: Tuple[float, float],
    dif_grid: int,
    fwhm_q: float,
    nk: int,
    nE: int,
    sigma_E_ebs: float,
    run_diffraction: bool,
    run_dos: bool,
    run_ebs: bool,
) -> dict:
    """
    Process all internal band_* directories for one mp_* directory.
    """
    print(f"\n=== Processing materials: {mp_dir} ===")
    band_dirs = get_dirs(band_prefix, d_base=mp_dir)

    etc_root = os.path.join(mp_dir, "etc")
    os.makedirs(etc_root, exist_ok=True)

    dif_ok = dos_ok = ebs_ok = 0

    for band_dir in band_dirs:
        d, o, b = _process_one_band_dir(
            band_dir=band_dir,
            dos_range=dos_range,
            dos_grid=dos_grid,
            tt_range=tt_range,
            dif_grid=dif_grid,
            fwhm_q=fwhm_q,
            nk=nk,
            nE=nE,
            sigma_E_ebs=sigma_E_ebs,
            etc_root=etc_root,
            run_diffraction=run_diffraction,
            run_dos=run_dos,
            run_ebs=run_ebs,
        )
        dif_ok += d
        dos_ok += o
        ebs_ok += b

    return {"dir": mp_dir, "dif_done": dif_ok, "dos_done": dos_ok, "ebs_done": ebs_ok}

# ----------------------------------------------------------------------
# CLI entry
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess VASP outputs (diffraction, DOS, band) for DOS2Bandnet."
    )
    parser.add_argument("--base-dir", type=str, default=".", help="root directory containing mp* folders")
    parser.add_argument("--prefix1", type=str, default="mp", help="prefix of materials directories")
    parser.add_argument("--prefix2", type=str, default="ebs", help="prefix of band/EBS directories")

    # DOS grid
    parser.add_argument("--dos-emin", type=float, default=-10.0, help="minimum energy for DOS grid (eV)")
    parser.add_argument("--dos-emax", type=float, default=5.0, help="maximum energy for DOS grid (eV)")
    parser.add_argument("--dos-grid", type=int, default=300, help="number of energy points for DOS")

    # Diffraction grid
    parser.add_argument("--tt-min", type=float, default=0.05, help="minimum 2θ (degree)")
    parser.add_argument("--tt-max", type=float, default=4.0, help="maximum 2θ (degree)")
    parser.add_argument("--dif-grid", type=int, default=300, help="number of q/2θ points")
    parser.add_argument("--fwhm-q", type=float, default=0.08, help="peak broadening parameter in q-space")

    # EBS grid
    parser.add_argument("--nk", type=int, default=96, help="k-grid size for ebs.npy")
    parser.add_argument("--nE", type=int, default=256, help="(reserved) energy grid size for EBS (currently unused)")
    parser.add_argument("--sigma-E-ebs", type=float, default=0.20, help="(reserved) sigma for k-direction smoothing")

    # multiprocessing
    parser.add_argument("--nproc", type=int, default=1, help="number of processes (0 -> use cpu_count)")
    parser.add_argument("--tasks", type=str, default="diffraction,dos,ebs", help="tasks to run (diffraction, dos, ebs)")

    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    mp_prefix = args.prefix1
    band_prefix = args.prefix2

    dos_range = (args.dos_emin, args.dos_emax)
    dos_grid = args.dos_grid

    tt_range = (args.tt_min, args.tt_max)
    dif_grid = args.dif_grid
    fwhm_q = args.fwhm_q

    nk = args.nk
    nE = args.nE
    sigma_E_ebs = args.sigma_E_ebs

    nproc = args.nproc if args.nproc > 0 else cpu_count()

    task_tokens = [t.strip().lower() for t in args.tasks.split(",") if t.strip()]
    run_diffraction = "diffraction" in task_tokens
    run_dos = "dos" in task_tokens
    run_ebs = "ebs" in task_tokens

    mp_dirs = get_dirs(mp_prefix, d_base=base_dir)
    if not mp_dirs:
        print(f"No directories starting with '{mp_prefix}' found under {base_dir}")
        sys.exit(0)

    print(f"Found {len(mp_dirs)} material directories under {base_dir}")
    print(f"Using {nproc} processes")
    print(f"Tasks enabled: diffraction={run_diffraction} dos={run_dos} ebs={run_ebs}")

    if nproc == 1:
        for d in mp_dirs:
            res = _worker(
                d,
                band_prefix=band_prefix,
                dos_range=dos_range,
                dos_grid=dos_grid,
                tt_range=tt_range,
                dif_grid=dif_grid,
                fwhm_q=fwhm_q,
                nk=nk,
                nE=nE,
                sigma_E_ebs=sigma_E_ebs,
                run_diffraction=run_diffraction,
                run_dos=run_dos,
                run_ebs=run_ebs,
            )
            print(f"\n[{res['dir']}] DIF:{res['dif_done']}  DOS:{res['dos_done']}  EBS:{res['ebs_done']}")
    else:
        with Pool(processes=nproc, maxtasksperchild=10) as pool:
            func = partial(
                _worker,
                band_prefix=band_prefix,
                dos_range=dos_range,
                dos_grid=dos_grid,
                tt_range=tt_range,
                dif_grid=dif_grid,
                fwhm_q=fwhm_q,
                nk=nk,
                nE=nE,
                sigma_E_ebs=sigma_E_ebs,
                run_diffraction=run_diffraction,
                run_dos=run_dos,
                run_ebs=run_ebs,
            )
            for res in pool.imap_unordered(func, mp_dirs, chunksize=1):
                print(f"\n[{res['dir']}] DIF:{res['dif_done']}  DOS:{res['dos_done']}  EBS:{res['ebs_done']}")


if __name__ == "__main__":
    main()
