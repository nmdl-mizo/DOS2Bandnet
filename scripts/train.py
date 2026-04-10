#!/usr/bin/env python
"""
CLI entrypoint for training DOS2BandNet.

This script is a thin wrapper around `dos2bandnet.train_core`:
- discovers data directories (or uses a provided txt file)
- constructs a TrainConfig
- runs VAE and/or LDM training
"""

import argparse
import os, re
from typing import List, Optional, Tuple, Union

from dos2bandnet.train_core import (
    TrainConfig,
    train_vae_with_val,
    train_ldm_with_val,
    train_ldm_finetune,
    resolve_ckpt,
    view_ranked_from_vae_ckpt,
    view_ranked_from_ldm_ckpt,
)

import torch

def get_dirs(prefix1: Union[str, List[str]],
             prefix2: Optional[Union[str, List[str]]] = None,
             d_base: str = "./") -> List[str]:
    # normalize prefix1
    if isinstance(prefix1, str):
        p1_list = [prefix1]
    elif isinstance(prefix1, list):
        p1_list = prefix1
    else:
        raise TypeError("prefix1 must be str or list[str]")

    res: List[str] = []

    if prefix2 is None:
        # depth1
        for ent in os.listdir(d_base):
            full1 = os.path.join(d_base, ent)
            if os.path.isdir(full1) and any(ent.startswith(p) for p in p1_list):
                res.append(full1)
    else:
        # normalize prefix2
        if isinstance(prefix2, str):
            p2_list = [prefix2]
        elif isinstance(prefix2, list):
            p2_list = prefix2
        else:
            raise TypeError("prefix2 must be str, list[str], or None")

        # depth2
        for ent in os.listdir(d_base):
            full1 = os.path.join(d_base, ent)
            if os.path.isdir(full1) and any(ent.startswith(p) for p in p1_list):
                for sub in os.listdir(full1):
                    full2 = os.path.join(full1, sub)
                    if os.path.isdir(full2) and any(sub.startswith(p) for p in p2_list):
                        res.append(full2)

    res.sort()
    return res


def _discover_dirs(
    data_base: str,
    prefix1: str,
    prefix2: Optional[str],
    dirs_txt: Optional[str],
) -> List[str]:
    """Return a list of sample directories.

    Priority:
      1) if dirs_txt is given: read list from that file
      2) else: scan `data_base` using get_dirs(prefix1, prefix2)
    """
    if dirs_txt is not None:
        if not os.path.isfile(dirs_txt):
            raise FileNotFoundError(f"dirs-txt not found: {dirs_txt}")
        with open(dirs_txt) as f:
            return [ln.strip() for ln in f if ln.strip()]

    # fallback: scan
    return get_dirs(prefix1=prefix1, prefix2=prefix2, d_base=data_base)

def _parse_split(split_str: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in split_str.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"split must have 3 comma-separated values, got: {split_str}")
    vals = tuple(float(p) for p in parts)
    if abs(sum(vals) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got: {vals}")
    return vals  # type: ignore[return-value]

def _next_finetune_run(out_root: str, pretrained_id: str) -> str:
    base_dir = os.path.join(out_root, pretrained_id)
    if not os.path.isdir(base_dir):
        return f"{pretrained_id}/finetune1"
    existing = []
    for ent in os.listdir(base_dir):
        m = re.match(r"^finetune(\d+)$", ent)
        if m:
            existing.append(int(m.group(1)))
    next_idx = max(existing, default=0) + 1
    print(f"Finetune outroot: {out_root}/{pretrained_id}/finetune{next_idx}")
    return f"{pretrained_id}/finetune{next_idx}"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DOS2BandNet")

    # data
    p.add_argument(
        "--data-base", type=str, default=".",
        help="Base directory that contains mp-*/ebs_* style folders (default: .)",
    )
    p.add_argument(
        "--prefix1", type=str, default="mp-",
        help="Top-level directory prefix (default: mp-)",
    )
    p.add_argument(
        "--prefix2", type=str, default="ebs_",
        help="Second-level directory prefix (default: ebs_)",
    )
    p.add_argument(
        "--dirs-txt", type=str, default=None,
        help="Optional: txt file that lists per-sample directories (one per line)",
    )

    # training mode
    p.add_argument(
        "--mode", type=str, choices=["vae", "ldm", "both", "finetune"],
        default="both",
        help="What to train: vae, ldm, or both (default: ldm)",
    )

    # run / output
    p.add_argument(
        "--run-name", type=str, default="exp1",
        help="Run name under out-root (default: exp1)",
    )
    p.add_argument(
        "--out-root", type=str, default="runs",
        help="Root directory to save logs/checkpoints (default: runs)",
    )

    # optional: vae ckpt path for LDM-only training
    p.add_argument(
        "--vae-ckpt", type=str, default=None,
        help="Path to pretrained VAE checkpoint (if None, auto-resolve in the same run)",
    )
    p.add_argument(
        "--pretrained-id", type=str, default=None,
        help="Run name (ID) of a pretrained model under --out-root (for finetune).",
    )
    p.add_argument(
        "--finetune-base", type=str, default=None,
        help="Base directory for finetune target data (for finetune mode).",
    )
    p.add_argument(
        "--finetune-split", type=str, default="0.8,0.1,0.1",
        help="Train/val/test split ratios for finetune target (default: 0.8,0.1,0.1).",
    )
    p.add_argument(
        "--finetune-lr", type=float, default=5e-6,
        help="Learning rate for finetune (default: 1e-5).",
    )
    p.add_argument(
        "--min-lr", type=float, default=1e-7,
        help="Minimum learning rate (default: 1e-7).",
    )
    p.add_argument(
        "--warmup-epochs", type=int, default=100,
        help="Warmup epochs (default: 100).",
    )

    # simple config overrides
    p.add_argument(
        "--batch-size", type=int, default=None,
        help="Override TrainConfig.batch_size (optional)",
    )
    p.add_argument(
        "--epochs-vae", type=int, default=None,
        help="Override TrainConfig.epochs_vae (optional)",
    )
    p.add_argument(
        "--epochs-diff", type=int, default=None,
        help="Override TrainConfig.epochs_diff (optional)",
    )
    p.add_argument(
        "--no-preview", dest="preview", action="store_false",
        help="Disable quartile previews after training (default: enabled)",
    )
    p.add_argument(
        "--preview-split", type=str, default="test",
        help="Dataset split to use for preview images (default: test)",
    )
    p.set_defaults(preview=True)

    return p.parse_args()


def main():
    args = parse_args()

    # 1) discover data dirs
    if args.mode == "finetune":
        if args.pretrained_id is None:
            raise ValueError("--pretrained-id is required for finetune mode.")
        if args.finetune_base is None:
            raise ValueError("--finetune-base is required for finetune mode.")
        dir_list = _discover_dirs(
            data_base=args.finetune_base,
            prefix1=args.prefix1,
            prefix2=args.prefix2,
            dirs_txt=args.dirs_txt,
        )
    else:
        dir_list = _discover_dirs(
            data_base=args.data_base,
            prefix1=args.prefix1,
            prefix2=args.prefix2,
            dirs_txt=args.dirs_txt,
        )

    if not dir_list:
        raise RuntimeError(
            "No data directories found. "
            "Check --data-base / --prefix1 / --prefix2 / --dirs-txt."
        )

    # 2) build config and apply simple overrides
    if args.mode == "finetune":
        ldm_ckpt = resolve_ckpt(args.out_root, args.pretrained_id, kind="ldm")
        ckpt = torch.load(ldm_ckpt, map_location="cpu")
        meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
        cfg = TrainConfig(**meta.get("config", {}))
        cfg.lr_diff = args.finetune_lr
        cfg.epochs_diff = args.epochs_diff
        cfg.min_lr = args.min_lr
        cfg.warmup_epochs = args.warmup_epochs
    else:
        cfg = TrainConfig()
        if args.batch_size is not None:
            cfg.batch_size = args.batch_size
        if args.epochs_vae is not None:
            cfg.epochs_vae = args.epochs_vae
        if args.epochs_diff is not None:
            cfg.epochs_diff = args.epochs_diff

    # 3) run training
    vae_info = None

    if args.mode in ("vae", "both"):
        print("[DOS2BandNet] Training VAE ...")
        vae_info = train_vae_with_val(
            dir_list,
            cfg,
            out_root=args.out_root,
            run_name=args.run_name,
            split=(0.8, 0.1, 0.1),
        )
        print(f"[DOS2BandNet] VAE done. Best ckpt: {vae_info['best_ckpt']}")

        if args.preview:
            try:
                view_ranked_from_vae_ckpt(
                    vae_info["best_ckpt"],
                    split=args.preview_split,
                    save_png=True,
                )
                print(f"[DOS2BandNet] VAE previews saved for split={args.preview_split}.")
            except Exception as exc:
                print(f"[DOS2BandNet] VAE preview failed: {exc}")

    if args.mode in ("ldm", "both"):
        print("[DOS2BandNet] Training LDM ...")

        # decide which VAE checkpoint to use
        vae_ckpt = args.vae_ckpt
        if vae_ckpt is None and vae_info is not None:
            vae_ckpt = vae_info["best_ckpt"]
        if vae_ckpt is None:
            try:
                vae_ckpt = resolve_ckpt(args.out_root, args.run_name, kind="vae")
                print(f"[DOS2BandNet] Resolved VAE checkpoint: {vae_ckpt}")
            except FileNotFoundError:
                print(
                    "[DOS2BandNet] WARNING: no VAE checkpoint found. "
                    "LDM will create a new VAE from scratch."
                )
                vae_ckpt = None

        ldm_info = train_ldm_with_val(
            vae_ckpt_path=vae_ckpt,
            cfg=cfg,
            out_root=args.out_root,
            run_name=args.run_name,
            dir_list=dir_list if str(getattr(cfg, "cond_model", "diffusion")).lower() == "simple" else None,
        )
        print(f"[DOS2BandNet] LDM done. Best ckpt: {ldm_info['best_ckpt']}")  # type: ignore[index]

        if args.preview:
            guidance = float(getattr(cfg, "preview_guidance", 1.0))
            try:
                view_ranked_from_ldm_ckpt(
                    ldm_info["best_ckpt"],
                    split=args.preview_split,
                    steps=cfg.preview_steps,
                    guidance=guidance,
                    use_ema=bool(getattr(cfg, "use_ema", True)),
                    save_png=True,
                    out_root=args.out_root,
                    run_name=args.run_name,
                )
                print(f"[DOS2BandNet] LDM previews saved for split={args.preview_split}.")
            except Exception as exc:
                print(f"[DOS2BandNet] LDM preview failed: {exc}")

    if args.mode == "finetune":
        print("[DOS2BandNet] Finetuning LDM ...")
        finetune_run = _next_finetune_run(args.out_root, args.pretrained_id)
        split = _parse_split(args.finetune_split)
        ldm_ckpt = resolve_ckpt(args.out_root, args.pretrained_id, kind="ldm")
        cond_model = str(getattr(cfg, "cond_model", "diffusion")).lower()
        vae_ckpt = None if cond_model == "simple" else resolve_ckpt(args.out_root, args.pretrained_id, kind="vae")

        ldm_info = train_ldm_finetune(
            vae_ckpt_path=vae_ckpt,
            ldm_ckpt_path=ldm_ckpt,
            dir_list=dir_list,
            cfg=cfg,
            out_root=args.out_root,
            run_name=finetune_run,
            split=split,
        )
        print(f"[DOS2BandNet] Finetune done. Best ckpt: {ldm_info['best_ckpt']}")  # type: ignore[index]

        if args.preview:
            guidance = float(getattr(cfg, "preview_guidance", 1.0))
            try:
                view_ranked_from_ldm_ckpt(
                    ldm_info["best_ckpt"],
                    split=args.preview_split,
                    steps=cfg.preview_steps,
                    guidance=guidance,
                    # use_ema=bool(getattr(cfg, "use_ema", True)),
                    use_ema=False,
                    save_png=True,
                    out_root=args.out_root,
                    run_name=finetune_run,
                )
                print(f"[DOS2BandNet] Finetune previews saved for split={args.preview_split}.")
            except Exception as exc:
                print(f"[DOS2BandNet] Finetune preview failed: {exc}")


if __name__ == "__main__":
    main()
