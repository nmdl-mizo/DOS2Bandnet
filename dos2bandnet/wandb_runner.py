#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_wandb.py (get_dirs version)
- Automatic directory discovery (get_dirs) → save split txt → VAE training → LDM training → save preview png/csv
- Create W&B sweep / run single agent / run multi-GPU agent_pool (multiple concurrent agents)
- Hyperparameter sweep: batch / scheduler / learning rate / weight decay / model width / condition encoder, etc.
- Automatically scale epochs_diff / StepLR.step_size / warmup_epochs according to batch size changes
- Upload final logs and best ckpt/preview as W&B Artifacts
- ★ Save config + record once more in wandb.log under params/*, args/*, data/*, sched/* categories

Examples:
# 1) Create sweep (prints SWEEP_ID)
python train_wandb.py --mode create \
  --project band-prediction --entity yeongrokj95-pnu --out_root runs

# 2) Run a single agent (in the current process)
CUDA_VISIBLE_DEVICES=0 python train_wandb.py --mode agent \
  --sweep_id <SWEEP_ID> --project band-prediction --entity yeongrokj95-pnu

# 3) Run multi-GPU agent_pool (2 agents each on GPU 0 and 1)
python train_wandb.py --mode agent_pool \
  --sweep_id <SWEEP_ID> --pool_gpus 0,1 --agents_per_gpu 2 \
  --project band-prediction --entity yeongrokj95-pnu
"""
import os
import sys
import time
import json
import argparse
import tqdm
import multiprocessing as mp
from dataclasses import asdict
from typing import Dict, Any, List, Optional

# ---- Import project utilities / training code ----
from .train_core import (  # noqa: E402
    TrainConfig,
    train_vae_with_val, train_ldm_with_val,
    view_ranked_from_ldm_ckpt, view_ranked_from_vae_ckpt,
)

# ----------------------------------------
# W&B import
# ----------------------------------------
import wandb  # noqa: E402


# ===================== Utilities =====================
def get_dirs(prefix1, prefix2=None, d_base="./"):
    if isinstance(prefix1, list):
        ValueError("Prefix1 has to be 'list'")
    res = []

    if prefix2 is None:
        # Find directories in d_base that satisfy the prefix1 condition (depth1)
        for ent in os.listdir(d_base):
            full1 = os.path.join(d_base, ent)
            if os.path.isdir(full1) and any(ent.startswith(p) for p in prefix1):
                res.append(full1)
    else:
        # If prefix2 is a string, convert it to a list
        p2_list = [prefix2] if isinstance(prefix2, str) else prefix2
        # Find directories in d_base that satisfy the prefix1 condition, then search inside them for directories satisfying the prefix2 condition (depth2)
        for ent in os.listdir(d_base):
            full1 = os.path.join(d_base, ent)
            if os.path.isdir(full1) and any(ent.startswith(p) for p in prefix1):
                for sub in os.listdir(full1):
                    full2 = os.path.join(full1, sub)
                    if os.path.isdir(full2) and any(sub.startswith(p) for p in p2_list):
                        res.append(full2)
    return res

def _parse_overrides(s: Optional[str]) -> Dict[str, Any]:
    if not s: return {}
    out = {}
    for kv in s.split(","):
        if "=" not in kv: continue
        k, v = kv.split("=", 1)
        k = k.strip(); v = v.strip()
        # infer type
        if v.lower() in ("true", "false"):
            out[k] = (v.lower() == "true")
        else:
            try:
                out[k] = int(v) if v.isdigit() or (v.startswith("-") and v[1:].isdigit()) else float(v)
            except Exception:
                out[k] = v
    return out


def _scale_epochs_and_step(cfg: TrainConfig, batch_size: int, base_bs: int,
                           epochs_diff_base: int, step_size_base_diff: int, step_size_base_vae: Optional[int],
                           warmup_ratio: float) -> None:
    """Scale epochs_diff/StepLR.step_size/warmup_epochs according to batch size changes."""
    base_bs = max(1, int(base_bs))
    batch_size = max(1, int(batch_size))
    scale = (float(batch_size) / base_bs + 1) / 2  # bs↑ → ep
    cfg.epochs_diff = max(1, int(round(epochs_diff_base * scale)))
    cfg.step_size_diff = max(1, int(round(step_size_base_diff * scale)))
    cfg.step_size_vae = max(1, int(round(step_size_base_vae * scale)))
    cfg.warmup_epochs = max(0, int(round(cfg.epochs_diff * float(warmup_ratio))))


def _cfg_from_wb(wb_cfg: "wandb.sdk.wandb_config.Config") -> TrainConfig:
    """wandb.config → TrainConfig (override only registered fields) + apply batch-dependent scaling."""
    cfg = TrainConfig()
    for k, v in dict(wb_cfg).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    _scale_epochs_and_step(
        cfg,
        batch_size=int(wb_cfg.get("batch_size", cfg.batch_size)),
        base_bs=int(wb_cfg.get("base_batch_size", 32)),
        epochs_diff_base=int(wb_cfg.get("epochs_diff_base", cfg.epochs_diff)),
        step_size_base_diff=int(wb_cfg.get("step_size_base", cfg.step_size_diff)),
        step_size_base_vae=(wb_cfg.get("step_size_base_vae", None)),
        warmup_ratio=float(wb_cfg.get("warmup_ratio", 0.2)),
    )
    return cfg


def _last_csv_row(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    if len(lines) <= 1:
        return {}
    header = lines[0].split(",")
    vals = lines[-1].split(",")
    out: Dict[str, Any] = {}
    for h, v in zip(header, vals):
        vv = v
        try:
            vv = float(v)
        except Exception:
            pass
        out[h] = vv
    return out


def _make_epoch_logger(phase: str):
    def _log(**kwargs):
        wandb.log({f"{phase}/{k}": v for k, v in kwargs.items()})
    return _log


def _count_lines(p: str) -> int:
    try:
        with open(p, "r") as f:
            return sum(1 for ln in f if ln.strip() and not ln.strip().startswith("#"))
    except Exception:
        return 0


def _log_static_params(cfg: TrainConfig, args: argparse.Namespace, step: Optional[int] = None):
    """
    Request restored: also send parameters through **log**.
    - save with wandb.config.update + record with wandb.log under params/*, args/*, sched/* categories
    """
    # 1) Update config (overwrite with allow_val_change=True even if already present)
    wandb.config.update(asdict(cfg), allow_val_change=True)

    # 2) Log once more under params/*
    params = {f"params/{k}": v for k, v in asdict(cfg).items()}

    # 3) Also log under args/*
    arg_keys = [
        "project", "entity", "out_root", "run_name_prefix",
        "prefix1", "prefix2", "sweep_id", "gpu",
        "pool_gpus", "agents_per_gpu", "count", "count_per_agent", "mode",
    ]
    for k in arg_keys:
        if hasattr(args, k):
            params[f"args/{k}"] = getattr(args, k)

    # 4) Also include scheduler-related summary
    params.update({
        "sched/type": str(getattr(cfg, "scheduler_type", "")),
        "sched/min_lr": float(getattr(cfg, "min_lr", 0.0)),
        "sched/step_size_diff": int(getattr(cfg, "step_size_diff", 0)),
        "sched/step_size_vae": int(getattr(cfg, "step_size_vae", 0)),
        "sched/gamma": float(getattr(cfg, "gamma", 0.0)),
        "sched/epochs_diff_effective": int(getattr(cfg, "epochs_diff", 0)),
        "sched/warmup_epochs_effective": int(getattr(cfg, "warmup_epochs", 0)),
    })

    wandb.log(params, step=step)


# ===================== Training routine (executed by agent) =====================
def _run_one(args) -> None:
    """One run executed by a single agent."""
    init_cfg = getattr(args, "init_config", None)
    run = wandb.init(project=args.project, entity=args.entity, reinit=False, config=init_cfg)
    run_name = f"{args.run_name_prefix}-{run.id}" if run is not None else f"{args.run_name_prefix}-none"
    try:
        wandb.run.name = run_name
    except Exception:
        pass

    # collect data
    dirs: List[str] = get_dirs(prefix1=args.prefix1, prefix2=args.prefix2)
    if len(dirs) == 0:
        raise RuntimeError(f"get_dirs returned empty. prefix1={args.prefix1}, prefix2={args.prefix2}")

    # build & expose config (+log)
    cfg = _cfg_from_wb(wandb.config)
    _log_static_params(cfg, args, step=0)  # ★ as requested, also record parameters through log

    out_root = args.out_root

    # training
    t0 = time.time()

    cond_model = str(getattr(cfg, "cond_model", "diffusion")).lower()

    if cond_model == "simple":
        vae_info = None
        ldm_info = train_ldm_with_val(
            None,
            cfg,
            out_root=out_root,
            run_name=run_name,
            epoch_logger=_make_epoch_logger("ldm"),
            dir_list=dirs,
        )
        t2 = time.time()
    else:
        vae_info = train_vae_with_val(
            dirs, cfg, out_root=out_root, run_name=run_name,
            epoch_logger=_make_epoch_logger("vae")
        )

        t1 = time.time()
        if not vae_info or "best_ckpt" not in vae_info or not os.path.isfile(vae_info["best_ckpt"]):
            wandb.log({"error/vae_best_ckpt_missing": True})
            raise RuntimeError("VAE best checkpoint not found — cannot proceed to LDM")

        # ---------- VAE PREVIEW ----------
        try:
            view_ranked_from_vae_ckpt(vae_info["best_ckpt"], split="test")
        except Exception as e:
            wandb.log({"error/preview_vae_test": str(e)})

        # ---------- LDM ----------
        ldm_info = train_ldm_with_val(
            vae_info["best_ckpt"], cfg, out_root=out_root, run_name=run_name,
            epoch_logger=_make_epoch_logger("ldm")
        )
        t2 = time.time()

    # log data split statistics (data/*)
    if vae_info is not None:
        sp = vae_info.get("split", {})
        data_log = {
            "data/train_n": _count_lines(sp.get("train", "")),
            "data/val_n": _count_lines(sp.get("val", "")),
            "data/test_n": _count_lines(sp.get("test", "")),
        }
        wandb.log(data_log)

    try:
        from .train_core import set_seed
        set_seed(cfg.seed)
        ret = view_ranked_from_ldm_ckpt(
            None, split="test", steps=max(1, int(cfg.preview_steps)), guidance=float(getattr(cfg, "preview_guidance", 1.0)),
                        out_root = out_root, run_name = run_name)

        if ret and "errors" in ret:
            import numpy as np
            test_L1_mean = float(np.mean(ret["errors"]))
            test_L1_median = float(np.median(ret["errors"]))
            metric_log ={
                "summary/test_L1": test_L1_mean,
                "summary/test_L1_median": test_L1_median,
            }
            if "ssim_scores" in ret and len(ret["ssim_scores"]) > 0:
                metric_log.update({
                    "summary/test_SSIM": float(np.mean(ret["ssim_scores"])),
                    "summary/test_SSIM_median": float(np.median(ret["ssim_scores"])),
                })
            if "inv_ssim_m1s" in ret and len(ret["inv_ssim_m1s"]) > 0:
                metric_log.update({
                    "summary/test_inv_ssim_m1": float(np.mean(ret["inv_ssim_m1s"])),
                    "summary/test_inv_ssim_m1_median": float(np.median(ret["inv_ssim_m1s"])),
                })
            if "top10_maes" in ret and len(ret["top10_maes"]) > 0:
                metric_log.update({
                    "summary/test_top10_mae": float(np.mean(ret["top10_maes"])),
                    "summary/test_top10_mae_median": float(np.median(ret["top10_maes"])),
                })
            wandb.log(metric_log)

              # --- NEW: also log preview indices using raw txt indices for exact reproducibility ---

            if isinstance(ret, dict):
                raw_idxs = ret.get("indices_raw")
                sorted_idxs = ret.get("indices_sorted")

                if raw_idxs is not None:
                    wandb.log({
                        "preview/indices_raw": raw_idxs,
                        "preview/indices_sorted": sorted_idxs if sorted_idxs is not None else []})
            # path to CSV with (index_raw, error)

                if ret.get("error_csv"):
                    wandb.log({"preview/error_csv": ret["error_csv"]})

    except Exception as e:
        wandb.log({"error/preview_ldm_test": str(e)})

    # aggregate logs/metrics
    if vae_info is not None:
        vae_last = _last_csv_row(vae_info["log"])
        wandb.log({
            "summary/val_total_vae": float(vae_last.get("val_total", 0.0)),
        })
    ldm_last = _last_csv_row(ldm_info["log"])
    wandb.log({
        "summary/val_loss_ldm": float(ldm_last.get("val_loss", 0.0)),
        "time/vae_s": t1 - t0,
        "time/ldm_s": t2 - t1,
        "time/total_s": t2 - t0,
    })

    # upload artifact
    art = wandb.Artifact(name=f"{run_name}-artifacts", type="results")
    root = os.path.join(out_root, run_name)
    for rel in [
        os.path.join("checkpoints", "vae_best.pt"),
        os.path.join("checkpoints", "ldm_best.pt"),
        os.path.join("logs", "vae_train.csv"),
        os.path.join("logs", "ldm_train.csv"),
    ]:
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            art.add_file(p, name=rel)
    prev_dir = os.path.join(root, "previews")
    if os.path.isdir(prev_dir):
        art.add_dir(prev_dir, name="previews")
    try:
        wandb.run.log_artifact(art)
    except Exception:
        pass

def replay_from_run(args) -> None:
    """
    Start a new run by copying config from a previous W&B run.
    --from_run can be
      1) the full path "entity/project/<run_id>", or
      2) "<run_id>" (in this case, --entity/--project are required)
    """
    api = wandb.Api()
    if "/" in args.from_run:
        run_path = args.from_run  # "entity/project/<run_id>"
    else:
        if not args.entity or not args.project:
            raise SystemExit("Error: if --from_run is given in <run_id> format, please also provide --entity and --project.")
        run_path = f"{args.entity}/{args.project}/{args.from_run}"

    src = api.run(run_path)
    prev_cfg: Dict[str, Any] = dict(src.config)

    # apply desired changes (change only lr, or connect several with , )
    if args.lr_diff_new is not None:
        prev_cfg["lr_diff"] = float(args.lr_diff_new)
    prev_cfg.update(_parse_overrides(args.override))

    # prefix for new run name: if absent, automatically use "re-<src.id>"
    if not args.run_name_prefix:
        args.run_name_prefix = f"re-{src.id}"

    # put config into init via args and reuse existing training routine
    args.init_config = prev_cfg
    _run_one(args)


# ===================== Sweep / Agent / Agent Pool =====================
def _default_sweep_config() -> Dict[str, Any]:
    """Minimal required parameters: concise + core only."""
    return {
        "method": "bayes",
        # "metric": {"name": "summary/val_loss_ldm", "goal": "minimize"},
        "metric": {"name": "summary/test_L1", "goal": "minimize"},
        "parameters": {
            # --- scaling baseline ---
            "batch_size": {"values": [32]},
            "epochs_diff_base": {"values": [300]},
            "base_batch_size": {"value": 32},
            "step_size_base": {"values": [90]},
            "step_size_base_vae": {"values": [30]},
            "warmup_ratio": {"values": [0.2]},
            # --- optimizer / scheduler ---
            "lr_diff": {"distribution": "log_uniform_values", "min": 8e-5, "max": 8e-4},
            "weight_decay": {"distribution": "log_uniform_values", "min": 1e-6, "max": 3e-4},
            "scheduler_type": {"values": ["cosine_warmup"]},
            # "scheduler_type": {"values": ["cosine_warmup", "cosine", "step"]},
            "min_lr": {"values": [1e-7, 1e-6]},
            "gamma": {"values": [0.5, 0.7]},
            "consine_tm_ratio": {"values": [1]},
            # --- model width / condition encoder ---
            "z_ch": {"values": [4]},
            "model_ch": {"values": [128]},
            "cond_dim": {"values": [256]},
            "enc_dos_type": {"values": ["deep"]},
            "enc_ed_type": {"values": ["deep"]},
            "dos_only": {"values": [False]},
            # "cond_model": {"values": ["diffusion"]}, #diffusion, simple
            "cond_model": {"values": ["simple"]}, #diffusion, simple
            # --- diffusion loss / prediction ---
            # "predict_type": {"values": ["v", "eps"]},
            "predict_type": {"values": ["v"]},
            "p2_gamma": {"values": [0.95]},
            "p2_k": {"values": [0.95]},
            "auto_estimate_latent_scale": {"values": ["true"]},
            "apply_wd_aux": {"values": ["true", "false"]},
            # "apply_wd_aux": {"values": ["false"]},
            # --- preview ---
            "preview_steps": {"values": [50]},
            "preview_eta": {"values": [0.0]},
            "preview_guidance": {"values": [1.0]},
            # --- dropout ---
            "dropout_p": {"values": [0.05]},
            "cond_drop_prob": {"values": [0.0]},
            # --- L1 support ---
            "genl1_lambda": {"values": [1]},
        },
    }


def create_sweep(args) -> None:
    sweep_cfg = _default_sweep_config()
    sweep_id = wandb.sweep(sweep_cfg, project=args.project, entity=args.entity)
    print(f"Create sweep with ID: {sweep_id}")
    print(f"Sweep URL: https://wandb.ai/{args.entity}/{args.project}/sweeps/{sweep_id}")
    print(f"SWEEP_ID: {sweep_id}")


def run_agent(sweep_id: str, count: int, args) -> None:
    """Run a single agent (in the current process)."""
    # fix environment variables
    if args.project: os.environ["WANDB_PROJECT"] = args.project
    if args.entity:  os.environ["WANDB_ENTITY"]  = args.entity
    # GPU can be restricted externally with CUDA_VISIBLE_DEVICES, and --gpu is also supported
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    wandb.agent(sweep_id, function=lambda: _run_one(args), count=count)


def _agent_worker(sweep_id: str, args_dict: Dict[str, Any], cuda_visible: Optional[str], count: int):
    """Worker for multiprocessing: single agent loop in each process."""
    # configure environment
    if args_dict.get("project"): os.environ["WANDB_PROJECT"] = str(args_dict["project"])
    if args_dict.get("entity"):  os.environ["WANDB_ENTITY"]  = str(args_dict["entity"])
    if cuda_visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible)
    # reconstruct args object (simple Dict → Namespace)
    NS = argparse.Namespace(**args_dict)
    wandb.agent(sweep_id, function=lambda: _run_one(NS), count=count)


def run_agent_pool(sweep_id: str, pool_gpus: List[int], agents_per_gpu: int, count_per_agent: int, args) -> None:
    """
    Run a multi-GPU/process agent pool.
    - For each GPU, launch agents_per_gpu processes and run them concurrently
    - Each process fixes CUDA_VISIBLE_DEVICES to a single GPU
    """
    procs: List[mp.Process] = []
    args_dict = vars(args).copy()
    # ignore --gpu in agent_pool mode
    args_dict["gpu"] = None
    for g in pool_gpus:
        for _ in range(max(1, int(agents_per_gpu))):
            p = mp.Process(target=_agent_worker,
                           args=(sweep_id, args_dict, str(g), int(count_per_agent)))
            p.daemon = False
            p.start()
            procs.append(p)
    # join
    for p in procs:
        p.join()
