# n2k_bandnet/train_core.py
import os, math, random, json, glob, time, copy
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
from dataclasses import dataclass, asdict
from typing import Tuple, List, Optional, Dict, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LambdaLR,
)
from tqdm import tqdm

# ✅ Modified only here: existing `from model import (` → package-relative import
from .model import (
    SimpleVAE,
    LatentDiffusionModel,
    SimpleBandModel,
    build_cond_encoder,
    make_schedule,
    q_sample,
    EMA,
    sample_ddim,
    ddim_timesteps,
)

# ===================== Config =====================
@dataclass
class TrainConfig:
    # seed
    seed: int = 1227

    # data
    BAND_SHAPE: Tuple[int, int] = (96, 256)
    DOS_LEN: int = 300
    ED_LEN: int = 300
    num_workers: int = 4
    batch_size: int = 32

    # vae
    vae_base: int = 64
    z_ch: int = 4
    downsample: int = 4
    kl_beta: float = 1e-3
    auto_estimate_latent_scale: bool = True
    estimate_batches: int = 96

    # epochs
    epochs_vae: int = 30
    epochs_diff: int = 300

    # lr
    lr_vae: float = 1e-4
    lr_diff: float = 6e-4
    weight_decay: float = 1e-4

    # scheduler
    scheduler_type: str = "cosine_warmup"  # step | cosine | cosine_warmup | cosine_restart
    step_size_diff: int = 100
    step_size_vae: int = 10
    gamma: float = 0.5
    min_lr: float = 1e-7
    warmup_epochs: int = int(epochs_diff * 0.2)
    cosine_t0: int = 1000
    cosine_tmult: int = 2
    consine_tm_ratio: int = 1

    # diffusion
    t_steps: int = 500
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    predict_type: str = "v"  # "v" | "eps"
    p2_gamma: float = 0.5
    p2_k: float = 1.0

    # CFG / EMA / Dropout
    cond_drop_prob: float = 0.0  # classifier-free guidance style condition dropout prob (training only)
    use_ema: bool = False
    ema_decay: float = 0.999
    dropout_p: float = 0.0  # architectural dropout probability across encoders/UNet/VAE

    # latent scale (z scale for ddpm)
    latent_scale: float = 0.18215

    # model
    cond_dim: int = 256
    model_ch: int = 128
    enc_dos_type: str = "deep"  # "wide" | "deep" | "simple"
    enc_ed_type: str = "deep"  # "wide" | "deep" | "simple"
    dos_only: bool = False      # If True, use only DOS as the condition (ignore ED input)
    cond_model: str = "diffusion"  # "diffusion" | "simple"

    # aux (optional)
    apply_wd_aux: bool = True
    wd_loss_weight: float = 0.3
    norm_mode: str = "sum"  # "sum" | "softmax"
    softmax_tau: float = 0.5
    bin_width: float = 1.0

    # preview
    preview_steps: int = 50
    preview_eta: float = 0.0
    preview_guidance: float = 1.0
    preview_log_every: int = 50

    # --- final-sample L1 (for loss; lightweight unrolling) ---
    genl1_lambda: float = 1.0  # Disable when set to 0
    genl1_steps: int = 8  # Number of DDIM unrolling steps (small)
    genl1_every: int = 20  # Period per mini-batch iteration (e.g., every 20 iters)
    genl1_batch: int = 4  # Sub-batch size used for this auxiliary term
    genl1_eta: float = 0.0  # DDIM eta
    genl1_guidance: float = 1.0  # guidance scale

# ===================== Utils =====================
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def device_auto():
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def get_current_lr(optimizer): return optimizer.param_groups[0]["lr"]

def format_seconds(sec: float) -> str:
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    sec = sec % 60
    if h: return f"{h}h{m:02d}m{sec:02d}s"
    if m: return f"{m}m{sec:02d}s"
    return f"{sec}s"


# ===================== Dataset =====================
class BandDataset(Dataset):
    """For each directory: dos.npy, diffrac.npy, band.npy (or ebs.npy / band/band.npy)"""
    def __init__(self, dir_list: List[str], band_shape=(96, 256), dos_len=300, ed_len=300):
        super().__init__()
        self.band_shape = band_shape; self.dos_len = dos_len; self.ed_len = ed_len
        self.samples = []
        for d in sorted(dir_list):
            dos = os.path.join(d, "dos.npy")
            ed  = os.path.join(d, "diffrac.npy")
            # flexible band path
            band_candidates = [
                os.path.join(d, "band.npy"),
                os.path.join(d, "ebs.npy"),
                os.path.join(d, "band", "band.npy"),
            ]
            band = next((p for p in band_candidates if os.path.isfile(p)), None)
            if os.path.isfile(dos) and os.path.isfile(ed) and band is not None:
                self.samples.append((dos, ed, band))
        if len(self.samples) == 0:
            raise RuntimeError("No samples: expect dos.npy, diffrac.npy, band.npy (or ebs.npy, band/band.npy).")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        dos_p, ed_p, band_p = self.samples[idx]
        dos = np.load(dos_p).astype(np.float32)
        ed  = np.load(ed_p).astype(np.float32)
        band = np.load(band_p).astype(np.float32)
        nE = self.band_shape[1]; nk = self.band_shape[0]
        if dos.ndim > 1: dos = dos.reshape(-1)
        if ed.ndim > 1: ed = ed.reshape(-1)
        # 0~1 normalize (robust)
        def norm01(x):
            lo, hi = np.percentile(x, 1), np.percentile(x, 99)
            if hi <= lo: hi = x.max(); lo = x.min()
            if hi <= lo: return np.zeros_like(x, dtype=np.float32)
            return ((x - lo) / (hi - lo)).clip(0, 1).astype(np.float32)
        dos = norm01(dos)[:self.DOS_LEN] if hasattr(self, "DOS_LEN") else norm01(dos)
        ed  = norm01(ed) [:self.ED_LEN ] if hasattr(self, "ED_LEN" ) else norm01(ed)

        if band.shape[0] < nk or band.shape[1] < nE:
            band = np.pad(band, ((0, nk - band.shape[0]), (0, nE - band.shape[1])))

        dos = dos[None, :]
        ed = ed[None, :]
        band = band[None, :, :]
        band = band * 2.0 - 1.0

        return {"dos": torch.from_numpy(dos),
                "ed": torch.from_numpy(ed),
                "band": torch.from_numpy(band)}


# ===================== LR Scheduler =====================
def build_lr_scheduler(optimizer, cfg: TrainConfig, kind: str = "diff"):
    st = cfg.scheduler_type.lower()
    if st == "step":
        step = cfg.step_size_diff if kind == "diff" else cfg.step_size_vae
        sched = StepLR(optimizer, step_size=step, gamma=cfg.gamma)
        info = f"StepLR({step},{cfg.gamma})"
    elif st == "cosine":
        Tm = cfg.epochs_diff // cfg.consine_tm_ratio if kind == "diff" else cfg.epochs_vae
        sched = CosineAnnealingLR(optimizer, T_max=Tm, eta_min=cfg.min_lr); info = f"Cosine(T_max={Tm}, eta_min={cfg.min_lr})"
    elif st == "cosine_restart":
        sched = CosineAnnealingWarmRestarts(optimizer, T_0=cfg.cosine_t0, T_mult=cfg.cosine_tmult, eta_min=cfg.min_lr)
        info = f"CosineRestart(T0={cfg.cosine_t0},Tmult={cfg.cosine_tmult},eta_min={cfg.min_lr})"
    elif st == "cosine_warmup":
        base_lr = cfg.lr_diff if kind == "diff" else cfg.lr_vae
        min_ratio = float(cfg.min_lr) / float(base_lr)
        Tm = cfg.epochs_diff if kind == "diff" else cfg.epochs_vae

        # warmup_epochs is an absolute value based on the 'diff stage' → converted proportionally for the current stage (kind)
        warmup_ratio = float(cfg.warmup_epochs) / float(max(1, cfg.epochs_diff))
        warmup_e = int(round(Tm * warmup_ratio))
        warmup_e = min(max(warmup_e, 0), max(0, Tm - 1))

        def lr_lambda(ep: int):
            if ep < warmup_e:
                return (ep + 1) / max(1, warmup_e)
            progress = (ep - warmup_e) / max(1, Tm - warmup_e)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        sched = LambdaLR(optimizer, lr_lambda=lr_lambda); info = f"Warmup({warmup_e})+Cosine->{cfg.min_lr}"
    else:
        raise ValueError(f"Unknown SCHEDULER_TYPE: {cfg.scheduler_type}")
    return sched, info


# ===================== IO Helpers (runs/<name>/...) =====================
def io_ensure_dirs(out_root: str, run_name: str):
    root = os.path.join(out_root, run_name)
    ds_dir = os.path.join(root, 'datasets'); os.makedirs(ds_dir, exist_ok=True)
    log_dir = os.path.join(root, 'logs'); os.makedirs(log_dir, exist_ok=True)
    ckpt_dir = os.path.join(root, 'checkpoints'); os.makedirs(ckpt_dir, exist_ok=True)
    prev_dir = os.path.join(root, 'previews'); os.makedirs(prev_dir, exist_ok=True)
    return root, ds_dir, log_dir, ckpt_dir, prev_dir

def write_list(path: str, items: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for s in items: f.write(str(s).strip()+"\n")

def read_list(path: str) -> List[str]:
    with open(path, 'r') as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith('#')]

def split_dirs_and_save(dir_list: List[str], out_root: str = 'runs', run_name: str = 'exp',
                         ratios=(0.8,0.1,0.1), seed: int = 1227):
    assert abs(sum(ratios)-1.0) < 1e-6, 'ratios must sum to 1'
    rng = random.Random(seed)
    arr = sorted(dir_list)
    rng.shuffle(arr)
    N = len(arr)
    n_tr = int(round(N*ratios[0])); n_va = int(round(N*ratios[1])); n_te = N - n_tr - n_va
    tr = arr[:n_tr]; va = arr[n_tr:n_tr+n_va]; te = arr[n_tr+n_va:]
    root, ds_dir, _, _, _ = io_ensure_dirs(out_root, run_name)
    tr_txt = os.path.join(ds_dir, 'train.txt'); va_txt = os.path.join(ds_dir, 'val.txt'); te_txt = os.path.join(ds_dir, 'test.txt')
    write_list(tr_txt, tr); write_list(va_txt, va); write_list(te_txt, te)
    return { 'root': root, 'train': tr_txt, 'val': va_txt, 'test': te_txt }


# ===================== Builders =====================
def build_dataloader(dir_list: List[str], cfg: TrainConfig, device: torch.device, shuffle=True, drop_last=True):
    use_cuda = (device.type == "cuda")
    ds = BandDataset(dir_list, band_shape=cfg.BAND_SHAPE, dos_len=cfg.DOS_LEN, ed_len=cfg.ED_LEN)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers,
                    pin_memory=use_cuda, drop_last=drop_last)
    return ds, dl

def build_dataloader_from_txt(txt_path: str, cfg: TrainConfig, device: torch.device, shuffle=False, drop_last=False):
    dirs = read_list(txt_path)
    use_cuda = (device.type == 'cuda')
    ds = BandDataset(dirs, band_shape=cfg.BAND_SHAPE, dos_len=cfg.DOS_LEN, ed_len=cfg.ED_LEN)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers,
                    pin_memory=use_cuda, drop_last=drop_last)
    return ds, dl

# --- helper: map dataset(sorted) index -> raw txt index ---
def _map_sorted_idx_to_raw(ds: BandDataset, txt_path: str):
    raw_dirs = read_list(txt_path)
    mapping = [] # mapping[j_sorted] = j_raw
    for j, (dos_p, ed_p, band_p) in enumerate(ds.samples):
        d = os.path.dirname(dos_p)
        try:
            mapping.append(raw_dirs.index(d))
        except ValueError:
            mapping.append(j) # fallback
    return mapping


def build_models(cfg: TrainConfig, device: torch.device):
    cond_model = str(getattr(cfg, "cond_model", "diffusion")).lower()
    if cond_model == "simple":
        vae = None
        ldm = SimpleBandModel(
            cond_dim=cfg.cond_dim, model_ch=cfg.model_ch,
            band_shape=cfg.BAND_SHAPE, drop_p=cfg.dropout_p, cond_drop_prob=cfg.cond_drop_prob
        ).to(device)
    else:
        vae = SimpleVAE(in_ch=1, base=cfg.vae_base, z_ch=cfg.z_ch, down=cfg.downsample, drop_p=cfg.dropout_p).to(device)
        ldm = LatentDiffusionModel(vae, cond_dim=cfg.cond_dim, z_ch=cfg.z_ch, model_ch=cfg.model_ch,
                                   T_total=cfg.t_steps, drop_p=cfg.dropout_p, cond_drop_prob=cfg.cond_drop_prob).to(device)
    enc_dos = build_cond_encoder(cfg.enc_dos_type, in_ch=1, out_dim=cfg.cond_dim // 2, drop_p=cfg.dropout_p).to(device)
    enc_ed = None if cfg.dos_only else build_cond_encoder(cfg.enc_ed_type, in_ch=1, out_dim=cfg.cond_dim // 2, drop_p=cfg.dropout_p).to(device)
    ldm.set_encoders(enc_dos, enc_ed)
    sched_ddpm = make_schedule(cfg.t_steps, cfg.beta_start, cfg.beta_end, device)
    return vae, ldm, sched_ddpm

def _ensure_simple_encoders(ldm, cfg: TrainConfig, device):
    if getattr(ldm, "enc_dos", None) is None:
        enc_dos = build_cond_encoder(cfg.enc_dos_type, in_ch=1, out_dim=cfg.cond_dim // 2, drop_p=cfg.dropout_p).to(device)
        enc_ed = None if cfg.dos_only else build_cond_encoder(cfg.enc_ed_type, in_ch=1, out_dim=cfg.cond_dim // 2, drop_p=cfg.dropout_p).to(device)
        ldm.set_encoders(enc_dos, enc_ed)
    return ldm

# ===================== Epoch loops =====================
def _vae_epoch(vae: SimpleVAE, loader: DataLoader, device, kl_beta: float, train: bool, opt: Optional[torch.optim.Optimizer] = None):
    if train: vae.train()
    else: vae.eval()
    rec_e=kl_e=tot_e=0.0; n=0
    for batch in loader:
        x0 = batch["band"].to(device, non_blocking=(device.type=="cuda"))
        if train:
            x_hat, mu, logvar = vae(x0, sample_posterior=True)
            loss, rec, kl = F.l1_loss(x_hat, x0) + kl_beta * ( -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) ), \
                            F.l1_loss(x_hat, x0), \
                            -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        else:
            with torch.no_grad():
                x_hat, mu, logvar = vae(x0, sample_posterior=False)
                loss = F.l1_loss(x_hat, x0) + kl_beta * ( -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) )
                rec  = F.l1_loss(x_hat, x0)
                kl   = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        rec_e+=float(rec.item()); kl_e+=float(kl.item()); tot_e+=float(loss.item()); n+=1
    return {"rec": rec_e/max(1,n), "kl": kl_e/max(1,n), "total": tot_e/max(1,n)}

def _ldm_epoch(ldm, vae, loader, sched, cfg, device, train: bool,
               opt: Optional[torch.optim.Optimizer]=None, ema: Optional[EMA]=None,
               lr_sch=None):
    if train: ldm.train()
    else: ldm.eval()
    run=0.0; steps=0
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))
    for batch in loader:
        dos=batch['dos'].to(device, non_blocking=(device.type=="cuda"))
        ed =batch['ed' ].to(device, non_blocking=(device.type=="cuda"))
        x0 =batch['band'].to(device, non_blocking=(device.type=="cuda"))
        cond_model = str(getattr(cfg, "cond_model", "diffusion")).lower()
        if cond_model == "simple":
            _ensure_simple_encoders(ldm, cfg, device)
            if train:
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    pred_band = ldm(dos, ed, uncond=False)
                    loss = F.l1_loss(pred_band, x0)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(ldm.parameters(), 1.0)
                scaler.step(opt);
                scaler.update()
                if cfg.use_ema and ema is not None: ema.update(ldm)
            else:
                with torch.no_grad():
                    pred_band = ldm(dos, ed, uncond=False)
                    loss = F.l1_loss(pred_band, x0)
            run += float(loss.item());
            steps += 1
            continue

        B = x0.size(0)
        with torch.no_grad():
            z0,_ = vae.encode(x0, sample_posterior=False); z0 = z0 * cfg.latent_scale
        t = torch.randint(0, cfg.t_steps, (B,), device=device, dtype=torch.long)
        z_t, eps = q_sample(z0, t, sched)
        if train:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                pred = ldm(z_t, t, dos, ed, uncond=False)
                a = sched.sqrt_alphas_cumprod[t].view(B,1,1,1)
                s = sched.sqrt_one_minus_alphas_cumprod[t].view(B,1,1,1)
                pt = cfg.predict_type.lower()
                if pt == "v":
                    v_true = a*eps - s*z0
                    snr = (a**2)/(s**2 + 1e-12)
                    w = (cfg.p2_k + snr).pow(-cfg.p2_gamma)
                    loss = (w*(pred-v_true).pow(2)).mean()
                else:
                    loss = F.mse_loss(pred, eps)

                # --- final generated sample L1 (lightweight unrolling) auxiliary term ---
                if train and cfg.genl1_lambda > 0 and (steps % int(cfg.genl1_every) == 0):
                    gen_l1 = _final_sample_L1(ldm, vae, sched, cfg, device, dos, ed, x0)
                    loss = loss + float(cfg.genl1_lambda) * gen_l1

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(ldm.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            if cfg.use_ema and ema is not None: ema.update(ldm)
        else:
            with torch.no_grad():
                pred = ldm(z_t, t, dos, ed, uncond=False)
                a = sched.sqrt_alphas_cumprod[t].view(B,1,1,1)
                s = sched.sqrt_one_minus_alphas_cumprod[t].view(B,1,1,1)
                pt = cfg.predict_type.lower()
                if pt == "v":
                    v_true = a*eps - s*z0
                    snr = (a**2)/(s**2 + 1e-12)
                    w = (cfg.p2_k + snr).pow(-cfg.p2_gamma)
                    loss = (w*(pred-v_true).pow(2)).mean()
                else:
                    loss = F.mse_loss(pred, eps)
        run += float(loss.item()); steps += 1
    return {"loss": run/max(1,steps)}


# ===================== Training APIs (with val & best ckpt) =====================
# ===== train.py: replace train_vae_with_val =====
def train_vae_with_val(
        dir_list: List[str],
        cfg: TrainConfig,
        out_root='runs',
        run_name='exp',
        split=(0.8,0.1,0.1),
        epoch_logger: Optional[Callable] = None):

    set_seed(cfg.seed)
    device = device_auto(); use_cuda=(device.type=='cuda')
    print(device)
    if use_cuda:
        torch.backends.cudnn.benchmark=True
        try: torch.set_float32_matmul_precision('high')
        except: pass

    split_info = split_dirs_and_save(dir_list, out_root=out_root, run_name=run_name, ratios=split, seed=cfg.seed)
    root, ds_dir, log_dir, ckpt_dir, prev_dir = io_ensure_dirs(out_root, run_name)

    # data loaders
    _, dl_tr = build_dataloader_from_txt(split_info['train'], cfg, device, shuffle=True, drop_last=True)
    _, dl_va = build_dataloader_from_txt(split_info['val'],   cfg, device, shuffle=False, drop_last=False)

    vae = SimpleVAE(in_ch=1, base=cfg.vae_base, z_ch=cfg.z_ch, down=cfg.downsample, drop_p=cfg.dropout_p).to(device)
    opt = AdamW(vae.parameters(), lr=cfg.lr_vae, weight_decay=0.0)
    sch, sch_info = build_lr_scheduler(opt, cfg, kind="vae")
    from tqdm import tqdm
    tqdm.write(f"[VAE] LR scheduler: {sch_info}")

    # logs
    log_csv = os.path.join(log_dir, 'vae_train.csv')
    with open(log_csv, 'w') as f:
        f.write('epoch,lr,epoch_sec,train_rec,train_kl,train_total,val_rec,val_kl,val_total\n')

    best = float('inf'); best_path = os.path.join(ckpt_dir, 'vae_best.pt')
    for ep in range(1, cfg.epochs_vae+1):
        start_t = time.time()
        lr_now = get_current_lr(opt)

        tr = _vae_epoch(vae, dl_tr, device, cfg.kl_beta, train=True,  opt=opt)
        va = _vae_epoch(vae, dl_va, device, cfg.kl_beta, train=False, opt=None)

        epoch_sec = time.time() - start_t
        with open(log_csv, 'a') as f:
            f.write(f"{ep},{lr_now:.8e},{epoch_sec:.3f},{tr['rec']:.6f},{tr['kl']:.6f},{tr['total']:.6f},{va['rec']:.6f},{va['kl']:.6f},{va['total']:.6f}\n")

        # >>> send per-epoch values to an external logger such as W&B
        if epoch_logger is not None:
            epoch_logger(
                epoch=ep, lr_now=lr_now, epoch_sec=epoch_sec,
                train_rec=tr['rec'], train_kl=tr['kl'], train_total=tr['total'],
                val_rec=va['rec'],   val_kl=va['kl'],   val_total=va['total'],
            )

        if va['total'] < best:
            best = va['total']
            meta = {"run_root": root, "datasets": split_info, "config": asdict(cfg)}
            torch.save({'vae_state': vae.state_dict(), 'arch': {'base': cfg.vae_base, 'z_ch': cfg.z_ch, 'down': cfg.downsample},
                        'metrics': {'best_val_total': float(best)}, 'meta': meta}, best_path)

        tqdm.write(
            "[VAE] ep {ep}/{epochs}  lr={lr:.3e}  time={time}  tr_rec={tr_rec:.3e}  "
            "va_rec={va_rec:.3e}  tr_total={tr_total:.3e}  va_total={va_total:.3e}  "
            "best={best:.3e}".format(
                ep=ep,
                epochs=cfg.epochs_vae,
                lr=lr_now,
                time=format_seconds(epoch_sec),
                tr_rec=tr["rec"],
                va_rec=va["rec"],
                tr_total=tr["total"],
                va_total=va["total"],
                best=best,
            )
        )
        sch.step()
    return {'best_ckpt': best_path, 'log': log_csv, 'split': split_info}


# ===== train_ldm_with_val =====
def train_ldm_with_val(vae_ckpt_path: Optional[str], cfg: TrainConfig,
                       out_root: str = "runs", run_name: str = "exp1",
                       epoch_logger: Optional[Callable] = None,
                        dir_list: Optional[List[str]] = None):
    set_seed(cfg.seed)
    device = device_auto(); use_cuda=(device.type=='cuda')
    if use_cuda:
        torch.backends.cudnn.benchmark=True
        try: torch.set_float32_matmul_precision('high')
        except: pass
    cond_model = str(getattr(cfg, "cond_model", "diffusion")).lower()
    if cond_model == "simple":
        if not dir_list:
            raise RuntimeError("In simple mode, dir_list is required.")
        split_info = split_dirs_and_save(dir_list, out_root=out_root, run_name=run_name, ratios=(0.8, 0.1, 0.1), seed=cfg.seed)
        root, ds_dir, log_dir, ckpt_dir, prev_dir = io_ensure_dirs(out_root, run_name)
        meta = {"run_root": root, "datasets": split_info}
    else:
        if vae_ckpt_path is None:
            vae_ckpt_path = resolve_ckpt(out_root, run_name, kind="vae")
            print(f"[auto] Using VAE ckpt: {vae_ckpt_path}")
        # meta/datasets from VAE ckpt
        ck_v = torch.load(vae_ckpt_path, map_location=device)
        meta = ck_v.get('meta', None)
        if meta is None:
            raise RuntimeError("VAE checkpoint missing 'meta' (run_root/datasets). Retrain VAE with this script.")
        split_info = meta['datasets']
        root, ds_dir, log_dir, ckpt_dir, prev_dir = io_ensure_dirs(out_root, run_name) if meta['run_root'] != run_name else io_ensure_dirs(out_root, run_name)

    # loaders
    _, dl_tr = build_dataloader_from_txt(split_info['train'], cfg, device, shuffle=True, drop_last=True)
    _, dl_va = build_dataloader_from_txt(split_info['val'],   cfg, device, shuffle=False, drop_last=False)

    cond_model = str(getattr(cfg, "cond_model", "diffusion")).lower()
    vae, ldm, sched = build_models(cfg, device)
    if cond_model != "simple":
        # In diffusion training, VAE is frozen (used in computation but parameters are not updated)
        for p in vae.parameters():
            p.requires_grad_(False)

        ck = torch.load(vae_ckpt_path, map_location=device)
        if isinstance(ck, dict) and "vae_state" in ck: vae.load_state_dict(ck["vae_state"], strict=False)
        else: vae.load_state_dict(ck, strict=False)

        # latent scale estimate (optional)
        LATENT_SCALE = cfg.latent_scale
        if cfg.auto_estimate_latent_scale:
            _, dl_est = build_dataloader_from_txt(split_info['train'], cfg, device, shuffle=True, drop_last=False)
            LATENT_SCALE = estimate_latent_scale(vae, dl_est, device, batches=cfg.estimate_batches)
        cfg.latent_scale = float(LATENT_SCALE)

    params = [p for p in ldm.parameters() if p.requires_grad]
    opt = AdamW(params, lr=cfg.lr_diff, weight_decay=cfg.weight_decay)

    # === (unified) epoch-based scheduler ===
    lr_sch, sch_info = build_lr_scheduler(opt, cfg, kind="diff")

    from tqdm import tqdm
    tqdm.write(f"[LDM] LR scheduler = {sch_info}")

    ema = EMA(ldm, cfg.ema_decay) if cfg.use_ema else None

    log_csv = os.path.join(log_dir, 'ldm_train.csv')
    with open(log_csv, 'w') as f:
        f.write('epoch,lr,epoch_sec,train_loss,val_loss\n')

    best = float('inf'); best_path = os.path.join(ckpt_dir, 'ldm_best.pt')
    for ep in range(1, cfg.epochs_diff+1):
        start_t = time.time()
        lr_now = get_current_lr(opt)

        tr = _ldm_epoch(ldm, vae, dl_tr, sched, cfg, device, train=True,  opt=opt, ema=ema, lr_sch=None)
        va = _ldm_epoch(ldm, vae, dl_va, sched, cfg, device, train=False, opt=None, ema=None, lr_sch=None)

        # ---- (every N epochs) DDIM-generated L1: val & test logging ----
        log_every = max(1, int(getattr(cfg, "preview_log_every", 10)))
        extra_logs = {}

        if (ep % log_every) == 0:
            # If EMA exists, create an evaluation copy with EMA weights
            ldm_eval = copy.deepcopy(ldm).to(device).eval()
            if cfg.use_ema and ema is not None:
                ema.copy_to(ldm_eval)
            # latent_scale has already been set above (cfg.latent_scale)
            steps_use = max(1, int(cfg.preview_steps))
            guidance = float(getattr(cfg, "preview_guidance", 1.0))
            try:
                # val
                val_txt = split_info.get("val", None)
                if val_txt:
                    v_metrics = _compute_split_metrics(ldm_eval, vae, sched, cfg, device, val_txt,
                                                       steps_use, guidance)
                    extra_logs.update({
                        "val_L1": v_metrics["mae_mean"],
                        "val_L1_median": v_metrics["mae_median"],
                        "val_SSIM": v_metrics["ssim_mean"],
                        "val_SSIM_median": v_metrics["ssim_median"],
                        "val_inv_ssim_m1": v_metrics["inv_ssim_m1_mean"],
                        "val_inv_ssim_m1_median": v_metrics["inv_ssim_m1_median"],
                        "val_top10_mae": v_metrics["top10_mae_mean"],
                        "val_top10_mae_median": v_metrics["top10_mae_median"],
                    })
                    # test (optional): keep disabled to reduce training-time evaluation overhead
                # test
                # test_txt = split_info.get("test", None)
                # if test_txt:
                #     t_mean, t_med = _compute_split_mae(ldm_eval, vae, sched, cfg, device, test_txt,
                #                                        steps_use, guidance)
                #     extra_logs.update({"test_L1": t_mean, "test_L1_median": t_med})
            except Exception as _e:
                extra_logs.update({"preview_error": str(_e)})

        lr_end = get_current_lr(opt)
        epoch_sec = time.time() - start_t
        with open(log_csv, 'a') as f:
            f.write(f"{ep},{lr_end:.8e},{epoch_sec:.3f},{tr['loss']:.6f},{va['loss']:.6f}\n")

        # >>> send per-epoch values to an external logger such as W&B
        if epoch_logger is not None:
            epoch_logger(epoch=ep, lr_now=lr_end, epoch_sec=epoch_sec,
                         train_loss = tr['loss'], val_loss = va['loss'], ** extra_logs)


        if va['loss'] < best:
            best = va['loss']
            meta_save = {"run_root": meta['run_root'], "datasets": split_info, "config": asdict(cfg), "vae_ckpt": vae_ckpt_path}
            torch.save({'epoch': ep, 'ldm_state': ldm.state_dict(), 'ema': (ema.shadow if (cfg.use_ema and ema is not None) else None),
                        'metrics': {'best_val_loss': float(best)}, 'meta': meta_save}, best_path)

        tqdm.write(
            "[LDM] ep {ep}/{epochs}  lr={lr:.3e}  time={time}  tr_loss={tr_loss:.3e}  "
            "va_loss={va_loss:.3e}  best={best:.3e}".format(
                ep=ep,
                epochs=cfg.epochs_diff,
                lr=lr_now,
                time=format_seconds(epoch_sec),
                tr_loss=tr["loss"],
                va_loss=va["loss"],
                best=best,
            )
        )
        lr_sch.step()

    return {'best_ckpt': best_path, 'log': log_csv, 'split': split_info}

def train_ldm_finetune(vae_ckpt_path: Optional[str], ldm_ckpt_path: str, dir_list: List[str], cfg: TrainConfig,
                       out_root: str = "runs", run_name: str = "finetune1",
                       split=(0.8, 0.1, 0.1), epoch_logger: Optional[Callable] = None):
    set_seed(cfg.seed)
    device = device_auto(); use_cuda=(device.type=='cuda')
    if use_cuda:
        torch.backends.cudnn.benchmark=True
        try: torch.set_float32_matmul_precision('high')
        except: pass

    split_info = split_dirs_and_save(dir_list, out_root=out_root, run_name=run_name, ratios=split, seed=cfg.seed)
    root, ds_dir, log_dir, ckpt_dir, prev_dir = io_ensure_dirs(out_root, run_name)

    # loaders
    _, dl_tr = build_dataloader_from_txt(split_info['train'], cfg, device, shuffle=True, drop_last=True)
    _, dl_va = build_dataloader_from_txt(split_info['val'],   cfg, device, shuffle=False, drop_last=False)

    cond_model = str(getattr(cfg, "cond_model", "diffusion")).lower()
    vae, ldm, sched = build_models(cfg, device)
    if cond_model != "simple":
        for p in vae.parameters():
            p.requires_grad_(False)

        ck_v = torch.load(vae_ckpt_path, map_location=device)
        if isinstance(ck_v, dict) and "vae_state" in ck_v: vae.load_state_dict(ck_v["vae_state"], strict=False)
        else: vae.load_state_dict(ck_v, strict=False)

    ck_l = torch.load(ldm_ckpt_path, map_location=device)
    if isinstance(ck_l, dict) and "ldm_state" in ck_l: ldm.load_state_dict(ck_l["ldm_state"], strict=False)
    else: ldm.load_state_dict(ck_l, strict=False)

    # latent scale estimate (optional)
    if cond_model != "simple":
        LATENT_SCALE = cfg.latent_scale
        if cfg.auto_estimate_latent_scale:
            _, dl_est = build_dataloader_from_txt(split_info['train'], cfg, device, shuffle=True, drop_last=False)
            LATENT_SCALE = estimate_latent_scale(vae, dl_est, device, batches=cfg.estimate_batches)
        cfg.latent_scale = float(LATENT_SCALE)

    params = [p for p in ldm.parameters() if p.requires_grad]
    opt = AdamW(params, lr=cfg.lr_diff, weight_decay=cfg.weight_decay)

    # === (unified) epoch-based scheduler ===
    lr_sch, sch_info = build_lr_scheduler(opt, cfg, kind="diff")

    from tqdm import tqdm
    tqdm.write(f"[LDM] LR scheduler = {sch_info}")

    ema = EMA(ldm, cfg.ema_decay) if cfg.use_ema else None
    if ema is not None and isinstance(ck_l, dict) and ck_l.get("ema") is not None:
        ema.shadow = ck_l["ema"]

    log_csv = os.path.join(log_dir, 'ldm_train.csv')
    with open(log_csv, 'w') as f:
        f.write('epoch,lr,epoch_sec,train_loss,val_loss\n')

    best = float('inf'); best_path = os.path.join(ckpt_dir, 'ldm_best.pt')
    for ep in range(1, cfg.epochs_diff+1):
        start_t = time.time()
        lr_now = get_current_lr(opt)

        tr = _ldm_epoch(ldm, vae, dl_tr, sched, cfg, device, train=True,  opt=opt, ema=ema, lr_sch=None)
        va = _ldm_epoch(ldm, vae, dl_va, sched, cfg, device, train=False, opt=None, ema=None, lr_sch=None)

        # ---- (every N epochs) DDIM-generated L1: val & test logging ----
        log_every = max(1, int(getattr(cfg, "preview_log_every", 10)))
        extra_logs = {}

        if (ep % log_every) == 0:
            # If EMA exists, create an evaluation copy with EMA weights
            ldm_eval = copy.deepcopy(ldm).to(device).eval()
            if cfg.use_ema and ema is not None:
                ema.copy_to(ldm_eval)
            # latent_scale has already been set above (cfg.latent_scale)
            steps_use = max(1, int(cfg.preview_steps))
            guidance = float(getattr(cfg, "preview_guidance", 1.0))
            try:
                # val
                val_txt = split_info.get("val", None)
                if val_txt:
                    v_metrics = _compute_split_metrics(ldm_eval, vae, sched, cfg, device, val_txt,
                                                       steps_use, guidance)
                    extra_logs.update({
                        "val_L1": v_metrics["mae_mean"],
                        "val_L1_median": v_metrics["mae_median"],
                        "val_SSIM": v_metrics["ssim_mean"],
                        "val_SSIM_median": v_metrics["ssim_median"],
                        "val_inv_ssim_m1": v_metrics["inv_ssim_m1_mean"],
                        "val_inv_ssim_m1_median": v_metrics["inv_ssim_m1_median"],
                        "val_top10_mae": v_metrics["top10_mae_mean"],
                        "val_top10_mae_median": v_metrics["top10_mae_median"],
                    })
            except Exception as _e:
                extra_logs.update({"preview_error": str(_e)})

        lr_end = get_current_lr(opt)
        epoch_sec = time.time() - start_t
        with open(log_csv, 'a') as f:
            f.write(f"{ep},{lr_end:.8e},{epoch_sec:.3f},{tr['loss']:.6f},{va['loss']:.6f}\n")

        # >>> send per-epoch values to an external logger such as W&B
        if epoch_logger is not None:
            epoch_logger(epoch=ep, lr_now=lr_end, epoch_sec=epoch_sec,
                         train_loss = tr['loss'], val_loss = va['loss'], ** extra_logs)


        if va['loss'] < best:
            best = va['loss']
            meta_save = {
                "run_root": root,
                "datasets": split_info,
                "config": asdict(cfg),
                "vae_ckpt": vae_ckpt_path,
                "pretrained_ldm_ckpt": ldm_ckpt_path,
            }
            torch.save({'epoch': ep, 'ldm_state': ldm.state_dict(), 'ema': (ema.shadow if (cfg.use_ema and ema is not None) else None),
                        'metrics': {'best_val_loss': float(best)}, 'meta': meta_save}, best_path)

        tqdm.write(
            "[LDM] ep {ep}/{epochs}  lr={lr:.3e}  time={time}  tr_loss={tr_loss:.3e}  "
            "va_loss={va_loss:.3e}  best={best:.3e}".format(
                ep=ep,
                epochs=cfg.epochs_diff,
                lr=lr_now,
                time=format_seconds(epoch_sec),
                tr_loss=tr["loss"],
                va_loss=va["loss"],
                best=best,
            )
        )
        lr_sch.step()

    return {'best_ckpt': best_path, 'log': log_csv, 'split': split_info}

def _final_sample_L1(ldm, vae, sched, cfg: TrainConfig, device,
                     dos, ed, x0):
    """
    Unroll DDIM by cfg.genl1_steps.
    Allow gradients only through the network at the final step,
    then compute and return the L1 between the final generated output
    (the decoded band) and the GT.
    """
    Bm = min(int(cfg.genl1_batch), dos.size(0))
    dos = dos[:Bm]; ed = ed[:Bm]; x0 = x0[:Bm]
    nk, nE = cfg.BAND_SHAPE
    H = nk // cfg.downsample; W = nE // cfg.downsample
    z_t = torch.randn(Bm, cfg.z_ch, H, W, device=device)

    t_list = ddim_timesteps(cfg.t_steps, int(cfg.genl1_steps))
    for i, t_scalar in enumerate(t_list):
        t = torch.full((Bm,), t_scalar, device=device, dtype=torch.long)
        # guidance
        def denoise(with_grad: bool):
            if cfg.genl1_guidance != 1.0:
                ctx = (torch.enable_grad() if with_grad else torch.no_grad())
                with ctx:
                    v_u = ldm(z_t, t, dos, ed, uncond=True)
                    v_c = ldm(z_t, t, dos, ed, uncond=False)
                    v = v_u + cfg.genl1_guidance * (v_c - v_u)
            else:
                ctx = (torch.enable_grad() if with_grad else torch.no_grad())
                with ctx:
                    v = ldm(z_t, t, dos, ed, uncond=False)
            return v

        last = (i == len(t_list) - 1)
        v_hat = denoise(with_grad=last)  # final step only uses grad

        alpha_t = sched.sqrt_alphas_cumprod[t_scalar]
        sigma_t = sched.sqrt_one_minus_alphas_cumprod[t_scalar]
        if cfg.predict_type.lower() == "v":
            x0_hat = alpha_t * z_t - sigma_t * v_hat
            eps_hat= sigma_t * z_t + alpha_t * v_hat
        else:
            eps_hat = v_hat
            x0_hat  = (z_t - sigma_t * eps_hat) / (alpha_t + 1e-12)

        if last:
            z_t = x0_hat
            break

        t_prev = t_list[i + 1]
        abar_t    = sched.alphas_cumprod[t_scalar]
        abar_prev = sched.alphas_cumprod[t_prev]
        sigma_ddim = cfg.genl1_eta * torch.sqrt((1 - abar_prev) / (1 - abar_t)) * torch.sqrt(1 - abar_t / abar_prev)
        noise = torch.randn_like(z_t) if cfg.genl1_eta > 0 else 0.0
        z_t = torch.sqrt(abar_prev) * x0_hat + torch.sqrt(1 - abar_prev - sigma_ddim ** 2) * eps_hat + sigma_ddim * noise

    z = z_t / cfg.latent_scale
    # VAE is frozen, but gradients with respect to input z are still needed, so do not use no_grad
    x_hat = vae.decode(z).clamp(-1, 1)
    return F.l1_loss(x_hat, x0)

# ===================== Latent scale estimate =====================
@torch.no_grad()
def estimate_latent_scale(vae: SimpleVAE, loader: DataLoader, device, batches: int = 16) -> float:
    cnt = 0
    errs = []
    vae.eval()
    for batch in loader:
        x0 = batch["band"].to(device)
        # posterior mean(z=mu) is used to compute reconstruction value → measure MAE
        x_hat, _, _ = vae(x0, sample_posterior=False)
        err = torch.mean(torch.abs(x_hat - x0)).item()
        errs.append(err)
        cnt += 1
        if cnt >= batches:
            break
    # stable numeric value: 1/(1+MAE)
    s = 1.0 / (1.0 + float(np.mean(errs))) if errs else 1.0
    return float(s)


# ===================== Preview helpers =====================
def _select_ranked_indices(errs: List[float], k_low=2, k_mid=2, k_high=2):
    order = np.argsort(errs)  # lower is better
    N = len(order)
    lows = order[:k_low].tolist()
    mids = order[N//2 - k_mid//2 : N//2 + (k_mid - k_mid//2)].tolist()
    highs = order[-k_high:].tolist()[::-1]
    idxs = lows + mids + highs
    groups = {"low": lows, "mid": mids, "high": highs}
    return idxs, groups

def _select_ranked_indices_quartiles(errors: List[float], k_each: int = 2):
    """
    Sort errors in ascending order and split into four quartiles (Q1~Q4),
    then select k_each samples (default: 2) from each quartile and return them.

    Returns:
      idxs: concatenated list in the order [selected from Q1, Q2, Q3, Q4]
      groups: {"q1":[...], "q2":[...], "q3":[...], "q4":[...]}
    """

    import numpy as np
    errs = np.asarray(errors, dtype=float)
    order = np.argsort(errs)  # ascending indices
    bins = np.array_split(order, 4)  # [Q1, Q2, Q3, Q4]
    labels = ["q1", "q2", "q3", "q4"]
    groups = {}
    idxs = []
    for lab, arr in zip(labels, bins):
        idx_list = arr.tolist()
        if len(idx_list) == 0:
            groups[lab] = []
            continue
        q_errs = errs[idx_list]
        med = float(np.median(q_errs))
        # stable tie-break in the order: distance → error value → index
        candidates = sorted(idx_list, key=lambda i: (abs(errs[i] - med), errs[i], i))
        # pick = candidates[:min(k_each, len(candidates))] # middle one
        pick = idx_list[:min(k_each, len(idx_list))]    # first one
        groups[lab] = pick
        idxs.extend(pick)
    return idxs, groups

def _ensure_preview_dir(run_root: str):
    prev_dir = os.path.join(run_root, "previews")
    os.makedirs(prev_dir, exist_ok=True)
    return prev_dir

def _batch_ssim(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0,
                k1: float = 0.01, k2: float = 0.03, eps: float = 1e-12) -> torch.Tensor:
    """
    Fast global SSIM approximation per sample for tensors shaped (B, C, H, W).
    Returns tensor of shape (B,).
    """
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    dims = (2, 3)
    mu_x = x.mean(dim=dims, keepdim=True)
    mu_y = y.mean(dim=dims, keepdim=True)
    var_x = ((x - mu_x) ** 2).mean(dim=dims, keepdim=True)
    var_y = ((y - mu_y) ** 2).mean(dim=dims, keepdim=True)
    cov_xy = ((x - mu_x) * (y - mu_y)).mean(dim=dims, keepdim=True)

    num = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
    den = (mu_x.pow(2) + mu_y.pow(2) + c1) * (var_x + var_y + c2)
    ssim_map = num / (den + eps)
    return ssim_map.mean(dim=(1, 2, 3))

def _batch_top10_mae(pred: torch.Tensor, gt: torch.Tensor, top_ratio: float = 0.10) -> torch.Tensor:
    """
    Per-sample MAE computed on the top-intensity region of GT.
    Region definition: top `top_ratio` fraction by GT intensity (default: top 10%).
    Returns tensor of shape (B,).
    """
    B = gt.size(0)
    flat_gt = gt.view(B, -1)
    q = torch.quantile(flat_gt, q=max(0.0, min(1.0, 1.0 - float(top_ratio))), dim=1, keepdim=True)
    mask = (flat_gt >= q).float()

    flat_err = torch.abs(pred - gt).view(B, -1)
    masked_sum = (flat_err * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return masked_sum / denom

@torch.no_grad()
def view_ranked_from_vae_ckpt(vae_ckpt_path: str, split: str = "test",
                              k_low=2, k_mid=2, k_high=2,
                              save_png: bool=True):
    device = device_auto()
    ck = torch.load(vae_ckpt_path, map_location=device)
    meta = ck.get('meta', None)
    if meta is None:
        raise RuntimeError("VAE ckpt missing meta (datasets/run_root). Train with this script.")
    cfg = TrainConfig(**meta.get('config', {}))
    txt_path = meta['datasets'].get(split)
    if txt_path is None: raise RuntimeError(f"Unknown split: {split}")
    ds, dl = build_dataloader_from_txt(txt_path, cfg, device, shuffle=False, drop_last=False)
    vae = SimpleVAE(in_ch=1, base=cfg.vae_base, z_ch=cfg.z_ch, down=cfg.downsample, drop_p=cfg.dropout_p).to(device)
    vae.load_state_dict(ck['vae_state'] if 'vae_state' in ck else ck, strict=False)
    vae.eval()

    errs, cache = [], []
    for batch in DataLoader(ds, batch_size=1, shuffle=False):
        x0 = batch["band"].to(device)
        z,_ = vae.encode(x0, sample_posterior=False)
        x_hat = vae.decode(z).clamp(-1, 1)
        err = torch.mean(torch.abs(x_hat - x0)).item()
        errs.append(err); cache.append((x0.cpu(), x_hat.cpu()))
    idxs, groups = _select_ranked_indices_quartiles(errs, k_each=2)

    prev_dir = _ensure_preview_dir(meta['run_root'])

    # ↓ list of directory paths: ds.samples[i] = (dos_path, ed_path, band_path)
    dir_list = [os.path.dirname(p0) for (p0, _, _) in ds.samples]

    order = np.argsort(errs)  # ascending
    csv_path = os.path.join(prev_dir, f"vae_{split}_sorted_errors.csv")
    with open(csv_path, "w") as f:
        f.write("index,error,dir\n")  # ← add dir to header
        for i in order:
            f.write(f"{int(i)},{errs[i]:.8f},{dir_list[i]}\n")  # ← record path

    outs = []
    import matplotlib.pyplot as plt
    for rank, i in enumerate(idxs):
        x0, x_hat = cache[i]
        x0 = x0[0,0].numpy(); xh = x_hat[0,0].numpy()
        fig, axs = plt.subplots(1,2, figsize=(6,4))
        axs[0].imshow((x0.T+1)/2, aspect="auto", origin="lower"); axs[0].set_title("GT"); axs[0].axis("off")
        axs[1].imshow((xh.T+1)/2, aspect="auto", origin="lower"); axs[1].set_title(f"Recon L1={errs[i]:.4f}"); axs[1].axis("off")
        plt.tight_layout()
        out = os.path.join(prev_dir, f"vae_{split}_rank_{rank:02d}_idx{i}.png")
        if save_png: plt.savefig(out, dpi=150); plt.close(fig)
        outs.append(out)
    return {"indices": idxs, "groups": groups, "errors": errs, "images": outs, "split": split, "error_csv": csv_path}


@torch.no_grad()
def view_ranked_from_ldm_ckpt(ldm_ckpt_path: Optional[str] = None, split: str = "test", steps: Optional[int]=None,
                              guidance: float=1.0, use_ema: bool=True, save_png: bool=True, out_root: str = "runs", run_name: str = "exp1"):
    device = device_auto()
    if ldm_ckpt_path is None:
        ldm_ckpt_path = resolve_ckpt(out_root, run_name, kind="ldm")
    ck = torch.load(ldm_ckpt_path, map_location=device)
    meta = ck.get('meta', None)
    if meta is None:
        raise RuntimeError("LDM ckpt missing meta (datasets/run_root/vae_ckpt). Train with this script.")
    cfg = TrainConfig(**meta.get('config', {}))
    txt_path = meta['datasets'].get(split)
    if txt_path is None: raise RuntimeError(f"Unknown split: {split}")
    ds, dl = build_dataloader_from_txt(txt_path, cfg, device, shuffle=False, drop_last=False)
    idx_map = _map_sorted_idx_to_raw(ds, txt_path)

    vae, ldm, sched = build_models(cfg, device)
    cond_model = str(getattr(cfg, "cond_model", "diffusion")).lower()
    if cond_model != "simple":
        ck_v = torch.load(meta['vae_ckpt'], map_location=device)
        vae.load_state_dict(ck_v['vae_state'] if isinstance(ck_v, dict) and 'vae_state' in ck_v else ck_v, strict=False)
    ldm.load_state_dict(ck['ldm_state'], strict=False)
    # if use_ema and ck.get('ema', None) is not None:
    #     e = EMA(ldm); e.shadow = ck['ema']; e.copy_to(ldm)
    if vae is not None:
        vae.eval()
    ldm.eval()

    use_steps = steps if steps is not None else max(1, cfg.preview_steps)

    errs, ssim_scores, inv_ssim_m1s, top10_maes, cache = [], [], [], [], []
    for batch in DataLoader(ds, batch_size=128, shuffle=False):
        dos = batch["dos"].to(device); ed = batch["ed"].to(device); x0 = batch["band"].to(device)
        if cond_model == "simple":
            _ensure_simple_encoders(ldm, cfg, device)
            pred = ldm(dos, ed, uncond=False).clamp(-1, 1)
            pred01 = (pred + 1.0) / 2.0
        else:
            pred01 = sample_ddim(ldm, dos, ed, sched, device,
                                 band_shape=cfg.BAND_SHAPE, downsample=cfg.downsample, z_ch=cfg.z_ch,
                                 steps=use_steps, eta=cfg.preview_eta, guidance=guidance,
                                 latent_scale=float(cfg.latent_scale), predict_type=cfg.predict_type)
        gt01 = (x0.clamp(-1,1)+1)/2.0
        err_b = torch.mean(torch.abs(pred01 - gt01), dim=(1, 2, 3))
        ssim_b = _batch_ssim(pred01, gt01)
        inv_ssim_m1_b = (1.0 / torch.clamp(ssim_b, min=1e-8)) - 1.0
        top10_b = _batch_top10_mae(pred01, gt01, top_ratio=0.10)
        errs.extend(err_b.cpu().tolist())
        ssim_scores.extend(ssim_b.cpu().tolist())
        inv_ssim_m1s.extend(inv_ssim_m1_b.cpu().tolist())
        top10_maes.extend(top10_b.cpu().tolist())
        for b in range(pred01.size(0)):
            cache.append((gt01[b:b + 1].cpu(), pred01[b:b + 1].cpu()))

        print(len(errs))
    idxs, groups = _select_ranked_indices_quartiles(errs, k_each=2)

    prev_dir = _ensure_preview_dir(meta['run_root'])

    # ↓ list of directory paths
    dir_list = [os.path.dirname(p0) for (p0, _, _) in ds.samples]

    order = np.argsort(errs)  # ascending
    csv_path = os.path.join(prev_dir, f"ldm_{split}_sorted_errors.csv")
    with open(csv_path, "w") as f:
        f.write("index,error,ssim,inv_ssim_m1,top10_mae,dir\n")
        for i in order:
            f.write(
                f"{int(i)},{errs[i]:.8f},{ssim_scores[i]:.8f},{inv_ssim_m1s[i]:.8f},{top10_maes[i]:.8f},{dir_list[i]}\n"
            )

    outs = []
    import matplotlib.pyplot as plt
    for rank, i in enumerate(idxs):
        gt, pr = cache[i]
        gt = gt[0,0].numpy(); pr = pr[0,0].numpy()
        # fig, axs = plt.subplots(1,2, figsize=(6,4))
        fig, axs = plt.subplots(1,3, figsize=(8,4))
        axs[0].imshow(gt.T, aspect="auto", origin="lower"); axs[0].set_title("GT"); axs[0].axis("off")
        axs[1].imshow(pr.T, aspect="auto", origin="lower"); axs[1].set_title(f"Pred L1={errs[i]:.4f}"); axs[1].axis("off")
        axs[2].imshow(np.abs(gt.T - pr.T), aspect="auto", origin="lower", vmin=0, vmax=1); axs[2].set_title("Difference"); axs[2].axis("off")
        plt.tight_layout()
        out = os.path.join(prev_dir, f"ldm_{split}_rank_{rank:02d}_idx{idx_map[i]}.png")
        if save_png: plt.savefig(out, dpi=200); plt.close(fig)
        outs.append(out)
    # return {"indices": idxs, "groups": groups, "errors": errs, "images": outs, "split": split, "error_csv": csv_path}
    return {
        "indices_sorted": idxs,
        "indices_raw": [int(idx_map[i]) for i in idxs],
        "groups": groups,
        "errors": errs,
        "ssim_scores": ssim_scores,
        "inv_ssim_m1s": inv_ssim_m1s,
        "top10_maes": top10_maes,
        "images": outs,
        "split": split,
        "error_csv": csv_path,
    }

@torch.no_grad()
def evaluate_ldm_ckpt_metrics(ldm_ckpt_path: Optional[str] = None, split: str = "test",
                              steps: Optional[int] = None, guidance: float = 1.0,
                              out_root: str = "runs", run_name: str = "exp1") -> dict:
    """
    Evaluate a trained LDM checkpoint on a split and return MAE/SSIM/top-10%-region MAE metrics.
    """
    device = device_auto()
    if ldm_ckpt_path is None:
        ldm_ckpt_path = resolve_ckpt(out_root, run_name, kind="ldm")
    ck = torch.load(ldm_ckpt_path, map_location=device)
    meta = ck.get("meta", None)
    if meta is None:
        raise RuntimeError("LDM ckpt missing meta (datasets/run_root/vae_ckpt). Train with this script.")

    cfg = TrainConfig(**meta.get("config", {}))
    txt_path = meta["datasets"].get(split)
    if txt_path is None:
        raise RuntimeError(f"Unknown split: {split}")

    vae, ldm, sched = build_models(cfg, device)
    cond_model = str(getattr(cfg, "cond_model", "diffusion")).lower()
    if cond_model != "simple":
        ck_v = torch.load(meta["vae_ckpt"], map_location=device)
        vae.load_state_dict(ck_v["vae_state"] if isinstance(ck_v, dict) and "vae_state" in ck_v else ck_v,
                            strict=False)
    ldm.load_state_dict(ck["ldm_state"], strict=False)
    ldm.eval()
    if vae is not None:
        vae.eval()

    use_steps = steps if steps is not None else max(1, int(cfg.preview_steps))
    metrics = _compute_split_metrics(
        ldm=ldm, vae=vae, sched=sched, cfg=cfg, device=device, txt_path=txt_path,
        steps=use_steps, guidance=float(guidance),
    )
    metrics.update({
        "split": split,
        "steps": int(use_steps),
        "guidance": float(guidance),
        "ckpt_path": str(ldm_ckpt_path),
    })
    return metrics

@torch.no_grad()
def _compute_split_metrics(ldm, vae, sched, cfg: TrainConfig, device, txt_path: str,
                           steps: int, guidance: float) -> dict:
    """
    Run DDIM sampling over a full split (.txt) and compute MAE, SSIM, and GT top-10%-region MAE.
    Returns mean/median summaries for each metric.
    """
    ds, _ = build_dataloader_from_txt(txt_path, cfg, device, shuffle=False, drop_last=False)
    dl = DataLoader(ds, batch_size=128, shuffle=False, drop_last=False,
                    pin_memory=(device.type=="cuda"), num_workers=cfg.num_workers)
    maes, ssims, inv_ssim_m1s, top10s = [], [], [], []
    cond_model = str(getattr(cfg, "cond_model", "diffusion")).lower()
    for batch in dl:
        dos = batch["dos"].to(device); ed = batch["ed"].to(device); x0 = batch["band"].to(device)
        if cond_model == "simple":
            _ensure_simple_encoders(ldm, cfg, device)
            pred = ldm(dos, ed, uncond=False).clamp(-1, 1)
            pred01 = (pred + 1.0) / 2.0
        else:
            pred01 = sample_ddim(ldm, dos, ed, sched, device,
                                 band_shape=cfg.BAND_SHAPE, downsample=cfg.downsample, z_ch=cfg.z_ch,
                                 steps=max(1, steps), eta=cfg.preview_eta, guidance=guidance,
                                 latent_scale=float(cfg.latent_scale), predict_type=cfg.predict_type)
        gt01 = (x0.clamp(-1,1)+1)/2.0
        mae_b = torch.mean(torch.abs(pred01 - gt01), dim=(1, 2, 3))
        ssim_b = _batch_ssim(pred01, gt01)
        inv_ssim_m1_b = (1.0 / torch.clamp(ssim_b, min=1e-8)) - 1.0
        top10_b = _batch_top10_mae(pred01, gt01, top_ratio=0.10)
        maes.extend(mae_b.detach().cpu().tolist())
        ssims.extend(ssim_b.detach().cpu().tolist())
        inv_ssim_m1s.extend(inv_ssim_m1_b.detach().cpu().tolist())
        top10s.extend(top10_b.detach().cpu().tolist())
    if len(maes) == 0:
        return {
            "mae_mean": float("nan"), "mae_median": float("nan"),
            "ssim_mean": float("nan"), "ssim_median": float("nan"),
            "inv_ssim_m1_mean": float("nan"), "inv_ssim_m1_median": float("nan"),
            "top10_mae_mean": float("nan"), "top10_mae_median": float("nan"),
        }
    import numpy as _np
    return {
        "mae_mean": float(_np.mean(maes)),
        "mae_median": float(_np.median(maes)),
        "ssim_mean": float(_np.mean(ssims)),
        "ssim_median": float(_np.median(ssims)),
        "inv_ssim_m1_mean": float(_np.mean(inv_ssim_m1s)),
        "inv_ssim_m1_median": float(_np.median(inv_ssim_m1s)),
        "top10_mae_mean": float(_np.mean(top10s)),
        "top10_mae_median": float(_np.median(top10s)),
    }

# ===================== CKPT resolver =====================
def resolve_ckpt(out_root: str, run_name: str, kind: str = "vae") -> str:
    """
    kind: 'vae' | 'ldm'
    Priority: <kind>_best.pt -> <kind>.pt -> latest <kind>*.pt by mtime
    """
    ckpt_dir = os.path.join(out_root, run_name, "checkpoints")
    cand = [
        os.path.join(ckpt_dir, f"{kind}_best.pt"),
        os.path.join(ckpt_dir, f"{kind}.pt"),
    ]
    for p in cand:
        if os.path.isfile(p):
            return p
    gl = sorted(glob.glob(os.path.join(ckpt_dir, f"{kind}*.pt")),
                key=lambda p: os.path.getmtime(p), reverse=True)
    if gl:
        return gl[0]
    raise FileNotFoundError(f"No {kind} checkpoint found in {ckpt_dir}")