# DOS2BandNet

DOS2BandNet is an AI framework that predicts unfolded band structures, also referred to as effective band structures (EBS), from density of states (DOS) and one-dimensional electron diffraction (ED).

---

## 1. Installation

Python 3.12+ is required.

```bash
# Recommended: create/activate a virtual environment first

pip install -r requirements.txt
# Please install PyTorch>=2.8 with CUDA. https://pytorch.org/get-started/locally/
pip install .
```

---

## 2. Data preprocessing

`scripts/preprocessing.py` generates the following files for each target `band*` directory:

- `diffrac.npy`: 1D electron diffraction intensity, processed from `POSCAR`
- `dos.npy`: DOS vector (or site-projected array), processed from `vasprun.xml`
- `ebs.npy`: 2D band intensity map, processed from `EBS.dat` (by VASPKIT)

### 2.1 Formulas (GitHub-rendered math)

#### (a) DOS interpolation and normalization

Given raw DOS samples $(E_i, D_i)$, interpolate onto a uniform energy grid $\tilde{E}_j$:

$$
\tilde{D}(\tilde{E}_j)=\mathrm{Interp}\big((E_i,D_i),\tilde{E}_j\big)
$$

If Gaussian smoothing is applied:

$$
\tilde{D}_{\sigma}(E)=\big(\tilde{D}*\mathcal{G}_{\sigma}\big)(E),\qquad
\mathcal{G}_{\sigma}(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-x^2/(2\sigma^2)}
$$

Then max-normalize:

$$
D_{\text{norm}}(E)=\frac{\tilde{D}_{\sigma}(E)}{\max_E \tilde{D}_{\sigma}(E)}
$$

#### (b) `EBS.dat` -> band map (`ebs.npy`)

`EBS.dat` is generated from the VASP outputs using **VASPKIT**.

For each point $(k_m,E_m,w_m)$ in `EBS.dat`, accumulate a 2D Gaussian kernel:

$$
I(k,E)=\sum_m w_m
\exp\!\left[-\frac{(k-k_m)^2}{2\sigma_k^2}\right]
\exp\!\left[-\frac{(E-E_m)^2}{2\sigma_E^2}\right]
$$

In code, $\sigma_k$ is scaled by path/energy span:

$$
\sigma_k \propto \sigma_E\frac{K_{\text{span}}}{E_{\text{span}}}
$$

Final normalization:

$$
I_{\text{norm}}(k,E)=\frac{I(k,E)}{\max_{k,E} I(k,E)}
$$

#### (c) 1D electron diffraction (`diffrac.npy`)

Reflections are selected along zone axis $[u, v, w]$, then broadened with Gaussian peaks:

$$
I(q)=\sum_h A_h\exp\!\left[-\frac{(q-q_h)^2}{2\sigma_q^2}\right],
\qquad
\sigma_q=\frac{\mathrm{FWHM}_q}{2\sqrt{2\ln 2}}
$$

Then max-scale the final intensity.

---

## 3. Usage examples

The pre-trained model and dataset are available on Zenodo (DOI:10.5281/zenodo.19490693).

Please download the dataset from Zenodo, extract it. Then run:

```bash
cd DOS2band_dataset
```

### 3.1 Preprocessing

The Zenodo dataset also provides preprocessed data.  
Therefore, this step is not required when using the released dataset.

```bash
python scripts/preprocessing.py \
  --prefix1 vac \
  --tasks diffraction \
  --dos-emin -10 \
  --fwhm-q 0.08 \
  --nproc 4
```

### 3.2 Finetuning (`scripts/train.py`)

```bash
python /path/to/dos2bandnet/scripts/train.py \
  --mode finetune \
  --out-root ./trained-model \
  --pretrained-id full-input_model \
  --finetune-base ./TaTe2_CDW/33SC_MD \
  --finetune-split 0.8,0.1,0.1 \
  --finetune-lr 5e-5 \
  --prefix1 orb \
  --prefix2 ebs \
  --epochs-diff 200 \
  --batch-size 32 \
  --min-lr 1e-7 \
  --warmup-epochs 60
```

### 3.3 Inference/visualization (`scripts/forward.py`)

```bash
python /path/to/dos2bandnet/scripts/forward.py full-input_model \
  --out_root ./trained-model \
  --dirs ./TaTe2_CDW/33CDW/ebs_100 \
  --steps 50 \
  --guidance 1.0 \
  --view_range -5 5 \
  --zoom_range 0 4 \
  --size-cm 12 9 \
  --hide-y-numbers \
  --dos-fwhm 0.15 \
  --ed-fwhm 0.08 \
  --ldm_ckpt ./trained-model/full-input_model/finetune_TaTe2_CDW/checkpoints/ldm_best.pt
```

---

## 4. Key CLI argument summary

### `scripts/preprocessing.py`

- `--base-dir`: data root
- `--prefix1`, `--prefix2`: directory prefixes
- `--tasks`: `diffraction,dos,ebs`
- `--dos-emin`, `--dos-emax`, `--dos-grid`: DOS grid settings
- `--tt-min`, `--tt-max`, `--dif-grid`, `--fwhm-q`: diffraction settings
- `--nk`, `--nE`, `--sigma-E-ebs`: EBS settings
- `--nproc`: number of worker processes

### `scripts/train.py`

- `--mode`: `vae|ldm|both|finetune`
- `--pretrained-id`: pretrained run ID for finetune
- `--finetune-base`, `--finetune-split`, `--finetune-lr`
- `--batch-size`, `--epochs-vae`, `--epochs-diff`

### `scripts/train_wandb.py`

- `--mode`: `create|agent|agent_pool|replay`
- `--project`, `--entity`, `--sweep_id`
- `--pool_gpus`, `--agents_per_gpu`, `--count_per_agent`

### `scripts/forward.py`

- positional `run_name`
- `--dirs`: inference target directories
- `--steps`, `--guidance`
- `--view_range`, `--zoom_range`, `--size-cm`
- `--dos-fwhm`, `--ed-fwhm`
- `--ldm_ckpt`: checkpoint path

## Contributors

- [Yeongrok Jin](https://github.com/Zerok-95) 
- [Mizoguchi Lab](https://github.com/nmdl-mizo)

## License

This project is licensed under the MIT License.
