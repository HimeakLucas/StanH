# STanH : Parametric Quantization for Variable Rate Learned Image Compression


Pytorch implementation of the paper "**STanH : Parametric Quantization for Variable Rate Learned Image Compression**",  accepted at TIP. This repository is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI) and [STF](https://github.com/Googolxx/STF)

[Paper link](https://arxiv.org/abs/2410.00557)


<div align="center">
<img src="imgs/stanh.png" alt="stanh" width="400"/>
<p><em>STanH activation function with 5 quantization
levels and for increasing values of inverse temperature β.</em></p>
</div>


## Abstract
In end-to-end learned image compression, encoder
and decoder are jointly trained to minimize a R + λD cost
function, where λ controls the trade-off between rate of the
quantized latent representation and image quality. Unfortunately,
a distinct encoder-decoder pair with millions of parameters must
be trained for each λ, hence the need to switch encoders and
to store multiple encoders and decoders on the user device for
every target rate. This paper proposes to exploit a differentiable
quantizer designed around a parametric sum of hyperbolic
tangents, called STanH , that relaxes the step-wise quantization
function. STanH is implemented as a differentiable activation
layer with learnable quantization parameters that can be plugged
into a pre-trained fixed rate model and refined to achieve different
target bitrates. Experimental results show that our method
enables variable rate coding with comparable efficiency to the
state-of-the-art, yet with significant savings in terms of ease of
deployment, training time, and storage costs.

<div align="center">
<img src="imgs/arch.png" alt="arch" width="600"/>
<p><em>The reference learned image compression architecture Zou22  (CNN-based architecture) with two STanH layers for quantizing the main latent space y and the hyperprior latent space z.</em></p>
</div>

## Preparation 
In order to use this code, You can create a conda environment, with CUDA if you use GPU (suggested)

```
conda create -n $NAME python=3.8
conda activate $NAME

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```


## Validate

Download our pretrained model (based on Zou2022) in the following directory from [here](https://drive.google.com/drive/folders/1LJ6nmQZJyMaJKFzr-sb2C9m9oxHE5pE5).



```
cd src 

python demo.py \
--image_path #path-for-the-image-to-encode \ 
--model_checkpoint #path-for-the-checkpoint-of-the-anchor \
--stanh_path #path-where-stanhs-are-saved \ 
--wandb_log #if-ypu-want-wandb-plot \ 
--entropy_estimation #estimation-of-entropy (faster) 
--path_save #path-where-to-save-results \
--interpolation #to include-interpolated-points
--device cuda
```

Until now, we use Torcach to perform Arithmetic coding (very slow), in future code will be adapted to use RANS as standard model in CompressAI library

---

## Research fork — how far does a parametric quantizer go as a domain adapter?

Undergraduate research fork (PIBIC). The question: the STanH quantizer has ~320 parameters,
so if it could specialize a frozen backbone to a new domain it would be an extremely cheap
adapter. Does it?

The anchor backbone (Zou22/WACNN, ~75 M parameters) released by the authors is kept
**frozen** and only a chosen block is refined per domain. Five adaptation modes span the
cost axis:

| mode | trainable | what it refines |
|---|---|---|
| `quantizer` | ~320 | STanH parameters (`w`, `b`) only |
| `encoder` | 6.9 M | analysis transform |
| `decoder` | 6.9 M | synthesis transform |
| `encoder_hyper` | 22 M | analysis transform + hyperprior (unfreezes the rate model) |
| `full` | 75 M | whole backbone |

Six target domains: chest x-ray, documents, retina fundus, aerial (DIOR), OCT and screen
content (RICO). Every cell is evaluated on its target domain and, cross-domain, on Kodak,
against the authors' generic derivations and the VVC reference software (VTM).

**The answer is negative, and that is the point**: quantizer-only does not adapt. The
residual is resolved but tiny, and it flips sign with the quality metric (+1.23% BD-Rate in
PSNR, -0.27% in MS-SSIM), an order of magnitude below the smallest adapter that works. The
fork also reports where the aggregate BD-Rate itself is unreliable — changing storage
precision of the *same trained model* (fp32 -> fp16) moves it 0.42 p.p. while the RD curve
stays put.

### Methodology notes

- **Leakage.** Splits are grouped by patient/volume, not by file. Where the source dataset
  keeps the group id (x-ray, OCT, retina), evaluation samples exclude every group seen in
  train *or* validation; for the other three the collection scripts discard the origin id,
  so only per-image disjointness can be guaranteed, and that is declared.
- **BD-Rate guards.** BD-Rate is reported only when the overlap window survives two checks
  (bootstrap floor of the window >= 1 dB, and a common floor when comparing samples).
  Otherwise the unit is the bpp-matched PSNR delta with a per-image paired bootstrap CI.
- `plots/analyze_finetuned.py` drops non-Pareto points *before* fixing the window, which is
  discontinuous: 0.001 bpp can move BD-Rate by tens of points. Point counts and window
  floors are reported alongside every comparison.

### Running

All scripts run **from the repository root** with `src` on the path:

```bash
conda activate stanh
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# evaluation (anchor + derivations, or anchor + delta)
python eval/evaluate_kodak.py                     # generic RD curve on Kodak
python eval/eval_full.py --models_dir models/<cell> --dataset <test_dir> \
    --limit 150 --out_json results/<cell>_rd.json
python eval/eval_vtm.py                           # VTM baseline (needs VVCSoftware_VTM built)

# training one adapter spectrum (8 rate points) for a domain
bash train/run_spectrum.sh <domain> <dataset_dir> [patch] [batch] [modes...]
bash train/run_spectrum.sh retina datasets/retina 256 16 encoder decoder full

# analysis and figures
python plots/analyze_finetuned.py --target_json <cell>_rd.json --cross_json <cell>_on_kodak.json
python plots/fig_dois_regimes.py
```

`eval/eval_full.py` defaults to `--limit 24`; the evaluation protocol here is 150 images, so
that flag is mandatory for any comparable number.

Datasets live under `datasets/` and checkpoints under `models/` (both git-ignored, except
the small paper derivations). The 300 MB anchor must be downloaded from the authors'
[Drive link](https://drive.google.com/drive/folders/1LJ6nmQZJyMaJKFzr-sb2C9m9oxHE5pE5)
into `models/original_paper/STanH/anchor/`.

Figure scripts and the report are in Portuguese (axis labels, captions); the code and its
comments are in English.