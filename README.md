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

## Research fork — STanH as a domain adapter

Undergraduate research fork (PIBIC). The STanH quantizer has ~320 parameters: if that were
enough to specialize a frozen backbone to a new domain, it would be an extremely cheap
adapter. It is not, and measuring how far it falls short is the result.

The anchor backbone (Zou22/WACNN, ~75 M) stays **frozen** while one block is refined per
domain, across five points of the cost axis:

| mode | trainable | refines |
|---|---|---|
| `quantizer` | ~320 | STanH parameters (`w`, `b`) |
| `encoder` | 6.9 M | analysis transform |
| `decoder` | 6.9 M | synthesis transform |
| `encoder_hyper` | 22 M | analysis transform + hyperprior |
| `full` | 75 M | whole backbone |

Six domains — chest x-ray, documents, retina, aerial (DIOR), OCT, screen content (RICO) —
each evaluated on its own test set and, cross-domain, on Kodak, against the authors' generic
derivations and VTM. Quantizer-only leaves a residual that is resolved but tiny, and that
flips sign with the metric (+1.23% BD-Rate in PSNR, -0.27% in MS-SSIM): an order of magnitude
below the smallest adapter that works. The fork also reports where aggregate BD-Rate is
itself unreliable — changing the storage precision of the *same trained model* (fp32 to
fp16) moves it 0.42 p.p. while the RD curve stays put.

Two conventions for reading the numbers. BD-Rate is reported only when the curve overlap
window survives a 1 dB bootstrap floor; otherwise the unit is the bpp-matched PSNR delta
with a per-image paired bootstrap CI. Splits are grouped by patient/volume wherever the
source dataset keeps that id (x-ray, OCT, retina); where it does not (documents, DIOR, RICO)
only per-image disjointness is claimed.

### Running

The RD curves behind every figure and table are versioned in `results/`, so the figures
reproduce without retraining. `requirements.txt` pins the environment they came from.

```bash
conda activate stanh
export PYTHONPATH=src

python plots/fig_dois_regimes.py                   # a figure, straight from results/
python eval/eval_full.py --models_dir models/<cell> \
    --dataset <test_dir> --limit 150 --out_json results/<cell>_rd.json
bash train/run_spectrum.sh <domain> <dataset_dir> [patch] [batch] [modes...]
```

`eval/eval_full.py` defaults to `--limit 24`; the protocol here is 150 images, so that flag
is mandatory for a comparable number.

Datasets live under `datasets/` and checkpoints under `models/` (both git-ignored, except
the small paper derivations). The 300 MB anchor must be downloaded from the authors'
[Drive link](https://drive.google.com/drive/folders/1LJ6nmQZJyMaJKFzr-sb2C9m9oxHE5pE5)
into `models/original_paper/STanH/anchor/`.