# Style-Enhanced CycleGAN for Unpaired Image-to-Image Translation: A New Approach to Medical Image Fusion Refinement

Code for the image-translation stage (white-light imaging, WLI, to narrow-band imaging, NBI) of the paper

> M. Einlou, S. Moghtaderi, K. A. Wahid, and K. E. Lukong, "Style-Enhanced CycleGAN for Unpaired Image-to-Image Translation: A New Approach to Medical Image Fusion Refinement," *IEEE Access*, vol. 14, pp. 56986–57001, 2026. [doi:10.1109/ACCESS.2026.3683591](https://doi.org/10.1109/ACCESS.2026.3683591)

Mokarrameh Einlou<sup>1,\*</sup>, Shiva Moghtaderi<sup>1,\*</sup>, Khan A. Wahid<sup>1</sup>, Kiven E. Lukong<sup>2</sup>

<sup>1</sup> Department of Electrical and Computer Engineering, University of Saskatchewan, Saskatoon, SK S7N 5A9, Canada<br>
<sup>2</sup> Department of Biochemistry, Microbiology and Immunology, University of Saskatchewan, Saskatoon, SK S7N 5E5, Canada<br>
<sup>\*</sup> Mokarrameh Einlou and Shiva Moghtaderi contributed equally to this work.<br>
Corresponding author: Shiva Moghtaderi (shiva.moghtaderi@usask.ca)

## Overview

The paper proposes a multimodal image fusion framework with two stages (Section III, Fig. 2). In the first stage, a realistic narrow-band image is generated from a white-light input with a Style-Enhanced CycleGAN for unpaired image-to-image translation. In the second stage, the white-light image and the generated narrow-band image are fused with a PCNN-based strategy.

This repository contains the first stage:

- the four experiments of Table 3: CycleGAN and Style-Enhanced CycleGAN, each with a ResNet-9 generator and a U-Net generator with seven levels;
- the comparison models of Tables 4 and 6: SPA GAN [40] (the authors' re-implementation) and CUT [60] (run with its official implementation, which is not included; a timing script is included);
- the FID and KID evaluation (Tables 4 and 5);
- the parameter counts and timing (Table 6).

It does not contain:

- the image-fusion stage (image decomposition with the NLM filter and the guided filter, ST-G-PCNN fusion; Section III-C, Figs. 5–8, Tables 7–9). For the authors' fusion methods, see [Related work](#related-work);
- trained weights;
- images. The datasets are available from their providers (see [Data](#data)). The lists of the images used in the paper are included, with SHA-256 hashes, together with a script that rebuilds the training and test folders from the official downloads.

## Repository layout

```
cycle_train.py, cycle_test.py             CycleGAN, ResNet-9 (Table 3, experiment 1)
cycle_train_unet.py, cycle_test_unet.py   CycleGAN, U-Net with 7 levels (experiment 2)
style_train.py, style_test.py             Style-Enhanced CycleGAN, ResNet-9 (experiment 3)
style_train_unet.py, style_test_unet.py   Style-Enhanced CycleGAN, U-Net with 7 levels (experiment 4)
spa_train.py, spa_test.py                 SPA GAN (comparison; the authors' re-implementation)
measure_inference_time.py                 inference time per image (Table 6)
measure_cut_inference_time.py             inference time of CUT (Table 6); runs inside a CUT checkout
param_count.ipynb                         parameter counts (Table 6)
prepare_data.py                           builds the dataset folders from the official downloads
manifests/                                lists of the images used in the paper
models/                                   network definitions
utils/                                    data loading, logging, FID/KID evaluation, loss curves
plato_scripts/                            Slurm scripts for the Plato cluster (University of Saskatchewan)
```

## Environment

The experiments were conducted on Plato, a Linux-based high-performance computing cluster hosted at the University of Saskatchewan. Each experiment was executed on a single NVIDIA A100 GPU with 80 GB of memory. The models were implemented in PyTorch 2.5 with CUDA 12.2 support (Section IV-B).

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # install the PyTorch 2.5 build for your CUDA version first
```

`requirements.txt` pins PyTorch to 2.5, as stated in the paper; the paper does not state the versions of the other packages.

The training scripts send losses and sample images to a [Visdom](https://github.com/fossasia/visdom) server. Start one with `python -m visdom.server`, or set `VISDOM_SERVER` and `VISDOM_PORT` (default `http://localhost`, port 8097). Without a server, training still runs and the Visdom client prints connection errors to the log.

`style_train.py` and `style_train_unet.py` download the ImageNet weights of VGG-19 from torchvision on first use.

## Data

As described in Section III-A of the paper, the primary data source is the PICCOLO dataset [35], expanded with two publicly available sources: NBI-InfFrames [36] and the Endoscopy Artefact Detection (EAD) dataset [37]–[39]. Because EAD is multimodal, only WLI and NBI images were selected from it, manually; images that were excessively blurry or exhibited unusual lighting conditions or significant artifacts were excluded, while some images with mild blur or artifacts remained in the training set. For PICCOLO, the original training and validation sets were combined into a single training set. NBI-InfFrames images were added to the NBI part of this training set, and EAD images to both the WLI and NBI parts. The test set is the PICCOLO test set in both settings.

Table 1 of the paper: "Composition of the cleaned and curated datasets used for training and evaluation, showing the number of patient cases and WLI/NBI images."

| Dataset | Number of patient cases | Number of WLI | Number of NBI |
|---|---|---|---|
| PICCOLO [35] | 40 | 2128 | 1305 |
| NBI-InfFrames [36] | 18 | 0 | 180 |
| EAD [37]–[39] | N/A | 989 | 19 |

Table 2 of the paper: "Number of training and test images for each modality before and after dataset expansion, showing the original PICCOLO dataset and the combined dataset incorporating PICCOLO, NBI-InfFrames, and EAD."

| Dataset setup | Modality | Training images | Test images | Image list |
|---|---|---|---|---|
| PICCOLO [35] | WLI | 1936 | 192 | `manifests/piccolo_only.csv` |
| | NBI | 1164 | 141 | |
| Combined | WLI | 2925 | 192 | `manifests/combined.csv` |
| | NBI | 1363 | 141 | |

No images are distributed with this repository. Download the datasets from their providers and accept their terms:

| Dataset | Where to get it | Terms stated by the provider | Images used |
|---|---|---|---|
| PICCOLO [35] | [Basque Biobank](https://www.biobancovasco.bioef.eus/en/Sample-and-data-catalog/Databases/PD178-PICCOLO-EN.html): access form or request to the Biobank | Research and educational use only; commercial use forbidden without prior written permission. The Biobank sets the conditions of access. | All 3,433 images (`piccolo dataset-release0.1`). |
| NBI-InfFrames [36] | [Zenodo, doi:10.5281/zenodo.1162784](https://doi.org/10.5281/zenodo.1162784) (`FRAMES.zip`) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) | The 180 informative frames (class `I` of the three folds). |
| EAD [37]–[39] | [Mendeley Data, doi:10.17632/c7fjbxcgj9](https://data.mendeley.com/datasets/c7fjbxcgj9), version 4 (`EndoCV2020_Endoscopy_Artefact_Detection_Segmentation_training_data_v2.zip`) | [CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/) (shown on the [version 4 page](https://data.mendeley.com/datasets/c7fjbxcgj9/4)) | 989 WLI and 19 NBI frames from `EAD2020_dataType_framesOnly/frames`. |

Cite the datasets as their providers request.

Each manifest row gives the split, the domain (A = WLI, B = NBI), the file name, the source dataset, the path inside the download, the SHA-256 hash of the file and its size in pixels. The images are used at their original resolution (PICCOLO 854 × 480 or 1920 × 1080; all test images are 854 × 480) and resized by the data loaders. The hashes were computed from the authors' copies of the downloads; the EAD version 4 zip has 504,211,398 bytes and SHA-256 `100885d49c2d047fad7b6a5102512824d1b21c6c3feac8134175665485dc0e5f`.

Build the folders with `prepare_data.py`. It copies the listed files unchanged and checks every hash:

```
# combined dataset
python prepare_data.py --setting combined --out datasets/scgan_comp \
    --piccolo_root "/path/to/piccolo dataset-release0.1" \
    --nbi_infframes_root /path/to/NBI-InfFrames \
    --ead_root /path/to/EndoCV2020_Endoscopy_Artefact_Detection_Segmentation_training_data_v2

# PICCOLO dataset only
python prepare_data.py --setting piccolo_only --out datasets/scgan \
    --piccolo_root "/path/to/piccolo dataset-release0.1"
```

The output has the layout the scripts read: `<out>/train/A`, `train/B`, `test/A`, `test/B`. `--layout cut` writes `trainA`, `trainB`, `testA` and `testB` for the CUT code instead. `--check_only` checks the downloads without writing anything. See `python prepare_data.py --help`.

The commands below use `datasets/scgan_comp/` for the combined dataset. The Slurm scripts in `plato_scripts/` use two folder names for it: `datasets/scgan_comp/` and `datasets/scgan_complete/`.

## Experiments

Table 3 of the paper: "Summary of experimental setups and model variants."

| Experiment ID | Model type | Generator backbone | Style modulation (mapping layers + AdaIN injection) | Latent dim | Scripts |
|---|---|---|---|---|---|
| 1 | CycleGAN (baseline) | ResNet-9 | no | – | `cycle_train.py`, `cycle_test.py` |
| 2 | CycleGAN (baseline) | U-Net (7 levels) | no | – | `cycle_train_unet.py`, `cycle_test_unet.py` |
| 3 | Style-Enhanced CycleGAN | ResNet-9 | yes | 512 | `style_train.py`, `style_test.py` |
| 4 | Style-Enhanced CycleGAN | U-Net (7 levels) | yes | 512 | `style_train_unet.py`, `style_test_unet.py` |

Training settings as stated in Section IV of the paper: the input images were resized to 256 × 256 pixels, randomly cropped, normalized to the range of [−1, 1], and augmented with horizontal flips and random rotations of ±10°. All networks were optimized using Adam (β1 = 0.5, β2 = 0.999) with an initial learning rate of 0.0002, linearly decayed to zero starting at epoch 50. Training was conducted for 200 epochs with a batch size of 1. In the style-enhanced configurations, the latent space dimension was 512 and the mapping network comprised eight fully connected layers. Where the scripts' defaults differ from this description, see [Differences between the code and the paper's description](#differences-between-the-code-and-the-papers-description).

The commands below use the scripts' default settings, as in the Slurm scripts:

```
# experiment 1: CycleGAN, ResNet-9                     -> checkpoints/cycle_res9_comp/
python cycle_train.py --dataroot datasets/scgan_comp/ --cuda

# experiment 2: CycleGAN, U-Net (7 levels)              -> checkpoints/cycle_unet/
python cycle_train_unet.py --dataroot datasets/scgan_comp/ --cuda

# experiment 3: Style-Enhanced CycleGAN, ResNet-9       -> checkpoints/scgan_res9/
python style_train.py --dataroot datasets/scgan_comp/ --cuda

# experiment 4: Style-Enhanced CycleGAN, U-Net          -> checkpoints/scgan_unet/
python style_train_unet.py --dataroot datasets/scgan_comp/ --cuda

# SPA GAN (comparison; requires a CUDA GPU)             -> checkpoints/spa_res9/
python spa_train.py --dataroot datasets/scgan_comp/ --cuda
```

Dataset variability experiment (Section IV-A, Table 5): the ResNet-based CycleGAN was trained once on the PICCOLO dataset only and once on the combined dataset. For the PICCOLO-only run, use `--dataroot datasets/scgan/`. `cycle_train.py` always writes to `checkpoints/cycle_res9_comp/`, so move that folder before the second run.

Loss curves (Fig. 9): `python utils/loss_vis.py --log_file <training log> --out loss_curves.svg` plots the per-epoch averages of the logged losses of a CycleGAN or Style-Enhanced CycleGAN run.

Run `python <script>.py --help` for all options of a script.

## Translating the test set

Each test script reads `test/A` and `test/B`, resizes every image to 256 × 256 and writes the translations, with the input file names, to `<checkpoint folder>/B` (WLI → NBI) and `<checkpoint folder>/A` (NBI → WLI).

```
python cycle_test.py       --dataroot datasets/scgan_comp/ --cuda   # -> checkpoints/cycle_res9_comp/{A,B}
python cycle_test_unet.py  --dataroot datasets/scgan_comp/ --cuda   # -> checkpoints/cycle_unet/{A,B}
python style_test.py       --dataroot datasets/scgan_comp/ --cuda   # -> checkpoints/scgan_res9/{A,B}
python style_test_unet.py  --dataroot datasets/scgan_comp/ --cuda   # -> checkpoints/scgan_unet/{A,B}
python spa_test.py         --dataroot datasets/scgan_comp/ --cuda   # -> checkpoints/spa_res9/{A,B}
```

## Evaluation (FID and KID)

The paper evaluates WLI-to-NBI translation with the Fréchet Inception Distance (FID) [59] and the Kernel Inception Distance (KID), which estimates the squared maximum mean discrepancy with a polynomial kernel (Section V). Lower values are better. The real images are the 141 NBI test images (`test/B`); the generated images are the 192 NBI images translated from the WLI test images (`<checkpoint folder>/B`).

```
python utils/evaluate_models.py --gt datasets/scgan_comp/test/B --kid_subset_size 100 \
    --models CycleGAN_ResNet9=checkpoints/cycle_res9_comp/B CycleGAN_UNet=checkpoints/cycle_unet/B \
             SCGAN_ResNet9=checkpoints/scgan_res9/B SCGAN_UNet=checkpoints/scgan_unet/B \
             SPA=checkpoints/spa_res9/B
```

FID values depend on the implementation. `utils/evaluate_models.py`, which the Slurm scripts call, computes FID and KID with [torch-fidelity](https://github.com/toshas/torch-fidelity) (`calculate_metrics`) with these settings:

- feature extractor `inception-v3-compat`, 2048-dimensional pool features; images are resized inside the extractor to 299 × 299;
- the real images are read at their original size (854 × 480), the generated images at 256 × 256;
- KID: 100 subsets of 100 images (`--kid_subset_size 100`), polynomial kernel of degree 3, γ = 1/2048, coef0 = 1, random seed 2020 (torch-fidelity defaults);
- the paper does not state the torch-fidelity version.

The script also prints the Inception Score, which the paper does not report. Add `--cuda` to run the feature extractor on a GPU.

## Comparison models

**SPA GAN [40].** `models/spa_gan.py`, `spa_train.py` and `spa_test.py` are the authors' re-implementation of SPA GAN with a ResNet generator of 9 residual blocks; they are not the official SPA GAN code. Their differences from the published method are listed in [Differences between the code and the paper's description](#differences-between-the-code-and-the-papers-description).

**CUT [60].** CUT with a ResNet-9 generator was run with its official implementation, <https://github.com/taesungp/contrastive-unpaired-translation>, which is not part of this repository. `prepare_data.py --layout cut` writes the data in the folder layout that code expects. `measure_cut_inference_time.py` measures CUT's inference time when copied into a checkout of that code (see the docstring at the top of the file).

## Parameters and timing (Table 6)

- **Parameters.** `param_count.ipynb` prints a `torchinfo` summary of each generator; run it from the repository root. The counts are per generator (one translation direction): 11,378,179 (CycleGAN, ResNet-9), 41,825,667 (CycleGAN, U-Net), 18,207,235 (Style-Enhanced CycleGAN, ResNet-9), 48,523,395 (Style-Enhanced CycleGAN, U-Net) and 11,570,308 (SPA GAN, 9 residual blocks). Of the Style-Enhanced totals, 2,101,248 parameters belong to the mapping network.
- **Training time per epoch.** Each training script prints `Average training time per epoch` at the end. The time covers the whole epoch, including data loading and Visdom logging.
- **Inference time.** `measure_inference_time.py` times the WLI-to-NBI generator of each model on one 256 × 256 image: 10 warm-up passes, then 100 timed passes with CUDA events. It needs a CUDA GPU and the trained generators (paths can be set with options; see `--help`):

```
python measure_inference_time.py --image datasets/scgan_comp/test/A/003_VP3_frame0030.png
```

## Results reported in the paper

The values below are quoted from the paper; they were not regenerated for this release.

Table 4: "FID and KID scores across different experimental setups and model variants."

| Model type | Generator backbone | FID | KID |
|---|---|---|---|
| CycleGAN [48] | ResNet-9 | 156.32 | 0.0633 ± 0.0036 |
| CycleGAN [48] | U-Net | 154.96 | 0.0651 ± 0.0046 |
| Style-Enhanced CycleGAN | ResNet-9 | 150.5 | 0.0550 ± 0.0037 |
| Style-Enhanced CycleGAN | U-Net | 180.22 | 0.1040 ± 0.0050 |
| SPA GAN [40]¹ | ResNet-9 | 204.78 | 0.1434 ± 0.0046 |
| CUT [60] | ResNet-9 | 158.93 | 0.0708 ± 0.0041 |

Table 5: "Impact of dataset composition on WLI-to-NBI translation performance. Comparison of FID and KID scores for models trained on the PICCOLO dataset versus the combined dataset (PICCOLO + NBI-InfFrames + EAD)."

| Dataset setup | FID score | KID score |
|---|---|---|
| PICCOLO Dataset [35] | 166.16 | 0.0766 ± 0.0044 |
| Combined Dataset | 156.32 | 0.0633 ± 0.0036 |

Table 6: "Comparison of model complexity and computational efficiency for different experimental setups and model variants."

| Experiment | Parameters (M) | Training time / epoch (sec) | Inference time / image (ms) |
|---|---|---|---|
| CycleGAN [48] (ResNet) | 11.3 | 630.72 | 6.24 ± 0.01 |
| CycleGAN [48] (U-Net) | 41.8 | 517.59 | 1.97 ± 0.12 |
| Style-Enhanced CycleGAN (ResNet) | 18.2 | 708.67 | 6.56 ± 0.03 |
| Style-Enhanced CycleGAN (U-Net) | 48.5 | 619.41 | 3.73 ± 0.09 |
| SPA GAN [40]¹ | 11.5 | 912.69 | 4.08 ± 0.01 |
| CUT [60] | 11.3 | 333 | 8.80 ± 0.04 |

¹ Computed with the authors' re-implementation of SPA GAN (see [Comparison models](#comparison-models)).

## Differences between the code and the paper's description

The code is released as it was used. Where it differs from the description in the paper, the difference is listed here:

1. The paper describes adversarial, cycle-consistency and identity losses (Algorithm 1). `style_train_unet.py` (experiment 4) also adds a VGG-19 style loss (weight `--lambda_style`, default 0.01) to the generator loss. `style_train.py` (experiment 3) creates this loss but does not use it.
2. In Algorithm 1, the style codes come from the mapping network, which is updated with the generators. In experiment 4, `UnetGenerator.forward` uses the stored running average of the style vector by default (`use_w_avg=True`), so the mapping network is not trained and the sampled latent vector does not change the output. In both Style-Enhanced models, the identity terms use the running average instead of the sampled style codes.
3. The paper describes the style vector as injected into multiple layers of the decoder. In the code, AdaIN is applied in the nine residual blocks of the ResNet generator (as drawn in Fig. 4), and after the convolutions of both the encoder and the decoder of the U-Net generator.
4. The default learning rate of `cycle_train.py` and `cycle_train_unet.py` is 0.0001; the other training scripts use 0.0002.
5. Training images are resized so that the shorter side is 286 pixels before the random 256 × 256 crop; test images are resized to 256 × 256.
6. SPA GAN: the re-implementation predicts its attention map with a head on the generator and has no feature-map loss, while the published SPA GAN computes attention in the discriminator. It is trained on resized, flipped frames without cropping or rotation, with an identity-loss weight of 0.5 and a fixed random seed. `measure_inference_time.py` times an SPA GAN generator with 6 residual blocks; training, testing and the parameter count use 9.
7. The Slurm script of experiment 4 (`plato_scripts/scgan_unet.sh`) requests a 40 GB slice of an A100 (`3g.40gb`).
8. Except for SPA GAN, no random seed is set, and `style_test.py` draws a new latent vector for every test image, so repeated runs give different images and scores.

## Related work

The fusion stage of the paper builds on the authors' earlier work [9], [53]:

- [9] S. Moghtaderi, M. Einlou, K. A. Wahid, and K. E. Lukong, "Advancing multimodal medical image fusion: An adaptive image decomposition approach based on multilevel guided filtering," *Roy. Soc. Open Sci.*, vol. 11, no. 4, Apr. 2024, Art. no. 231762. [doi:10.1098/rsos.231762](https://doi.org/10.1098/rsos.231762)
- [53] S. Moghtaderi, O. Yaghoobian, K. A. Wahid, and K. E. Lukong, "Endoscopic image enhancement: Wavelet transform and guided filter decomposition-based fusion approach," *J. Imag.*, vol. 10, no. 1, p. 28, Jan. 2024. [doi:10.3390/jimaging10010028](https://doi.org/10.3390/jimaging10010028)

## References

Numbering as in the paper.

- [35] L. F. Sánchez-Peralta, J. B. Pagador, A. Picón, Á. J. Calderón, F. Polo, N. Andraka, R. Bilbao, B. Glover, C. L. Saratxaga, and F. M. Sánchez-Margallo, "PICCOLO white-light and narrow-band imaging colonoscopic dataset: A performance comparative of models and datasets," *Appl. Sci.*, vol. 10, no. 23, p. 8501, Nov. 2020.
- [36] S. Moccia, G. O. Vanone, E. D. Momi, A. Laborai, L. Guastini, G. Peretti, and L. S. Mattos, "Learning-based classification of informative laryngoscopic frames," *Comput. Methods Programs Biomed.*, vol. 158, pp. 21–30, May 2018.
- [37] S. Ali, F. Zhou, C. Daul, B. Braden, A. Bailey, S. Realdon, J. East, G. Wagniéres, V. Loschenov, E. Grisan, W. Blondel, and J. Rittscher, "Endoscopy artifact detection (EAD 2019) challenge dataset," 2019, arXiv:1905.03209.
- [38] S. Ali, F. Zhou, A. Bailey, B. Braden, J. East, X. Lu, and J. Rittscher, "A deep learning framework for quality assessment and restoration in video endoscopy," *Med. Image Anal.*, vol. 68, Feb. 2021, Art. no. 101900.
- [39] S. Ali et al., "An objective comparison of detection and segmentation algorithms for artefacts in clinical endoscopy," *Sci. Rep.*, vol. 10, no. 1, p. 2748, Feb. 2020.
- [40] H. Emami, M. M. Aliabadi, M. Dong, and R. B. Chinnam, "SPA-GAN: Spatial attention GAN for image-to-image translation," *IEEE Trans. Multimedia*, vol. 23, pp. 391–401, 2021.
- [48] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, "Unpaired image-to-image translation using cycle-consistent adversarial networks," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, Oct. 2017, pp. 2242–2251.
- [49] T. Karras, S. Laine, and T. Aila, "A style-based generator architecture for generative adversarial networks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Jun. 2019, pp. 4396–4405.
- [59] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, "GANs trained by a two time-scale update rule converge to a local Nash equilibrium," in *Proc. Adv. Neural Inf. Process. Syst.*, vol. 30, 2017, pp. 6626–6637.
- [60] T. Park, A. A. Efros, R. Zhang, and J.-Y. Zhu, "Contrastive learning for unpaired image-to-image translation," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2020, pp. 319–345.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{einlou2026styleenhanced,
  author  = {Einlou, Mokarrameh and Moghtaderi, Shiva and Wahid, Khan A. and Lukong, Kiven E.},
  title   = {Style-Enhanced {CycleGAN} for Unpaired Image-to-Image Translation: A New Approach to Medical Image Fusion Refinement},
  journal = {IEEE Access},
  volume  = {14},
  pages   = {56986--57001},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3683591}
}
```

`CITATION.cff` contains the same information for GitHub's "Cite this repository" button.

## License

The code is released under the GNU General Public License v3.0 (`LICENSE`). It is derived from [PyTorch-CycleGAN](https://github.com/ai-tor/PyTorch-CycleGAN) (GPL-3.0) and contains code adapted from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) (BSD); see `NOTICE`. FID and KID are computed with [torch-fidelity](https://github.com/toshas/torch-fidelity). The datasets are not covered by this license; their own terms apply. The paper is published under CC BY 4.0.

## Acknowledgment

This work was supported by the New Frontiers in Research Fund Exploration (NFRFE).

From the paper: "The authors are grateful to Dr. Mina Niazi (an Assistant Professor of Gastroenterology and Hepatology at the University of Saskatchewan), Dr. Nikrad Shahnavaz (an Associate Professor of Medicine and a Distinguished Physician at Emory University), and Dr. Amzad Hossain for providing subjective evaluation of the proposed method."
