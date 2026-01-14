# Video Polyp Segmentation using Implicit Networks

[![Paper](https://img.shields.io/badge/Paper-MIDL%202024-blue)](https://raw.githubusercontent.com/mlresearch/v250/main/assets/dahan24c/dahan24c.pdf)
[![Conference](https://img.shields.io/badge/Conference-MIDL%202024-green)](https://2024.midl.io/)

**Official implementation** of the paper:

> **Video Polyp Segmentation using Implicit Networks**  
> Aviad Dahan, Tal Shaharabany, Raja Giryes, Lior Wolf  
> *Medical Imaging with Deep Learning (MIDL), 2024*  
> [[Paper PDF]](https://raw.githubusercontent.com/mlresearch/v250/main/assets/dahan24c/dahan24c.pdf) [[Google Scholar]](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=6OxJ6hIAAAAJ&citation_for_view=6OxJ6hIAAAAJ:u-x6o8ySG0sC)

## Abstract

Polyp segmentation in endoscopic videos is an essential task in medical image and video analysis, requiring pixel-level accuracy to accurately identify and localize polyps within the video sequences. Addressing this task unveils the intricate interplay of dynamic changes in the video and the complexities involved in tracking polyps across frames. Our research presents an innovative approach to effectively meet these challenges that integrates, at test time, a pre-trained image (2D) model with a new form of implicit representation. By leveraging the temporal understanding provided by implicit networks and enhancing it with optical flow-based temporal losses, we significantly enhance the precision and consistency of polyp segmentation across sequential frames.

## Overview

The method uses an Implicit Multi-Layer Perceptron (IMLP) network to refine noisy segmentation masks by enforcing temporal consistency through optical flow. Given initial segmentation predictions (e.g., from SAM or other methods), the network learns a continuous representation of the mask that respects motion boundaries.

## Main Training Scripts

| Script | Description |
|--------|-------------|
| `train_h_video.py` | Main training script for DAVIS and SUN-SEG datasets |
| `train_dino_video.py` | DINO-based training with vision transformer features |
| `train_h_capsule.py` | Training script for capsule endoscopy data |

## Project Structure

```
├── train_h_video.py          # Main training script
├── train_dino_video.py       # DINO-based training
├── train_h_capsule.py        # Capsule endoscopy training
├── ref_model.py              # RefineModel definition
├── loss_utils.py             # Optical flow and rigidity losses
├── unwrap_utils.py           # Data loading and preprocessing
├── utils_dino.py             # DINO feature extraction utilities
├── utils.py                  # General utilities
├── vision_transformer.py     # Vision Transformer implementation
├── dino_eval_method.py       # Evaluation utilities
├── models/
│   ├── implicit_neural_networks.py  # IMLP network
│   ├── base.py
│   ├── resnet.py
│   ├── vgg16.py
│   └── hardnet.py
└── davis2017/                # DAVIS dataset utilities
    ├── davis.py
    ├── evaluation.py
    ├── metrics.py
    ├── results.py
    └── utils.py
```

## Requirements

```
torch
torchvision
numpy
opencv-python
Pillow
natsort
tqdm
imageio
```

## Usage

### Training on SUN-SEG Dataset

```bash
python train_h_video.py \
    --data_folder /path/to/SUN-SEG/TestHardDataset/Unseen \
    --dataset sun-hard \
    --it 10 \
    --lr 1e-3
```

### Training on Capsule Endoscopy Data

```bash
python train_h_capsule.py \
    --data_folder /path/to/capsule_data \
    --dataset capsule \
    --it 500
```

### Training on DAVIS Dataset

```bash
python train_h_video.py \
    --data_folder /path/to/DAVIS \
    --dataset davis2017 \
    --it 100 \
    --lr 1e-3 \
    --samples 50000
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_folder` | `./data` | Path to the dataset folder |
| `--dataset` | `sun-hard` | Dataset type: davis2017, davis2016, sun-easy, sun-hard, capsule |
| `--it` | 10 | Number of training iterations |
| `--lr` | 1e-3 | Learning rate |
| `--samples` | 50000 | Number of samples per iteration |
| `--resx` | 448 | Resolution X |
| `--resy` | 256 | Resolution Y |
| `--w_of` | 1 | Optical flow loss weight |
| `--number_of_channels_alpha` | 400 | Hidden dimension of IMLP |
| `--number_of_layers_alpha` | 8 | Number of layers in IMLP |
| `--positional_encoding_num_alpha` | 16 | Positional encoding dimension |

## Model Architecture

The core model (`RefineModel`) uses an Implicit Multi-Layer Perceptron (IMLP) that:
- Takes (x, y, t) coordinates as input with positional encoding
- Outputs per-pixel class probabilities
- Uses skip connections at layers 4 and 6
- Enforces temporal consistency via optical flow loss

## Dataset Structure

Expected folder structure for datasets:

```
dataset_folder/
├── Frame/           # Input video frames
│   └── video_name/
│       ├── 00000.jpg
│       └── ...
├── GT/              # Ground truth masks
│   └── video_name/
│       ├── 00000.png
│       └── ...
├── preds/           # Initial predictions (e.g., from SAM)
│   └── video_name/
│       ├── 00000.png
│       └── ...
└── flow/            # Precomputed optical flow
    └── video_name/
        ├── 00000.jpg_00001.jpg.npy
        └── ...
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{dahan2024video,
  title={Video Polyp Segmentation using Implicit Networks},
  author={Dahan, Aviad and Shaharabany, Tal and Giryes, Raja and Wolf, Lior},
  booktitle={Medical Imaging with Deep Learning},
  year={2024}
}
```

## License

This project is for research purposes.
