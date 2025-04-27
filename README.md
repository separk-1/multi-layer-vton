# IDM-VTON (Modified and Extended for Multi-Layered Virtual Try-On)

<div align="center">
<h1>IDM-VTON: Improving Diffusion Models for Authentic Virtual Try-on in the Wild</h1>
<a href='https://idm-vton.github.io'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/abs/2403.05139'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='https://huggingface.co/spaces/yisol/IDM-VTON'><img src='https://img.shields.io/badge/Hugging%20Face-Demo-yellow'></a>
<a href='https://huggingface.co/yisol/IDM-VTON'><img src='https://img.shields.io/badge/Hugging%20Face-Model-blue'></a>
</div>

> **Note**: This project was conducted as part of the **10-623 Generative AI** course at **Carnegie Mellon University**.  
> ([Course website](https://www.cs.cmu.edu/~mgormley/courses/10423/))

---

## Project Overview

This project builds on ["Improving Diffusion Models for Authentic Virtual Try-on in the Wild"](https://arxiv.org/abs/2403.05139) (IDM-VTON).  
We **reproduce** the original IDM-VTON pipeline and **extend** it to accommodate **multi-layered clothing** scenarios, enabling **garment-on-garment** virtual try-on.

Our main contributions:
- Reproduced the baseline results of IDM-VTON.
- Modified the masking and inference pipeline to handle layered garments.
- Adapted the code for customized datasets.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yisol/IDM-VTON.git
cd IDM-VTON
```

### 2. Environment Setup (Windows)

```bash
conda env create -f environment_windows.yaml
conda activate idm

pip install huggingface_hub==0.20.3 --force-reinstall

pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install mediapipe
```

### 3. Preprocess Data (Optional)

For customized datasets:

```bash
python resizing.py          # Resize images
python generate_mask.py     # Generate agnostic masks
python generate_pose.py     # Generate densepose information
```

### 4. Download Pretrained Model

We use the official pretrained model hosted on Hugging Face:

```bash
--pretrained_model_name_or_path "yisol/IDM-VTON"
```

Alternatively, manually download it from [here](https://huggingface.co/yisol/IDM-VTON).

### 5. Download Dataset (Optional - VITON-HD)

If you wish to use the official VITON-HD dataset:

```bash
mkdir ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

pip install kaggle

kaggle datasets download -d marquis03/high-resolution-viton-zalando-dataset
unzip high-resolution-viton-zalando-dataset.zip -d viton_hd_dataset
```

---

## Folder Structure

Organize your dataset as follows:

```
datasets/
└── my_vest_data/
    ├── test/
        ├── image/
        ├── image-densepose/
        ├── agnostic-mask/
        ├── cloth/
        └── vitonhd_test_tagged.json
```

- `image/`: Person images
- `image-densepose/`: DensePose outputs
- `agnostic-mask/`: Customized body masks
- `cloth/`: Target clothing images
- `vitonhd_test_tagged.json`: Metadata for matching images and clothes

---

## Inference

After setup, run:

```bash
accelerate launch inference.py \
    --pretrained_model_name_or_path "yisol/IDM-VTON" \
    --width 512 \
    --height 768 \
    --num_inference_steps 20 \
    --output_dir "result" \
    --unpaired \
    --data_dir "./datasets/my_vest_data" \
    --test_batch_size 1 \
    --guidance_scale 2.0
```

Or simply:

```bash
sh inference_custom.sh
```

The results will be saved in the `result/` directory.

---

## Our Extensions

We build on top of the original IDM-VTON implementation with the following extensions for our course project:

- **Baseline Reproduction**: Reproduced the baseline results of IDM-VTON faithfully.
- **Multi-Layered Clothing Extension**: Extended the virtual try-on pipeline to accommodate garment-on-garment scenarios.
- **Customized Agnostic Mask Generation**: Modified the masking pipeline to better handle multi-layered input.
- **Resolution Adjustment**: Adjusted inference resolution to 512 × 768 for our dataset and experiments.

---

## Citation

If you use this work, please cite the original IDM-VTON paper:

```bibtex
@article{choi2024improving,
  title={Improving Diffusion Models for Authentic Virtual Try-on in the Wild},
  author={Choi, Yisol and Kwak, Sangkyung and Lee, Kyungmin and Choi, Hyungwon and Shin, Jinwoo},
  journal={arXiv preprint arXiv:2403.05139},
  year={2024}
}
```

---

## License

This project is licensed under the [CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

---

