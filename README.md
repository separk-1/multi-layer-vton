# Improving Virtual Try-On to Accommodate Multi-Layered  Clothing: Garment-on-Garment Virtual Try-on 

> **Note**: This project was conducted as part of the **10-623 Generative AI** course at **Carnegie Mellon University**.  
> ([Course website](https://www.cs.cmu.edu/~mgormley/courses/10423/))

---
## Original Code

**IDM-VTON: Improving Diffusion Models for Authentic Virtual Try-on in the Wild**

- [Project Page](https://idm-vton.github.io)  
- [Paper (Arxiv)](https://arxiv.org/abs/2403.05139)  
- [Hugging Face Demo](https://huggingface.co/spaces/yisol/IDM-VTON)  
- [Hugging Face Model](https://huggingface.co/yisol/IDM-VTON)

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
git clone https://github.com/separk-1/multi-layer-vton.git
cd multi-layer-vton
```

### 2. Environment Setup (Windows)

```bash
conda env create -f environment_windows.yaml
conda activate idm

pip install huggingface_hub==0.20.3

pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install mediapipe
```

### 3. Preprocess Data (Required for Custom Datasets)

You must generate DensePose maps and upper-cloth masks before running inference.

You can simply run:

```bash
sh preprocess.sh
```

The `preprocess.sh` file internally runs the following two steps:

- Step 1: Generate DensePose (pose map):

  ```bash
  python extract_densepose.py \
    --config detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml \
    --model detectron2/projects/DensePose/ckpt/model_final_162be9.pkl \
    --input datasets/test_joon/test/image/joon.jpg \
    --output_dir datasets/test_joon/test/image-densepose \
    --width 576 \
    --height 768
  ```

- Step 2: Generate upper-cloth masks:

  ```bash
  python extract_upper_cloth.py \
    --input datasets/test_joon/test/image/joon.jpg \
    --output_jpg datasets/test_joon/test/agnostic-mask/joon.jpg \
    --output_png datasets/test_joon/test/agnostic-mask/joon_mask.png \
    --densepose datasets/test_joon/test/image-densepose/I.npy \
    --naturality 1 \
    --ratio 0.1 \
    --vest 1 \
    --vest_padding 10
  ```

> ⚠️ **Important Notes:**
> - You must generate DensePose maps (`extract_densepose.py`) **before** generating upper-cloth masks (`extract_upper_cloth.py`).
> - Input person images (located at `datasets/my_vest_data/test/image/`) must be resized to **576×768**.
> - Input clothing images (located at `datasets/my_vest_data/test/cloth/`) can have any size.

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
└── <DATASET_NAME>/    # or test_joon/
    └── test/
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
sh inference.sh
```

The results will be saved in the `result/` directory.

---

## Evaluation

After generating results, you can evaluate the performance using our evaluation script.

First, organize the evaluation directory like this:

```
evaluation/
├── generated/
│   └── <IMAGE_NAME>.jpg
├── ground_truth/
│   └── <IMAGE_NAME>.jpg
└── eval.py
```

Then, run:

```bash
cd evaluation
python eval.py --ground_truth ./ground_truth --generated ./generated
```

This script will compute:
- LPIPS (Perceptual Similarity)
- SSIM (Structural Similarity)
- CLIP-Image Similarity (Semantic Alignment)
- FID (Image Realism)

> ⚠️ **Note:** Make sure all images in `generated/` and `ground_truth/` are resized to the same resolution before running evaluation.

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
