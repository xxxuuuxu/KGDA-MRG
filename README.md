# KGDA-MRG: Knowledge-Guided Dynamic Alignment for Medical Report Generation

[![Code License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

Official PyTorch implementation of the paper: **"Knowledge-Guided Dynamic Alignment with Saliency-Aware Enhancement for Medical Report Generation"**.

## 🛠️ Environment Setup

We recommend using Anaconda to manage the environment. The codebase has been tested on a single RTX 4090 GPU with Ubuntu 20.04.

1. Clone the repository:
```bash
git clone [https://github.com/](https://github.com/)[需填写：你的GitHub用户名]/[需填写：你的仓库名].git
cd [需填写：你的仓库名]
```

2. Create and activate a conda environment:
```bash
conda create -n kgda_mrg python=3.8
conda activate kgda_mrg
```

3. Install the required dependencies:
```bash
# Install PyTorch (Please adjust the CUDA version according to your local machine)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other requirements
pip install -r requirements.txt
```
*(Note: Ensure you have installed the correct version of `causal-conv1d` and `mamba-ssm` if your KDDA module requires them.)*

## 🗂️ Data Preparation

### 1. MIMIC-CXR Dataset
We evaluate our model on the widely used MIMIC-CXR dataset.
1. Download the [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) dataset.
2. Place the images and annotation files in the `data/mimic_cxr/` directory.

The expected directory structure is:
```text
data/
└── mimic_cxr/
    ├── images/
    │   ├── p10/
    │   ├── p11/
    │   └── ...
    ├── annotation.json
    └── [需填写：如果有别的拆分文件，如 train.csv 等，请补充]
```

## 🏃‍♂️ Training & Evaluation

### Training
To train the KGDA-MRG model from scratch on a single GPU, simply run:

```bash
python train.py \
    --config configs/[需填写：你的配置文件名，如 mimic_cxr_config.yaml] \
    --batch_size [需填写：如 16] \
    --epochs [需填写：如 50]
```

### Evaluation
To evaluate the model and compute NLG (BLEU, METEOR, ROUGE-L) and CE (CheXpert Precision, Recall, F1) metrics, use the following command:

```bash
python test.py \
    --config configs/[需填写：你的配置文件名.yaml] \
    --resume checkpoints/[需填写：你提供的预训练模型名称.pth]
```
*(We provide our best pre-trained weights [here]([需填写：权重下载链接]), which achieved 0.519 F1 on MIMIC-CXR).*

## 📊 Main Results

Comparison of computational cost and performance on MIMIC-CXR:

| Method | Params (M) | FLOPs (G) | BLEU-4 | F1 (CE) |
|:---|:---:|:---:|:---:|:---:|
| R2Gen | 90.8 | 148.8 | 0.103 | 0.276 |
| PromptMRG | ~180+ | - | 0.110 | 0.476 |
| **KGDA-MRG (Ours)** | **~71** | **~37** | **0.137** | **0.519** |

*For full experimental results and ablation studies, please refer to our paper.*

## 📝 Citation

If you find this code or our paper useful for your research, please cite:

```bibtex
@article{[需填写：随便起个代号，如 yourname2024kgda],
  title={[需填写：你的论文标题]},
  author={[需填写：作者全名列表，如 Lastname, Firstname and Lastname, Firstname]},
  journal={arXiv preprint arXiv:[需填写：如果有]},
  year={[需填写：2024]}
}
```

## 📧 Contact
If you have any questions, please feel free to open an issue or contact `[需填写：你的邮箱]`.
