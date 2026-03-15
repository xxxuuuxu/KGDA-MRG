# KGDA-MRG: Knowledge-Guided Dynamic Alignment for Medical Report Generation

[![Code License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

Official PyTorch implementation of the paper: **"Knowledge-Guided Dynamic Alignment with Saliency-Aware Enhancement for Medical Report Generation"**.

## 🛠️ Environment Setup

We recommend using Anaconda to manage the environment. The codebase has been tested on a single RTX 4090 GPU with Ubuntu 20.04.

1. Clone the repository:
```bash
git clone https://github.com/xxxuuuxu/KGDA-MRG.git
cd KGDA-MRG
```

2. Create and activate a conda environment:
```bash
conda create -n kgda_mrg python=3.8
conda activate kgda_mrg
```

3. Install the required dependencies:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other requirements
pip install -r requirements.txt
```

## 🗂️ Data Preparation

To reproduce the results in our paper, please download the following datasets and organize them into the `data/` directory.

### 1. MIMIC-CXR Dataset
1. Download the [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) dataset from PhysioNet.
2. Place the images and annotation files in the `data/mimic_cxr/` directory.

### 2. IU X-Ray Dataset
1. Download the Indiana University Chest X-Ray dataset from [OpenI](https://openi.nlm.nih.gov/faq) or its Kaggle mirrors.
2. Place the extracted images and report files in the `data/iu_xray/` directory.

### 3. FFA-IR Dataset
1. Download the [FFA-IR](https://physionet.org/content/ffa-ir-medical-report/1.0.0/) (Fundus Fluorescein Angiography Images and Reports) dataset from PhysioNet.
2. *(Note: To extract all image files correctly, use the command `cat FAIR.tar.gz.* | tar -zxv` as officially recommended).*
3. Place the unzipped case directories (which contain the FFA images) and the annotation files in the `data/ffa_ir/` directory.

### Expected Directory Structure:
```text
data/
├── mimic_cxr/
│   ├── images/
│   └── annotation.json
├── iu_xray/
│   ├── images/
│   └── annotation.json
└── ffa_ir/
    ├── images/
    │   ├── patient_0001/
    │   ├── patient_0002/
    │   └── ...
    └── annotation.json
```

## 🏃‍♂️ Training & Evaluation

### Training
To train the KGDA-MRG model from scratch on a single GPU, simply run:

```bash
bash train_mimic.sh
bash train_iu.sh
bash train_ffair.sh
```

### Evaluation
To evaluate the model and compute NLG (BLEU, METEOR, ROUGE-L) and CE (CheXpert Precision, Recall, F1) metrics, use the following command:

```bash
bash test_mimic.sh
bash test_iu.sh
bash test_ffair.sh
```

