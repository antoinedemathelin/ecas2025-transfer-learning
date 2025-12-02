# ECAS 2025 - Transfer Learning Tutorial Materials

This repository contains the tutorial materials for the Transfer Learning workshop at **ECAS 2025**.

TDs are available on Colab!
- PINNS : [pinns-tutorial](https://github.com/nguyenkhoa0209/pinns_tutorial)
- Synthetic Datasets: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11rn9N_n-kAS1ZHamHIst2hd0hTRWHvxZ?usp=sharing)
- Real-World Datasets: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13VZbOO5zkYNTBNwUux_M079-cq6yz0HO?usp=sharing)

You can also try Binder:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/antoinedemathelin/ecas2025-transfer-learning/HEAD) (slower)

Or run the notebooks on your laptop (see [Installation](#Installation))

## 📚 Overview

This hands-on tutorial introduces fundamental concepts and practical applications of **Transfer Learning** and **Domain Adaptation** techniques. Through interactive Jupyter notebooks, participants will explore how to leverage pre-trained models and adapt them to new domains where training data may be limited or distributed differently.

## 📂 Repository Contents

- **Jupyter Notebooks**: Interactive exercises with guided questions and implementations
- **Datasets**: Sample datasets for hands-on practice

## 🛠️ Prerequisites

### Software Requirements
- Python 3.9+
- Jupyter Notebook or JupyterLab
- Required libraries (see `requirements.txt`)

## 🚀 Getting Started

### Installation

1. Clone this repository:
```bash
git clone https://github.com/antoinedemathelin/ecas2025-transfer-learning.git
cd ecas2025-transfer-learning
```
Or download the zip file and extract it.

2. Create a virtual environment (recommended):
```bash
python -m venv ecas-tl
source ecas-tl/bin/activate  # On Windows: ecas-tl\Scripts\activate
```
Or use conda:
```bash
conda create -n ecas-tl python=3.11
conda activate ecas-tl
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
