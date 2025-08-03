# CRL-MIL
# CRL-MIL: Causal Representation Learning for Multi-Instance Learning

This repository contains the official implementation of the **CRL-MIL** framework, which leverages causal representation learning to improve weakly supervised whole slide image (WSI) classification in the multiple instance learning (MIL) setting.

---

## 📂 Dataset Preparation
Prepare your WSI data and place the slide-level data splits (train/val/test) into the data/ directory. Example datasets include:
 - BRACS
 - CPTAC-Lung
Each dataset folder should include the corresponding .csv split files.

## 🏃‍♂️ Training & Evaluation
Train and evaluate the CRL-MIL model using:
    ``` python train/manage.py
    ```
## 🧠 Models
The Models/ directory includes:
- abmil.py: Attention-based MIL baseline
- TransMIL.py: Transformer-based MIL backbone
- ILRA.py: Transformer-based MIL backbone
- WiKG.py: GNN-based MIL backbone
- model_loader.py: Loads and configures model backbones
- Additional files for attention modules and other utilities


## ⚙️ Configuration
Training configurations are defined in:
- Config/base.py
- Config/train_args.py
- Config/args_base.py

You can create or edit these files to modify model architecture, training hyperparameters, etc.

## 📊 Results
We evaluate on multiple benchmark datasets (e.g., BRACS, CPTAC), reporting:
- Accuracy
- AUC
- F1 Score
- MCC

## 📄 License
This project is intended for academic research purposes only. Please contact the authors for commercial licensing.


📬 Contact
If you have any questions or issues, feel free to open an issue or reach out by email.



