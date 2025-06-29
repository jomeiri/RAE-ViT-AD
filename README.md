# RAE-ViT: Regional Attention-Enhanced Vision Transformer for AD Classification

This repository contains the implementation of RAE-ViT for Alzheimer’s disease (AD) classification using sMRI and PET data from the ADNI dataset, as described in our paper. The model achieves 94.2% accuracy and 0.96 AUC on sMRI, with multimodal sMRI-PET fusion improving to 95.8% accuracy and 0.97 AUC.

## Requirements
- Python 3.6+
- PyTorch 1.10
- NumPy 1.19
- nibabel, scikit-learn, imblearn, matplotlib, seaborn
- Install via: `pip install -r requirements.txt`

## Dataset
- Download ADNI sMRI and PET data from [http://adni.loni.usc.edu].
- Preprocess using SPM12 for skull-stripping and normalization (128x128x128).
- Directory structure: `adni/train/subject_modality_label.nii`, `adni/test/...`.

## Files
- `rae_vit.py`: Model architecture (RAE-ViT with RAM, hierarchical self-attention, multi-scale embeddings, cross-modal attention).
- `preprocess.py`: ADNI dataset loading, normalization, SMOTE, and Gaussian noise addition.
- `train.py`: Training script with weighted cross-entropy loss and evaluation.
- `evaluate.py`: Evaluation script for metrics and attention map visualization.

## Usage
1. **Train RAE-ViT:**
   ```bash
   python train.py
Update path/to/adni/train and path/to/adni/test in train.py.
•	Evaluate Noise Robustness:
•	python evaluate.py
Generates confusion matrix and attention maps under 10% Gaussian noise.
Replication
•	Train on ADNI (521 MCI, 255 AD, 376 NC) with 5-fold cross-validation.
•	Evaluate metrics: accuracy (94.2%), AUC (0.96), sensitivity (91.8%), specificity (95.7%).
•	Generate attention maps aligned with hippocampus (Dice: 0.89) and ventricles (Dice: 0.85).
License
GNU AGPLv3
Citation
[my Paper Citation]
Contact
alirezajomeiri@iau.ac.ir


#### File: `requirements.txt`
torch==1.10.0 
numpy==1.19.0 
nibabel==3.2.1 
scikit-learn==1.0.2 
imbalanced-learn==0.8.0 
matplotlib==3.4.3 
seaborn==0.11.2


---

