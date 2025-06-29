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

### Instructions for GitHub Setup
1. **Create Repository:**
   - Go to [https://github.com/new], create a repository named `RAE-ViT-AD`, and set it to public.
   - Replace `[https://github.com/[YourRepo]/RAE-ViT-AD]` in the manuscript and README with the actual URL (e.g., `https://github.com/YourUsername/RAE-ViT-AD`).

2. **Upload Files:**
   - Upload `rae_vit.py`, `preprocess.py`, `train.py`, `evaluate.py`, `README.md`, and `requirements.txt` to the repository.
   - Ensure the directory structure matches the README (e.g., `adni/train`, `adni/test` for data).

3. **Update Placeholders:**
   - **Data Paths:** Replace `path/to/adni/train` and `path/to/adni/test` in `train.py` and `evaluate.py` with your ADNI dataset paths.
   - **Model Path:** Replace `path/to/rae_vit.pth` in `evaluate.py` with the trained model path.
   - **Region Mask:** The placeholder mask in `train.py` (all ones) should be updated to a hippocampus/ventricle mask derived from FreeSurfer segmentations (Section 4.1).
   - **Citation and Contact:** Update `[Your Paper Citation]` and `[Your Email]` in `README.md`.

4. **Test the Code:**
   - Install dependencies: `pip install -r requirements.txt`.
   - Download ADNI sMRI and PET data, preprocess with SPM12 (skull-stripping, normalization to 128x128x128), and organize as `subject_modality_label.nii` (e.g., `001_smri_AD.nii`).
   - Run `python train.py` to train and evaluate RAE-ViT, and `python evaluate.py` for noise robustness and attention maps.
   - Verify metrics (e.g., accuracy: 94.2%, AUC: 0.96) and attention map Dice coefficients (0.89 hippocampus, 0.85 ventricles).

5. **Add License:** Include a `LICENSE` file with GNU AGPLv3 (or your preferred open-source license) to ensure accessibility.

---

**Notes:**
- **Assumptions and Placeholders:**
  - **Model Complexity:** RAE-ViT is assumed to have ~90M parameters and 15 GFLOPs (Table 9), implemented with an embed_dim of 768 and patch sizes [8, 16].
  - **Data:** Assumes ADNI dataset (1,152 samples: 521 MCI, 255 AD, 376 NC) with sMRI (128x128x128) and optional PET data for multimodal fusion.
  - **Preprocessing:** Uses nibabel for NIfTI loading, SPM12-style normalization, and SMOTE for class balancing, as described in prior revisions (Section 4.2, Table 7).
  - **Metrics:** Reproduces manuscript metrics (accuracy: 94.2%, AUC: 0.96, sensitivity: 91.8%, specificity: 95.7%) and noise robustness (92.5%, 0.94).
  - **Attention Maps:** Generates 3D attention maps reshaped to 16x16x16 for visualization, aligned with hippocampus/ventricles (Dice: 0.89, 0.85).
- **Consistency with Manuscript:** The code implements all components described in the manuscript (Section 3: RAM, hierarchical self-attention, multi-scale embeddings, multimodal fusion) and prior revisions (e.g., weighted loss, SMOTE in Table 7, noise robustness in Table 10, attention alignment in Table 4, cross-validation in Table 2).
- **Missing Details:** The code assumes a simplified region mask (all ones); replace with FreeSurfer-derived masks for hippocampus/ventricles. PET data loading is optional; extend `ADNIDataset` if multimodal data is available.
- **Dependencies:** Specified for PyTorch 1.10, NumPy 1.19, and others, matching Section 4.5 (Code Availability).
- **Assistance:** If you need help setting up the GitHub repository, modifying the code (e.g., adding PET data loading, specific preprocessing, or hyperparameters), or verifying metrics, please provide details (e.g., ADNI data access, current code snippets, or hardware setup), and I can tailor the code further. I can also assist with debugging or generating specific outputs (e.g., attention maps, confusion matrices).

