# scRNA-seq Cell Classifier

A complete single-cell RNA-seq pipeline: load PBMC3k data → preprocess & QC → dimensionality reduction & clustering → marker-based cell type annotation → Random Forest classification.

## 📋 Table of Contents
- [Overview](#overview)
- [Pipeline Stages](#pipeline-stages)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)

## 🔬 Overview

This project downloads the classic **PBMC3k** scRNA-seq dataset, performs quality control and normalization, reduces dimensionality via PCA/UMAP, clusters cells with the Leiden algorithm, annotates clusters into known immune cell types using marker genes, and finally trains a **Random Forest classifier** on PCA embeddings to predict cell type.

## 🧬 Pipeline Stages

1. **Load Data** — Downloads PBMC 3k dataset via `scanpy.datasets.pbmc3k()`
2. **Preprocess** — Cell/gene filtering, mitochondrial QC, normalization, log-transform, HVG selection, scaling
3. **Dimensionality Reduction + Clustering** — PCA (50 PCs), neighbor graph, UMAP, Leiden clustering
4. **Annotation** — Marker-gene scoring to label clusters (CD4 T-cell, CD8 T-cell, B-cell, NK cell, Monocyte, Dendritic cell, Platelet)
5. **ML Classification** — Random Forest trained on PCA features, with confusion matrix & feature importance plots

## ⚙️ Installation

### Prerequisites
- Python **3.10.11**
- Git

### Step 1: Clone the repository
```bash
git clone https://github.com/Pummyrathore41/scRNA-seq-Cell-Classifier.git
cd scRNA-seq-Cell-Classifier
```

### Step 2: Create a virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Deactivate (when done)
```bash
deactivate
```

## ▶️ Usage

Run the full pipeline (single command runs all 5 stages):

```bash
python pipeline.py
```

This will:
- Create `data/` and `results/` folders automatically
- Download the PBMC3k dataset (~ first run only, requires internet)
- Save plots and the trained model artifacts into `results/`

## 📊 Results

After running, check the `results/` folder for:

| File | Description |
|---|---|
| `violin_qc.png` | QC metrics (genes/cell, total counts, % mitochondrial) |
| `pca_variance.png` | PCA variance ratio (elbow plot) |
| `umap_clusters.png` | UMAP colored by Leiden clusters |
| `umap_celltypes.png` | UMAP colored by annotated cell types |
| `dotplot_markers.png` | Marker gene expression dot plot per cell type |
| `confusion_matrix.png` | Random Forest confusion matrix on test set |
| `feature_importance.png` | Top 20 important PCA components |

### Sample Console Output
