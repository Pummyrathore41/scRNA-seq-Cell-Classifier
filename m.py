"""
scRNA-seq Cell Classifier — Full Pipeline
==========================================
Stages:
    1. Load Data
    2. Preprocess
    3. Dimensionality Reduction + Clustering
    4. Annotation
    5. ML Classification

Run:
    python pipeline.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ── Global settings ───────────────────────────────────────────────────────────
RESULTS_DIR = "results"
DATA_DIR    = "data"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

sc.settings.verbosity = 1
sc.settings.figdir    = RESULTS_DIR

RANDOM_STATE = 42

def banner(stage, title):
    print("\n" + "=" * 60)
    print(f"  STAGE {stage} — {title}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
def stage1_load() -> sc.AnnData:
    banner(1, "Load Data")

    adata = sc.datasets.pbmc3k()

    print(f"  Cells   : {adata.n_obs:,}")
    print(f"  Genes   : {adata.n_vars:,}")
    print(f"  Sparsity: {100*(1 - adata.X.nnz/(adata.n_obs*adata.n_vars)):.1f}% zeros")

    adata.write_h5ad(os.path.join(DATA_DIR, "pbmc3k_raw.h5ad"))
    print(f"  Saved raw → {DATA_DIR}/pbmc3k_raw.h5ad")

    return adata


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — PREPROCESS
# ══════════════════════════════════════════════════════════════════════════════
def stage2_preprocess(adata: sc.AnnData) -> sc.AnnData:
    banner(2, "Preprocess")

    # -- Gene names as index
    adata.var_names_make_unique()

    # -- Basic quality metrics
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

    # -- Filter low-quality cells
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"  After cell/gene filter : {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # -- Mitochondrial gene filter (dead/broken cells have high MT expression)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs["pct_counts_mt"] < 5].copy()
    print(f"  After MT filter        : {adata.n_obs:,} cells")

    # -- QC violin plot
    sc.pl.violin(
        adata,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        jitter=0.4, multi_panel=True,
        save="_qc.png", show=False
    )
    print(f"  QC plot saved → {RESULTS_DIR}/violin_qc.png")

    # -- Normalize: each cell sums to 10,000 counts (TPM-like)
    sc.pp.normalize_total(adata, target_sum=1e4)

    # -- Log-transform: log(x+1) to compress dynamic range
    sc.pp.log1p(adata)

    # -- Store normalized+log counts as the main layer
    adata.raw = adata

    # -- Highly variable genes: focus on the most informative genes
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    print(f"  Highly variable genes  : {adata.var.highly_variable.sum():,}")
    adata = adata[:, adata.var.highly_variable].copy()

    # -- Scale to unit variance (required for PCA)
    sc.pp.scale(adata, max_value=10)

    print("  Preprocessing complete.")
    return adata


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — DIMENSIONALITY REDUCTION + CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
def stage3_reduce_cluster(adata: sc.AnnData) -> sc.AnnData:
    banner(3, "Dimensionality Reduction + Clustering")

    # -- PCA: 33k genes → 50 principal components
    sc.tl.pca(adata, svd_solver="arpack", random_state=RANDOM_STATE)
    sc.pl.pca_variance_ratio(adata, n_pcs=50, save="_variance.png", show=False)
    print("  PCA done (50 components)")

    # -- Neighborhood graph: basis for clustering and UMAP
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40, random_state=RANDOM_STATE)

    # -- UMAP: 50 PCA dims → 2D visualization
    sc.tl.umap(adata, random_state=RANDOM_STATE)
    print("  UMAP done")

    # -- Leiden clustering: graph-based community detection
    sc.tl.leiden(adata, resolution=0.5, random_state=RANDOM_STATE)
    n_clusters = adata.obs["leiden"].nunique()
    print(f"  Leiden clustering done : {n_clusters} clusters found")

    # -- UMAP plot colored by cluster
    sc.pl.umap(adata, color=["leiden"], legend_loc="on data",
               title="Leiden Clusters", save="_clusters.png", show=False)
    print(f"  UMAP plot saved → {RESULTS_DIR}/umap_clusters.png")

    return adata


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — ANNOTATION
# ══════════════════════════════════════════════════════════════════════════════
def stage4_annotate(adata: sc.AnnData) -> sc.AnnData:
    banner(4, "Cell Type Annotation")

    # Known marker genes (from published literature)
    marker_genes = {
        "CD4 T-cell"       : ["CD3D", "CD3E", "IL7R", "CD4"],
        "CD8 T-cell"       : ["CD3D", "CD3E", "CD8A", "CD8B"],
        "B-cell"           : ["CD19", "MS4A1", "CD79A"],
        "NK cell"          : ["GNLY", "NKG7", "GZMB"],
        "Monocyte"         : ["LYZ", "CD14", "CST3", "MS4A7"],
        "Dendritic cell"   : ["FCER1A", "CST3", "LILRA4"],
        "Platelet"         : ["PPBP", "PF4"],
    }

    # Score each cell for each cell type using marker genes
    for cell_type, genes in marker_genes.items():
        # Keep only genes present in the dataset
        valid = [g for g in genes if g in adata.raw.var_names]
        if valid:
            sc.tl.score_genes(adata, valid, score_name=cell_type, use_raw=True)

    score_cols = list(marker_genes.keys())

    # Assign cell type = highest scoring marker set
    adata.obs["cell_type"] = (
        adata.obs[score_cols]
        .idxmax(axis=1)
        .astype(str)
    )

    print("\n  Cell type distribution:")
    dist = adata.obs["cell_type"].value_counts()
    for ct, count in dist.items():
        print(f"    {ct:<20} {count:>5} cells")

    # UMAP colored by annotated cell type
    sc.pl.umap(adata, color=["cell_type"],
               title="Annotated Cell Types", save="_celltypes.png", show=False)
    print(f"\n  Annotated UMAP saved → {RESULTS_DIR}/umap_celltypes.png")

    # Dot plot: marker gene expression per cell type
    all_markers = list({g for genes in marker_genes.values() for g in genes})
    valid_markers = [g for g in all_markers if g in adata.raw.var_names]
    sc.pl.dotplot(adata, valid_markers, groupby="cell_type",
                  use_raw=True, save="_markers.png", show=False)
    print(f"  Marker dotplot saved  → {RESULTS_DIR}/dotplot_markers.png")

    return adata


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — ML CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
def stage5_classify(adata: sc.AnnData):
    banner(5, "ML Classification — Random Forest")

    # -- Features: PCA embedding (50 components) — compact and noise-reduced
    X = adata.obsm["X_pca"]
    y = adata.obs["cell_type"].values

    print(f"  Features : {X.shape[1]} PCA components")
    print(f"  Samples  : {X.shape[0]:,} cells")
    print(f"  Classes  : {np.unique(y).tolist()}")

    # -- Train / test split (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # -- Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced"   # handles imbalanced cell type frequencies
    )
    clf.fit(X_train, y_train)
    print("  Model trained.")

    # -- Evaluate
    y_pred = clf.predict(X_test)
    acc    = (y_pred == y_test).mean()
    print(f"\n  Test Accuracy : {acc*100:.2f}%")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # -- Confusion matrix heatmap
    labels = sorted(np.unique(y))
    cm     = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df  = pd.DataFrame(cm, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues",
                linewidths=0.5, ax=ax)
    ax.set_title("Confusion Matrix — Cell Type Classifier", fontsize=14, pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {cm_path}")

    # -- Feature importance (top 20 PCA components)
    importances = pd.Series(clf.feature_importances_,
                            index=[f"PC{i+1}" for i in range(X.shape[1])])
    top20 = importances.nlargest(20)

    fig, ax = plt.subplots(figsize=(8, 5))
    top20.sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Top 20 Most Important PCA Components", fontsize=13)
    ax.set_xlabel("Feature Importance")
    plt.tight_layout()
    fi_path = os.path.join(RESULTS_DIR, "feature_importance.png")
    fig.savefig(fi_path, dpi=150)
    plt.close()
    print(f"  Feature importance saved → {fi_path}")

    return clf


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    adata         = stage1_load()
    adata         = stage2_preprocess(adata)
    adata         = stage3_reduce_cluster(adata)
    adata         = stage4_annotate(adata)
    clf           = stage5_classify(adata)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  All results saved in → {RESULTS_DIR}/")
    print("=" * 60)