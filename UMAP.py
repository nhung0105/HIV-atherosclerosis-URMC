import os
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np

# === Load Data ===
adata_path = "/Users/nhungnguyen/Desktop/Research URoch Dr. Thakar/data/scBONITA/combined_genes.h5ad"
output_dir = "./figures"
os.makedirs(output_dir, exist_ok=True)
adata = ad.read_h5ad(adata_path)

# === Preprocessing ===
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata.copy()
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
sc.pp.scale(adata)

# === PCA ===
sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca(adata, save="_pca.png")

# === Neighbors + Clustering ===
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)
sc.tl.leiden(adata, resolution=1.0, key_added='leiden')

# === UMAP ===
sc.tl.umap(adata)
sc.pl.umap(adata, color='leiden', title='UMAP with Leiden Clusters', save="_umap_leiden.png")

# === Gene expression on UMAP ===
gene = "NFKB1"
if gene in adata.var_names:
    sc.pl.umap(adata, color=gene, color_map='viridis', title=f'{gene} Expression on UMAP', save=f"_umap_{gene}.png")
else:
    print(f"{gene} not found in adata.var_names")

# === Optional: PAGA on UMAP ===
sc.tl.paga(adata, groups='leiden')
sc.pl.paga(adata, color='leiden', threshold=0.03, title='PAGA Graph (UMAP)', save="_paga_umap.png")
sc.pl.paga_compare(adata, basis='umap', color='leiden', title='PAGA Paths on UMAP', save="_paga_paths_umap.png")

# === Pseudotime: Diffusion Map + DPT ===
root_cluster = '9'  # adjust this if needed
iroot = np.where(adata.obs['leiden'] == root_cluster)[0]
if len(iroot) == 0:
    raise ValueError("Root cluster not found.")
adata.uns['iroot'] = iroot[0]

sc.tl.diffmap(adata)
sc.tl.dpt(adata)

# === Pseudotime bins and plots on UMAP ===
adata.obs['pseudotime_bin'] = pd.cut(
    adata.obs['dpt_pseudotime'],
    bins=[0, 0.05, 0.1, 1.0],
    labels=['early', 'mid', 'late']
)

sc.pl.umap(adata, color='dpt_pseudotime', color_map='RdBu_r', title='Pseudotime on UMAP', save="_dpt_umap.png")
sc.pl.umap(adata, color='pseudotime_bin', palette=['blue', 'orange', 'red'],
           title='Pseudotime Bins on UMAP', save="_dpt_bins_umap.png")
