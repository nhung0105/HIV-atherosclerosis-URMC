import os
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np

# === Paths ===
adata_path = "/Users/nhungnguyen/Desktop/Research URoch Dr. Thakar/data/scBONITA/combined_genes.h5ad"
output_dir = "./figures"
os.makedirs(output_dir, exist_ok=True)

# === Load ===
adata = ad.read_h5ad(adata_path)
print(f"AnnData loaded: {adata.shape}")

# === Preprocessing ===
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata.copy()
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
sc.pp.scale(adata)

# === PCA ===
sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca(adata, color=None, title='PCA', save="_pca.png")

# === Neighbors & Clustering ===
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)
sc.tl.leiden(adata, resolution=1.0, key_added='leiden')

# === tSNE (on PCA-reduced space) ===
sc.tl.tsne(adata, use_rep='X_pca', n_pcs=20, perplexity=30)
sc.pl.embedding(adata, basis='tsne', color='leiden', title='tSNE with Leiden', save="_tsne_leiden.png")

# === Optional: PAGA ===
sc.tl.paga(adata, groups='leiden')
sc.pl.paga(adata, color='leiden', threshold=0.03, title='PAGA Graph', save="_paga.png")

# === Gene Expression Overlay ===
gene = "NFKB1"
if gene in adata.var_names:
    sc.pl.embedding(adata, basis='tsne', color=gene, color_map='viridis',
                    title=f'{gene} Expression on tSNE', save=f"_tsne_{gene}.png")
else:
    print(f"{gene} not found in var_names")

# === Pseudotime Analysis (Diffusion + DPT) ===
root_cluster = '9'  # adjust if needed
iroot = np.where(adata.obs['leiden'] == root_cluster)[0]
if len(iroot) == 0:
    raise ValueError("Root cluster not found.")
adata.uns['iroot'] = iroot[0]

sc.tl.diffmap(adata)
sc.tl.dpt(adata)

# === Pseudotime binning ===
adata.obs['pseudotime_bin'] = pd.cut(
    adata.obs['dpt_pseudotime'],
    bins=[0, 0.05, 0.1, 1.0],
    labels=['early', 'mid', 'late']
)

# === Pseudotime on tSNE ===
sc.pl.embedding(adata, basis='tsne', color='dpt_pseudotime', color_map='RdBu_r',
                title='Pseudotime on tSNE', save="_dpt_tsne.png")
sc.pl.embedding(adata, basis='tsne', color='pseudotime_bin',
                palette=['blue', 'orange', 'red'], title='Pseudotime Bins on tSNE', save="_dpt_bin_tsne.png")
