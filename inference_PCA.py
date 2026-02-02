import os
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np

# === Set up paths ===
adata_path = "/Users/nhungnguyen/Desktop/Research URoch Dr. Thakar/data/scBONITA/HIV_TN_AS_genes.h5ad"
output_dir = "./figures"
os.makedirs(output_dir, exist_ok=True)

# === Load AnnData ===
adata = ad.read_h5ad(adata_path)
print(f"Loaded AnnData with {adata.n_obs} cells and {adata.n_vars} genes.")

# === Preprocessing ===
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata.copy()
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
sc.pp.scale(adata)

# === PCA ===
sc.tl.pca(adata, svd_solver='arpack')

# === Build kNN graph from PCA ===
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)  # PCA is the default

# === Clustering and PAGA ===
sc.tl.leiden(adata, resolution=1.0, key_added='leiden')
sc.tl.paga(adata, groups='leiden')

# === tSNE for visualization ===
sc.tl.tsne(adata, use_rep='X_pca', n_pcs=20, perplexity=30)

# === Plot tSNE with Leiden ===
sc.pl.embedding(adata, basis='tsne', color='leiden', title='tSNE with Leiden Clusters',
                save="_your_tsne_leiden.png")

# === PAGA on tSNE layout ===
sc.pl.paga_compare(adata, basis='tsne', color='leiden', title='PAGA on tSNE',
                   save="_your_paga_tsne.png")

# === Gene expression (optional) ===
gene = "NFKB1"
if gene in adata.var_names:
    sc.pl.embedding(adata, basis='tsne', color=gene, color_map='viridis',
                    title=f'{gene} Expression on tSNE', save=f"_your_tsne_{gene}.png")

    sc.pl.paga_compare(adata, basis='tsne', color=gene, color_map='viridis',
                       title=f'PAGA on tSNE colored by {gene}', save=f"_your_paga_{gene}.png")
else:
    print(f"Gene {gene} not found in adata.var_names")

# === Diffusion Map + Pseudotime ===
root_cluster = '9'  # adjust if needed
iroot = np.where(adata.obs['leiden'] == root_cluster)[0]
if len(iroot) == 0:
    raise ValueError("Root cluster not found.")
adata.uns['iroot'] = iroot[0]

sc.tl.diffmap(adata)
sc.tl.dpt(adata)

# === Bin pseudotime ===
adata.obs['pseudotime_bin'] = pd.cut(
    adata.obs['dpt_pseudotime'],
    bins=[0, 0.05, 0.1, 1.0],
    labels=['early', 'mid', 'late']
)

# === Plot pseudotime ===
sc.pl.embedding(adata, basis='tsne', color='dpt_pseudotime', color_map='RdBu_r',
                title='DPT Pseudotime on tSNE', save="_your_dpt_tsne.png")

sc.pl.embedding(adata, basis='tsne', color='pseudotime_bin',
                palette=['blue', 'orange', 'red'], title='Pseudotime Bins on tSNE',
                save="_your_dpt_bins_tsne.png")


##PAGA

import os
import scanpy as sc
import anndata as ad

# === Load your AnnData ===
adata_path = "/Users/nhungnguyen/Desktop/Research URoch Dr. Thakar/data/scBONITA/HIV_TN_AS_genes.h5ad"
adata = ad.read_h5ad(adata_path)

# === Minimal preprocessing (if needed) ===
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata.copy()
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
sc.pp.scale(adata)
sc.tl.pca(adata)

# === Build neighbor graph using PCA ===
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)

# === Cluster the cells ===
sc.tl.leiden(adata, resolution=1.0, key_added='leiden')

# === Run PAGA ===
sc.tl.paga(adata, groups='leiden')

# === Plot PAGA trajectory ===
sc.pl.paga(adata, color='leiden', threshold=0.03, title='PAGA Graph: Expression-based', save="_paga_only.png")
