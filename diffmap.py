import os
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === File path ===
adata_path = "/Users/nhungnguyen/Desktop/Research URoch Dr. Thakar/data/scBONITA/HIV_TN_AS_genes.h5ad"
if not os.path.exists(adata_path):
    print(f"File not found: {adata_path}")
    exit(1)

# === Load data ===
adata = ad.read_h5ad(adata_path)
print("AnnData loaded:", adata.shape)

# === Normalize and log transform ===
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata.copy()  # Save raw for later access

# === PCA + neighbors ===
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=min(adata.shape[1], 30))

# === Leiden clustering (for coloring)
sc.tl.leiden(adata, resolution=1.0)

# === Compute diffusion map ===
sc.tl.diffmap(adata)
print("Diffusion map computed. Shape:", adata.obsm['X_diffmap'].shape)

# === Raw diffusion plot (diagnostic)
dc = adata.obsm['X_diffmap']
plt.figure(figsize=(6, 5))
plt.scatter(dc[:, 0], dc[:, 1], s=30, alpha=0.7)
plt.xlabel("DC1")
plt.ylabel("DC2")
plt.title("Raw Diffusion Map (DC1 vs DC2)")
plt.tight_layout()
plt.show()

# === Plot colored by Leiden clusters
sc.pl.embedding(adata, basis='diffmap', color='leiden', title='Diffusion Map: Leiden Clusters')

# === Diffusion Pseudotime (DPT)
adata.uns['iroot'] = np.argmin(dc[:, 0])  # set root as cell with lowest DC1
sc.tl.dpt(adata)

# === Plot pseudotime
sc.pl.embedding(adata, basis='diffmap', color='dpt_pseudotime', title='Diffusion Pseudotime (DC1 root)')

# === Export diffusion coordinates
dc_df = pd.DataFrame(dc, index=adata.obs_names, columns=[f'DC{i+1}' for i in range(dc.shape[1])])
dc_df.to_csv("diffusion_components.csv")
print("Saved diffusion components to diffusion_components.csv")

# === Optional: Save AnnData with results
# adata.write("/Users/nhungnguyen/Desktop/Research URoch Dr. Thakar/output_diffmap_annotated.h5ad")
