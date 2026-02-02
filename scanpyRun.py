import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
adata = sc.read("/Users/nhungnguyen/Desktop/Research URoch Dr. Thakar/data/scBONITA/HIV_TN_AS_genes.h5ad")

# Preprocessing
sc.pp.filter_genes(adata, min_counts=1)
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
adata.raw = adata.copy()
sc.pp.highly_variable_genes(adata, n_top_genes=1000)
adata = adata[:, adata.var.highly_variable]
sc.pp.scale(adata)


sc.tl.pca(adata, svd_solver='arpack') 
##sc.tl.tsne
##if neighbors cannot be called then use our own knn
##NFKB1

# Neighbors and clustering
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)
sc.tl.leiden(adata, resolution=1.0)

# PAGA 
## They were asking to plot PAGA first
sc.tl.paga(adata, groups='leiden')
sc.pl.paga(adata, threshold=0.03, show=True)

# Initialize layout with PAGA
sc.tl.draw_graph(adata, init_pos='paga')
sc.pl.draw_graph(adata, color='leiden', show=True)


# Diffusion pseudotime
adata.uns['iroot'] = np.flatnonzero(adata.obs['leiden'] == adata.obs['leiden'].cat.categories[0])[0]
sc.tl.dpt(adata)
sc.pl.draw_graph(adata, color='dpt_pseudotime', show=True)

# Use real gene names (replace these with top genes from your data)
genes_to_plot = adata.var_names[:10]  # Or provide your own gene list

# Define PAGA paths based on the clustering result
paga_paths = [
    ("Path1", ['0', '1', '2']),
    ("Path2", ['0', '3', '4']),
    ("Path3", ['0', '5', '6']),
]

# Plot gene expression along PAGA paths
fig, axs = plt.subplots(ncols=len(paga_paths), figsize=(4 * len(paga_paths), 3), gridspec_kw={'wspace': 0.4})
for i, (desc, path) in enumerate(paga_paths):
    data = sc.pl.paga_path(
        adata,
        path,
        genes_to_plot,
        use_raw=False,
        n_avg=50,
        annotations=["dpt_pseudotime"],
        groups_key="leiden",
        return_data=True,
        ax=axs[i],
        title=desc
    )
    data.to_csv(f"paga_path_{desc}.csv")
plt.tight_layout()
plt.show()

# Optional: Save processed AnnData
adata.write("HIV_TN_AS_genes_processed.h5ad")
