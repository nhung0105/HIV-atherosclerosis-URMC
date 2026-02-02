
####### GAMR gene trend plot

import os
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import cellrank as cr
import cellrank.kernels as kernels
import magic
import palantir
import palantir.utils as pal_utils
from cellrank.models import GAMR

# Disable multiprocessing to avoid the EOFError
import multiprocessing as mp
mp.set_start_method('fork', force=True)  # Use fork instead of spawn on macOS

# Set environment variable to avoid numba issues
os.environ['NUMBA_DISABLE_JIT'] = '1'

adata_path = "/Users/nhungnguyen/Desktop/Research URoch Dr. Thakar/data/scBONITA/combined_genes_subset.h5ad"
embeddings_filename = "/Users/nhungnguyen/Desktop/Research URoch Dr. Thakar/data/scBONITA/cell_tsne_embeddings.txt"
transition_matrix_file = "/Users/nhungnguyen/Desktop/Research URoch Dr. Thakar/data/scBONITA/cell_transition_matrix.csv"

# Load AnnData
adata = ad.read_h5ad(adata_path)
print(f"AnnData: {adata.shape[0]} cells, {adata.shape[1]} genes")

# Load tSNE embeddings
with open(embeddings_filename, 'r') as f:
    header = f.readline().strip().split('\t')
tsne_df = pd.read_csv(embeddings_filename, sep='\t', header=None, skiprows=0, index_col=0)
tsne_df = tsne_df.reindex(adata.obs_names)
adata.obsm['X_tsne'] = tsne_df.values
tsne_df = pd.DataFrame(adata.obsm['X_tsne'], index=adata.obs_names)

# Plot tSNE
sc.pl.embedding(adata, basis='X_tsne', title='tSNE Plot (custom embedding)')

# kNN graph in tSNE space
k = 30
tsne_embedding = adata.obsm['X_tsne']
nbrs = NearestNeighbors(n_neighbors=k+1).fit(tsne_embedding)
distances, indices = nbrs.kneighbors(tsne_embedding)
n_cells = tsne_embedding.shape[0]
sigma = np.mean(distances[:, 1:])
similarities = np.exp(-distances[:, 1:]**2 / (2. * sigma**2))
connectivities = csr_matrix((similarities.flatten(), 
                            (np.repeat(np.arange(n_cells), k), indices[:,1:].flatten())), 
                           shape=(n_cells, n_cells)).maximum(csr_matrix((similarities.flatten(), 
                            (np.repeat(np.arange(n_cells), k), indices[:,1:].flatten())), 
                           shape=(n_cells, n_cells)).T)
adata.obsp['connectivities'] = connectivities

# Connectivity Kernel
ck = kernels.ConnectivityKernel(adata)
ck.compute_transition_matrix()
print("Connectivity kernel ready.")

# Precomputed Kernel from transition matrix
transition_df = pd.read_csv(transition_matrix_file, header=None)
transition_df.columns = adata.obs_names
transition_df.index = adata.obs_names
transition_array = transition_df.values.astype(float)
row_sums = transition_array.sum(axis=1)
zero_rows = np.where(row_sums == 0)[0]
if len(zero_rows) > 0:
    transition_array[zero_rows] = 1.0 / transition_array.shape[1]
row_sums[row_sums == 0] = 1
transition_array = transition_array / row_sums[:, np.newaxis]
transition_matrix = csr_matrix(transition_array)
adata.obsp["cellrank_transition_matrix"] = transition_matrix
pk = kernels.PrecomputedKernel(adata, obsp_key="cellrank_transition_matrix")
print("Precomputed kernel ready.")

# Ensure tSNE basis for plotting
if 'tsne' not in adata.obsm and 'X_tsne' in adata.obsm:
    adata.obsm['tsne'] = adata.obsm['X_tsne']

# CellRank Estimator using both kernels
if __name__ == "__main__":
    # Use ConnectivityKernel
    g1 = cr.estimators.GPCCA(ck)
    g1.compute_schur(n_components=10)
    g1.compute_macrostates(n_states=4)
    g1.fit(n_states=3)
    g1.predict_initial_states()
    g1.predict_terminal_states(method="top_n", n_states=2)
    g1.plot_macrostates(which='initial', discrete=True, s=100)
    g1.compute_fate_probabilities(n_jobs=1)
    g1.plot_fate_probabilities()

    # Combine both kernels
    w1 = 1
    w2 = 1
    combined_kernel = w1 * ck + w2 * pk
    g_combo = cr.estimators.GPCCA(combined_kernel)
    g_combo.compute_macrostates(n_states=3)
    g_combo.compute_schur()
    g_combo.predict_initial_states()
    g_combo.predict_terminal_states(method="top_n", n_states=2)
    g_combo.plot_macrostates(which='initial', discrete=True, s=100)
    g_combo.compute_fate_probabilities(n_jobs=1)
    g_combo.plot_fate_probabilities()

    # Get root cell
    root_cell = adata.obs_names[np.argmax(g_combo.initial_states_probabilities)]

    # Filter genes to ones that exist
    genes = ["GSK3B", "AGER", "ARHGEF1", "ROCK2", "RHOA", "BCL2", "BAX"]
    genes = [g for g in genes if g in adata.var_names]
    print("Available genes:", genes)

    # Impute data using MAGIC (ensure dense matrix)
    magic_op = magic.MAGIC()
    adata.X = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
    adata.X = magic_op.fit_transform(adata.X)
    adata.layers["imputed"] = adata.X.copy()

    # Pseudotime (Palantir)
    tsne_df = pd.DataFrame(adata.obsm['X_tsne'], index=adata.obs_names)
    pr_res = palantir.core.run_palantir(tsne_df, root_cell, knn=30)
    adata.obs["pseudotime"] = (
        pr_res.pseudotime - pr_res.pseudotime.min()
    ) / (pr_res.pseudotime.max() - pr_res.pseudotime.min())

    # Store CellRank results
    adata.obsm["to_terminal_states"] = g_combo.fate_probabilities
    adata.obs["initial_states"] = g_combo.initial_states
    adata.obs["terminal_states"] = g_combo.terminal_states
    adata.obs["macrostates"] = g_combo.macrostates

    
    # Replace the gene trends plotting section with this corrected version:

# Get lineage names from CellRank Lineage object
lineage_names = list(g_combo.fate_probabilities.names)
print(f"Available lineages: {lineage_names}")
print(f"Fate probabilities type: {type(g_combo.fate_probabilities)}")
print(f"Fate probabilities shape: {g_combo.fate_probabilities.shape}")

print("Creating GAMR model instances for CellRank plotting...")



#  Try with single model instance (simpler approach)
try:
    print("Using single model instance approach...")
    
    # Create a single model instance
    model_instance = GAMR(adata, n_knots=6, smoothing_penalty=10.0)
    
    cr.pl.gene_trends(
        adata,
        model=model_instance,
        genes=genes,
        lineages=lineage_names,
        data_key="imputed",
        time_key="pseudotime",
        same_plot=True,
        hide_cells=True,
        conf_int=True,
        save="cellrank_single_model.pdf"
    )
    print("✓ Method 1 successful: Single model instance worked!")
    
except Exception as e:
    print(f"✗ Method 1 failed: {e}")

# METHOD 2: Manual plotting with CellRank's computed gene trends
try:
    print("Computing gene trends and plotting manually...")
    
    # Compute gene trends for each gene-lineage pair
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(len(genes), len(lineage_names), 
                            figsize=(6*len(lineage_names), 4*len(genes)))
    
    if len(genes) == 1:
        axes = [axes]
    if len(lineage_names) == 1:
        axes = [[ax] for ax in axes]
    
    for i, gene in enumerate(genes):
        for j, lineage in enumerate(lineage_names):
            ax = axes[i][j] if len(genes) > 1 else axes[j]
            
            try:
                # Create and fit model for this gene-lineage combination
                model = GAMR(adata, n_knots=6, smoothing_penalty=10.0)
                model = model.prepare(gene, lineage, time_key="pseudotime", data_key="imputed")
                model = model.fit()
                
                # Get predictions
                pseudotime_range = np.linspace(adata.obs["pseudotime"].min(), 
                                             adata.obs["pseudotime"].max(), 100)
                predictions = model.predict(pseudotime_range)
                
                # Plot
                ax.plot(pseudotime_range, predictions, 'r-', linewidth=2, label='Trend')
                
                # Add scatter of actual data points
                gene_expr = adata[:, gene].layers["imputed"].flatten()
                lineage_probs = g_combo.fate_probabilities[:, lineage].X.flatten()
                pseudotime = adata.obs["pseudotime"].values
                
                # Weight by lineage probability and add some transparency
                ax.scatter(pseudotime, gene_expr, 
                          alpha=lineage_probs * 0.7, 
                          s=10, c='blue')
                
                ax.set_xlabel('Pseudotime')
                ax.set_ylabel(f'{gene} Expression')
                ax.set_title(f'{gene} - Lineage {lineage}')
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Failed to fit\n{gene}-{lineage}', 
                       transform=ax.transAxes, ha='center', va='center')
                print(f"Failed to fit {gene}-{lineage}: {e}")
    
    plt.tight_layout()
    plt.savefig("cellrank_manual_trends.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Method 2 successful: Manual plotting completed!")
    
except Exception as e:
    print(f"✗ Method 2 failed: {e}")


print("\nCellrank Completed!")
print(f"\nGenes analyzed: {genes}")
print(f"Lineages: {lineage_names}")
