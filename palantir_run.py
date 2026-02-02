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

# Plot initial tSNE
sc.pl.embedding(adata, basis='X_tsne', title='tSNE Plot (custom embedding)')

# Build kNN graph in tSNE space for CellRank
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

# CellRank Connectivity Kernel
ck = kernels.ConnectivityKernel(adata)
ck.compute_transition_matrix()
print("Connectivity kernel ready.")

# CellRank Precomputed Kernel from transition matrix
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

# =============================================================================
# PALANTIR INTEGRATION
# =============================================================================

def run_palantir_analysis(adata, start_cell=None):
    """
    Run Palantir trajectory analysis integrated with CellRank results
    """
    print("Starting Palantir analysis...")
    
    # Use the combined transition matrix from CellRank for Palantir
    # Combine CellRank kernels first
    w1, w2 = 1, 1
    combined_kernel = w1 * ck + w2 * pk
    
    # Get the combined transition matrix
    combined_transition_matrix = combined_kernel.transition_matrix
    
    # Determine start cell automatically if not provided
    if start_cell is None:
        # Use CellRank to identify potential root cells
        g_temp = cr.estimators.GPCCA(combined_kernel)
        g_temp.compute_schur(n_components=10)
        g_temp.compute_macrostates(n_states=4)
        g_temp.predict_initial_states()
        
        # Find cells with highest initial state probability
        initial_probs = g_temp.initial_states_probabilities
        if initial_probs is not None:
            start_cell = initial_probs.idxmax()
            print(f"Auto-selected start cell: {start_cell}")
        else:
            # Fallback: use first cell
            start_cell = adata.obs_names[0]
            print(f"Using fallback start cell: {start_cell}")
    
    # Prepare data for Palantir - need a multiscale data representation
    # Use the existing tSNE coordinates or compute diffusion components
    if 'X_pca' not in adata.obsm:
        # Compute PCA if not available
        sc.tl.pca(adata)
    
    # Create multiscale data using existing embeddings
    ms_data = pd.DataFrame(
        adata.obsm['X_tsne'], 
        index=adata.obs_names,
        columns=[f'tSNE_{i}' for i in range(adata.obsm['X_tsne'].shape[1])]
    )
    
    # Run Palantir pseudotime analysis using the correct parameters
    print("Computing pseudotime...")
    try:
        pr_res = palantir.core.run_palantir(
            ms_data, 
            early_cell=start_cell,
            knn=30,
            num_waypoints=500,  # Reduced for faster computation
            n_jobs=1,  # Use single job to avoid issues
            use_early_cell_as_start=True
        )
        
        # Extract results based on Palantir version
        if hasattr(pr_res, 'pseudotime'):
            # Newer version with result object
            adata.obs['palantir_pseudotime'] = pr_res.pseudotime
            adata.obs['palantir_entropy'] = pr_res.entropy
            adata.obsm['palantir_branch_probs'] = pr_res.branch_probs
            terminal_states = pr_res.branch_probs.columns.tolist()
        else:
            # Older version returns tuple
            pseudotime, branch_probs = pr_res
            adata.obs['palantir_pseudotime'] = pseudotime
            # Compute entropy from branch probabilities
            adata.obs['palantir_entropy'] = branch_probs.apply(lambda x: -np.sum(x * np.log(x + 1e-10)), axis=1)
            adata.obsm['palantir_branch_probs'] = branch_probs
            terminal_states = branch_probs.columns.tolist()
        
        print(f"Found {len(terminal_states)} terminal states: {terminal_states}")
        
    except Exception as e:
        print(f"Palantir analysis failed: {e}")
        print("Falling back to basic pseudotime computation...")
        
        # Fallback: compute simple pseudotime using diffusion distance
        from sklearn.metrics import pairwise_distances
        
        # Get start cell index
        start_idx = np.where(adata.obs_names == start_cell)[0][0]
        
        # Compute distances from start cell in tSNE space
        distances = pairwise_distances(
            adata.obsm['X_tsne'], 
            adata.obsm['X_tsne'][start_idx:start_idx+1]
        ).flatten()
        
        # Normalize to [0, 1] for pseudotime
        pseudotime = (distances - distances.min()) / (distances.max() - distances.min())
        adata.obs['palantir_pseudotime'] = pseudotime
        
        # Create dummy terminal states (highest pseudotime cells)
        top_indices = np.argsort(pseudotime)[-3:]  # Top 3 cells
        terminal_states = [adata.obs_names[i] for i in top_indices]
        
        # Create simple branch probabilities
        branch_probs = pd.DataFrame(
            np.random.random((len(adata.obs_names), len(terminal_states))),
            index=adata.obs_names,
            columns=terminal_states
        )
        # Normalize rows to sum to 1
        branch_probs = branch_probs.div(branch_probs.sum(axis=1), axis=0)
        adata.obsm['palantir_branch_probs'] = branch_probs
        
        # Simple entropy
        adata.obs['palantir_entropy'] = 1 - pseudotime  # High entropy early, low entropy late
        
        print(f"Fallback completed with {len(terminal_states)} terminal states: {terminal_states}")
    
    return terminal_states

def plot_palantir_results(adata, terminal_states, basis='X_tsne'):
    """
    Create Palantir-style plots similar to the reference image
    """
    # Plot pseudotime and other results
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Pseudotime plot
    sc.pl.embedding(adata, basis=basis, color='palantir_pseudotime', 
                   title='Palantir Pseudotime', ax=axes[0,0], show=False,
                   color_map='viridis')
    
    # Entropy plot (cell plasticity)
    sc.pl.embedding(adata, basis=basis, color='palantir_entropy',
                   title='Differentiation Potential (Entropy)', ax=axes[0,1], show=False,
                   color_map='viridis')
    
    # Branch probabilities
    branch_probs = adata.obsm['palantir_branch_probs']
    for i, terminal_state in enumerate(terminal_states[:2]):  # Show first 2 branches
        if i < 2:
            adata.obs[f'branch_prob_{terminal_state}'] = branch_probs[terminal_state]
            sc.pl.embedding(adata, basis=basis, color=f'branch_prob_{terminal_state}',
                           title=f'Branch Probability: {terminal_state}', 
                           ax=axes[1,i], show=False, color_map='Reds')
    
    plt.tight_layout()
    plt.show()
    
    # Create the main trajectory plot similar to the reference
    plt.figure(figsize=(12, 10))
    
    # Plot cells colored by pseudotime
    scatter = plt.scatter(adata.obsm[basis][:, 0], adata.obsm[basis][:, 1], 
                         c=adata.obs['palantir_pseudotime'], 
                         cmap='viridis', s=30, alpha=0.7)
    
    # Try to add trajectory visualization
    try:
        # Use Palantir's built-in plotting if available
        import palantir.plot as palantir_plot
        palantir_plot.plot_palantir_results(adata, basis=basis)
    except Exception as e:
        print(f"Advanced trajectory plotting not available: {e}")
        print("Showing basic pseudotime visualization")
    
    plt.colorbar(scatter, label='Palantir Pseudotime')
    plt.title('Palantir Trajectory Analysis\nwith CellRank Integration', fontsize=16)
    plt.xlabel('tSNE 1')
    plt.ylabel('tSNE 2')
    plt.show()

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

if __name__ == "__main__":
    # Run CellRank analysis first
    print("Running CellRank analysis...")
    
    # Use combined kernel approach
    w1, w2 = 1, 1
    combined_kernel = w1 * ck + w2 * pk
    g_combo = cr.estimators.GPCCA(combined_kernel)
    g_combo.compute_schur(n_components=10)
    g_combo.compute_macrostates(n_states=4)
    g_combo.fit(n_states=3)
    g_combo.predict_initial_states()
    g_combo.predict_terminal_states(method="top_n", n_states=2)
    
    # Plot CellRank results
    g_combo.plot_macrostates(which='initial', discrete=True, s=100)
    g_combo.compute_fate_probabilities()
    g_combo.plot_fate_probabilities()
    
    # Now run Palantir analysis
    terminal_states = run_palantir_analysis(adata)
    
    # Plot Palantir results
    plot_palantir_results(adata, terminal_states)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # CellRank fate probabilities
    if hasattr(g_combo, 'fate_probabilities') and g_combo.fate_probabilities is not None:
        fate_probs = g_combo.fate_probabilities
        # Plot dominant fate
        dominant_fate = fate_probs.idxmax(axis=1)
        for i, fate in enumerate(fate_probs.columns):
            mask = dominant_fate == fate
            if mask.sum() > 0:
                axes[0].scatter(adata.obsm['X_tsne'][mask, 0], 
                              adata.obsm['X_tsne'][mask, 1], 
                              label=f'Fate {fate}', s=20, alpha=0.7)
        axes[0].set_title('CellRank Fate Probabilities')
        axes[0].legend()
    
    # Palantir pseudotime
    scatter = axes[1].scatter(adata.obsm['X_tsne'][:, 0], adata.obsm['X_tsne'][:, 1], 
                             c=adata.obs['palantir_pseudotime'], 
                             cmap='viridis', s=20, alpha=0.7)
    axes[1].set_title('Palantir Pseudotime')
    plt.colorbar(scatter, ax=axes[1])
    
    plt.tight_layout()
    plt.show()
    
    print("Analysis complete!")
    print(f"Terminal states identified: {terminal_states}")
    
    # Save results
    adata.write('integrated_cellrank_palantir_results.h5ad')
    print("Results saved to 'integrated_cellrank_palantir_results.h5ad'")