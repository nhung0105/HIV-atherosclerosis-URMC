import pandas as pd
import scanpy as sc

# Load CSV
df = pd.read_csv("/Users/nhungnguyen/Desktop/Research URoch Dr. Thakar/data/scBONITA/combined_cluster_T_cells.csv")

# Rename first column to 'gene' for clarity
df.rename(columns={df.columns[0]: "gene"}, inplace=True)

# Filter genes
genes_to_keep = ["CDC42", "CASP7", "NFKB1", "NCF2", "NLRP3", "NOS3", "TRAF6", "MAP2K6", "RAC1", "CD36", "DDIT3", 
                 "MAP2K4", "SRC", "MAP3K5", "AGER", "ATF4", "RXRA", "TANK", "CD14", "ARHGEF1", "NFKBIA", "IL18", 
                 "ROCK2", "ICAM1", "GSK3B", "HSPA4", "IL6", "PTK2", "CASP8", "LYN", "TLR2", "IKBKG", "IRAK4", 
                 "BAX", "MAP2K3", "CASP3", "IRF7", "RXRB", "RAP1A", "TLR4", "FOS", "NRAS", "JAK2", "ERN1", 
                 "NCF4", "TRAF3", "JUN", "CASP9", "IL12A", "NFATC1", "TNF", "EIF2S1", "KRAS", "HSPA5", "IL1B", 
                 "PYCARD", "RAP1B", "EIF2AK3", "CCL5", "CASP6", "MYD88", "NFATC3", "CCL3", "TNFRSF1A", "RELA", 
                 "NFATC2", "MAP3K7", "BID", "TLR6", "TRAF2", "IKBKE", "NCF1", "IRF3", "MAP2K7", "TIRAP", "STAT3", 
                 "TBK1", "IKBKB", "RHOA", "CASP1", "HRAS", "XBP1", "IRAK1", "BCL2", "CYBA", "LY96"]
filtered_df = df[df["gene"].isin(genes_to_keep)]

# Save filtered CSV
filtered_df.to_csv("filtered_combined_cluster_T_cell.csv", index=False)

# Prepare for AnnData: set gene names as index
filtered_df.set_index("gene", inplace=True)

# Create AnnData object (transpose to cells x genes)
adata = sc.AnnData(filtered_df.T)

# Save .h5ad
adata.write("/Users/nhungnguyen/Desktop/Research URoch Dr. Thakar/data/scBONITA/combined_genes.h5ad")

print("AnnData object saved.")
print("Shape (cells, genes):", adata.shape)
