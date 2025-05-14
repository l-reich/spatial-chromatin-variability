import scanpy as sc
import numpy as np
import pandas as pd
import muon as mu

from src.utils import get_centroid
from src import config

### Load data

segmented_cells_path = config.DATA_DIR / "spatial_data" / "MERFISH_MOp" / "processed_data" / "segmented_cells_mouse1sample4.csv"

# Load file containing cell boundaries
segmented_cells = pd.read_csv(segmented_cells_path).reset_index().rename(columns={"index": "cell_id"})

slice = "mouse1_slice201"

segmented_cells = segmented_cells[segmented_cells["slice_id"] == slice]

# Load counts h5ad file

counts_path = config.DATA_DIR / "spatial_data" / "MERFISH_MOp" / "processed_data" / "counts.h5ad"

#adata_st = sc.read_h5ad("../../data/spatial_data/MERFISH_MOp/processed_data/counts.h5ad")
adata_st = sc.read_h5ad(counts_path)

# Load metadata

labels_path = config.DATA_DIR / "spatial_data" / "MERFISH_MOp" / "processed_data" / "cell_labels.csv"

#cell_labels = pd.read_csv("../../data/spatial_data/MERFISH_MOp/processed_data/cell_labels.csv", index_col=0)  # Use sample_id as index
cell_labels = pd.read_csv(labels_path, index_col=0)  # Use sample_id as index

# Ensure index matches adata_st.var
cell_labels = cell_labels.loc[adata_st.obs.index]  

# Add to AnnData
adata_st.obs = cell_labels

adata_st = adata_st[adata_st.obs["slice_id"] == slice]

segmented_cells["centroid_x"], segmented_cells["centroid_y"] = zip(*segmented_cells.apply(get_centroid, axis=1))

segmented_cells["cell_id"] = segmented_cells["Unnamed: 0"].astype(str)
adata_st.obs["cell_id"] = adata_st.obs.index.astype(str)

adata_st.obs = adata_st.obs.merge(segmented_cells[["cell_id", "centroid_x", "centroid_y"]], 
                                  on="cell_id", how="left")
adata_st.obsm["spatial"] = np.array(adata_st.obs[["centroid_x", "centroid_y"]])


### removal of subcortical cells

# Extract Nxph4 expression values
nxph4_expr = adata_st[:, "Nxph4"].X.toarray().flatten()  # Ensure it's a 1D array

# Create a boolean mask for cells with Nxph4 > cutoff
high_nxph4_mask = nxph4_expr > 7.5

# Subset the AnnData object
adata_st_high_nxph4 = adata_st[high_nxph4_mask].copy()

# Extract spatial coordinates
spatial_coords_high = adata_st_high_nxph4.obsm["spatial"]
x_high = spatial_coords_high[:, 0]  # X-coordinates
y_high = spatial_coords_high[:, 1]  # Y-coordinates

# Fit a quadratic polynomial
coeffs = np.polyfit(x_high, y_high, deg=2)  # Fit y = ax² + bx + c
poly_curve = np.poly1d(coeffs)

# Generate fitted y-values for a smooth curve
x_fit = np.linspace(x_high.min(), x_high.max(), 100)
y_fit = poly_curve(x_fit)

# Extract full spatial coordinates
spatial_coords = adata_st.obsm["spatial"]

# Compute boundary y-values for all x positions
boundary_y = poly_curve(spatial_coords[:, 0])

# Identify subcortical cells (cells below the fitted boundary)
subcortical_mask = spatial_coords[:, 1] > boundary_y

# Remove subcortical cells from the full dataset
adata_filtered = adata_st[~subcortical_mask].copy()

# Extract Nxph4 expression values
hi_nx = adata_filtered[:, "Nxph4"].X.toarray().flatten()

# Create a boolean mask for cells with Nxph4 > cutoff
hi_nxph4_mask = hi_nx > 7.5

adata_filtered=adata_filtered[~hi_nxph4_mask]

### Drop the NaN column in obsm

mask_nan_rows = np.any(np.isnan(adata_filtered.obsm["spatial"]), axis=1)

mask_finite_rows = ~mask_nan_rows

adata_filtered = adata_filtered[mask_finite_rows]

### QC/filtering cells with low gene counts

sc.pp.calculate_qc_metrics(
    adata_filtered, inplace=True, log1p=True, percent_top=(5, 10, 50)
)

print(f"Total number of cells: {adata_filtered.n_obs}")
mu.pp.filter_obs(
    adata_filtered,
    "total_counts",
    lambda x:  x >= 55,
)
print(f"Number of cells after filtering on total_fragment_counts: {adata_filtered.n_obs}")

### Preprocessing done, write h5ad file on disk

h5ad_path = config.H5AD_DIR / "c_sp_ad.h5ad"

adata_filtered.write(filename=h5ad_path)