import numpy as np
import pandas as pd
import anndata
import matplotlib.pyplot as plt
import seaborn as sns
import squidpy as sq
from scipy.stats import pearsonr, spearmanr, kruskal


def get_top_correlated_peaks_near_gene(
    gene_name: str,
    expr: anndata.AnnData,
    atac: anndata.AnnData,
    g_p_similarity: np.ndarray,
    top_n: int = 100,
    max_distance: int = 1_000_000  # in base pairs
) -> pd.DataFrame:
    """
    Retrieve top correlated ATAC peaks for a gene, restricted to peaks on the same chromosome
    and within a certain distance from the gene's midpoint.

    Parameters:
    - gene_name (str): Name of the gene.
    - expr (AnnData): Expression data with gene positions in .var ('chrom', 'start', 'end').
    - atac (AnnData): ATAC data with peak positions in .var ('chrom', 'start', 'end').
    - g_p_similarity (np.ndarray): Gene-peak correlation matrix (genes x peaks).
    - top_n (int): Number of top correlated peaks to consider.
    - max_distance (int): Max genomic distance from gene midpoint to include peaks.

    Returns:
    - pd.DataFrame: Peaks on the same chromosome within distance, sorted by distance.
    """
    if gene_name not in expr.var_names:
        raise ValueError(f"Gene {gene_name} not found in expr.var_names.")
    
    gene_info = expr.var.loc[gene_name]
    chrom = gene_info["chrom"]
    if pd.isnull(gene_info.get("start")) or pd.isnull(gene_info.get("end")):
        raise ValueError(f"Coordinates missing for gene {gene_name}.")

    gene_mid = (gene_info["start"] + gene_info["end"]) // 2

    gene_idx = expr.var_names.get_loc(gene_name)
    peak_scores = g_p_similarity[gene_idx, :]
    top_peak_indices = np.argsort(peak_scores)[::-1][:top_n]

    top_peaks = atac.var.iloc[top_peak_indices].copy()
    top_peaks = top_peaks[top_peaks["chrom"] == chrom]

    # Compute peak midpoint and distance
    top_peaks["peak_mid"] = (top_peaks["start"] + top_peaks["end"]) // 2
    top_peaks["distance"] = (top_peaks["peak_mid"] - gene_mid).abs()

    # Filter and return
    return top_peaks[top_peaks["distance"] <= max_distance].sort_values("distance")


def get_filtered_peak_results_by_gene(
    gene_list,
    expr,
    atac,
    g_p_similarity,
    ccre_bed,
    top_n=200,
    max_distance=100_000,
    keep_annotated=True
):
    """
    Identify top correlated ATAC peaks for each gene in a list, optionally filtering
    for overlap with known cCRE annotations.

    Parameters:
    - gene_list (list): List of gene names to evaluate.
    - expr (AnnData): AnnData object with gene expression and gene positions 
                      in `.var['chrom']`, `.var['start']`, `.var['end']`.
    - atac (AnnData): AnnData object with ATAC peaks and positions 
                      in `.var['chrom']`, `.var['start']`, `.var['end']`.
    - g_p_similarity (np.ndarray): Gene-peak similarity matrix (shape: genes × peaks).
    - ccre_bed (pd.DataFrame): DataFrame with cCRE annotations, containing columns 
                               'chrom', 'start', and 'end'.
    - top_n (int): Number of top correlated peaks to consider per gene.
    - max_distance (int): Maximum genomic distance (bp) from gene to peak to include.
    - keep_annotated (bool): 
        - True: keep only peaks that overlap a cCRE
        - False: keep only peaks that do NOT overlap any cCRE

    Returns:
    - dict: Mapping from gene name to a DataFrame of filtered correlated peaks.
    """
    results_by_gene = {}

    for gene in gene_list:
        try:
            peaks = get_top_correlated_peaks_near_gene(
                gene_name=gene,
                expr=expr,
                atac=atac,
                g_p_similarity=g_p_similarity,
                top_n=top_n,
                max_distance=max_distance
            )

            if not peaks.empty:
                # Prepare peak coordinates
                peaks_reset = peaks.reset_index().rename(columns={
                    "start": "start_peak", "end": "end_peak", "index": "peak_name"
                })

                # Merge with cCREs on chromosome
                merged = peaks_reset.merge(ccre_bed, how="inner", on="chrom")

                # Overlap condition
                overlaps = merged[
                    (merged["start_peak"] <= merged["end"]) &
                    (merged["end_peak"] >= merged["start"])
                ]

                overlapping_peak_names = overlaps["peak_name"].unique()

                if keep_annotated:
                    peaks = peaks.loc[peaks.index.isin(overlapping_peak_names)]
                else:
                    peaks = peaks.loc[~peaks.index.isin(overlapping_peak_names)]

                if not peaks.empty:
                    results_by_gene[gene] = peaks

        except Exception as e:
            print(f"Skipping {gene}: {e}")

    return results_by_gene


def get_overlapping_ccres(peak_name: str, ccre_bed: pd.DataFrame) -> pd.DataFrame:
    """
    Return overlapping cCREs from a BED dataframe for a given peak.

    Parameters:
    - peak_name (str): Peak in the format 'chrX:start-end' (e.g., 'chr1:12345-12456').
    - ccre_bed (pd.DataFrame): BED-style DataFrame with columns 'chrom', 'start', 'end'.

    Returns:
    - pd.DataFrame: Subset of ccre_bed rows that overlap with the peak.
    """
    try:
        chrom, coords = peak_name.split(":")
        peak_start, peak_end = map(int, coords.split("-"))
    except Exception as e:
        raise ValueError(f"Invalid peak format: {peak_name}") from e

    overlaps = ccre_bed[
        (ccre_bed["chrom"] == chrom) &
        (ccre_bed["start"] <= peak_end) &
        (ccre_bed["end"] >= peak_start)
    ]
    
    return overlaps


def get_genes_associated_with_peaks(
    peak_names,
    expr,
    atac,
    g_p_similarity,
    ccre_bed=None,
    keep_annotated=None,
    max_distance=100_000,
    top_n_per_peak=10
):
    """
    Identify genes that are strongly correlated and genomically proximal to given peaks.

    Parameters:
    - peak_names (list): List of peak names (e.g., 'chr1:12345-12456').
    - expr (AnnData): AnnData with gene annotations in `.var['chrom']`, `.var['start']`, `.var['end']`.
    - atac (AnnData): AnnData with peak annotations in `.var['chrom']`, `.var['start']`, `.var['end']`.
    - g_p_similarity (np.ndarray): Gene-peak similarity matrix (genes x peaks).
    - ccre_bed (pd.DataFrame, optional): BED DataFrame with cCRE annotations.
    - keep_annotated (bool, optional): If True, only use peaks that overlap a cCRE.
                                       If False, only use peaks that do NOT overlap a cCRE.
                                       If None, skip cCRE filtering.
    - max_distance (int): Max distance (in bp) allowed between peak and gene midpoint.
    - top_n_per_peak (int): Maximum number of genes to return per peak.

    Returns:
    - dict: Mapping from peak name to list of associated genes.
    - list: Unique list of all associated genes.
    """
    peak_to_index = {name: i for i, name in enumerate(atac.var_names)}

    if keep_annotated is not None and ccre_bed is None:
        raise ValueError("ccre_bed must be provided if keep_annotated is True or False")

    # Transpose similarity to peak × gene
    similarity_T = g_p_similarity.T

    top_genes_by_peak = {}

    for peak_name in peak_names:
        if peak_name not in peak_to_index:
            continue

        # Optionally filter by cCRE annotation
        if keep_annotated is not None:
            overlaps = get_overlapping_ccres(peak_name, ccre_bed)
            if keep_annotated and overlaps.empty:
                continue  # skip non-annotated
            if not keep_annotated and not overlaps.empty:
                continue  # skip annotated

        peak_idx = peak_to_index[peak_name]
        peak_row = atac.var.loc[peak_name]

        peak_chr = peak_row["chrom"]
        peak_mid = (peak_row["start"] + peak_row["end"]) // 2

        correlations = similarity_T[peak_idx]
        top_gene_indices = np.argsort(correlations)[::-1]

        filtered_genes = []

        for gene_idx in top_gene_indices:
            gene_name = expr.var_names[gene_idx]
            gene_row = expr.var.loc[gene_name]

            if gene_row["chrom"] != peak_chr:
                continue

            gene_mid = (gene_row["start"] + gene_row["end"]) // 2
            distance = abs(peak_mid - gene_mid)

            if distance <= max_distance:
                filtered_genes.append(gene_name)

            if len(filtered_genes) == top_n_per_peak:
                break

        if filtered_genes:
            top_genes_by_peak[peak_name] = filtered_genes

    #unique_genes = list(set(g for genes in top_genes_by_peak.values() for g in genes))
    #return top_genes_by_peak, unique_genes
    return top_genes_by_peak


def plot_peak_gene_pairs(
    mapping_dict,
    expr_adata,
    atac_adata,
    mode='peak_to_gene',
    n_pairs=10,
    size=20
):
    """
    Plot all gene-peak spatial expression pairs side-by-side using squidpy.

    Parameters:
    - mapping_dict (dict): Mapping from peak → [genes] or gene → [peaks] or DataFrame
    - expr_adata (AnnData): Expression AnnData (for genes)
    - atac_adata (AnnData): ATAC AnnData (for peaks)
    - mode (str): 'peak_to_gene' or 'gene_to_peak'
    - n_pairs (int): Max number of pairs to plot
    - size (int): Point size in plots

    Returns:
    - None
    """
    pairs = []

    if mode == 'peak_to_gene':
        for peak, genes in mapping_dict.items():
            for gene in genes:
                pairs.append((peak, gene))

    elif mode == 'gene_to_peak':
        for gene, peaks in mapping_dict.items():
            if isinstance(peaks, pd.DataFrame):
                peak_names = peaks.index.tolist()
            else:
                peak_names = peaks
            for peak in peak_names:
                pairs.append((peak, gene))

    else:
        raise ValueError("mode must be 'peak_to_gene' or 'gene_to_peak'")

    # Limit total number of pairs
    pairs = pairs[:n_pairs]

    for i, (peak, gene) in enumerate(pairs):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(f"Pair {i+1}: {gene} ↔ {peak}", fontsize=14)

        if gene in expr_adata.var_names:
            sq.pl.spatial_scatter(
                expr_adata, color=gene, ax=axes[0], size=size, shape=None
            )
            axes[0].set_title(f"Gene: {gene}")
        else:
            axes[0].set_title(f"Gene: {gene} (not found)")

        if peak in atac_adata.var_names:
            sq.pl.spatial_scatter(
                atac_adata, color=peak, ax=axes[1], size=size, shape=None
            )
            axes[1].set_title(f"Peak: {peak}")
        else:
            axes[1].set_title(f"Peak: {peak} (not found)")

        plt.tight_layout()
        plt.show()


def plot_peak_peak_pairs(
    mapping_dict,
    atac_adata,
    n_pairs=10,
    size=20
):
    """
    Plot all peak–peak spatial accessibility pairs side-by-side using squidpy,
    annotated with their genomic distance.

    Parameters:
    - mapping_dict (dict): Mapping from peak → [other peaks]
    - atac_adata (AnnData): AnnData object with peak annotations
    - n_pairs (int): Max number of peak–peak pairs to plot
    - size (int): Dot size for scatter plots

    Returns:
    - None
    """
    pairs = []

    for peak, other_peaks in mapping_dict.items():
        for other_peak in other_peaks:
            pairs.append((peak, other_peak))

    # Limit number of pairs
    pairs = pairs[:n_pairs]

    for i, (peak1, peak2) in enumerate(pairs):
        # Default distance string
        distance_str = "unknown"

        # Try to compute genomic distance
        if all(p in atac_adata.var_names for p in [peak1, peak2]):
            row1 = atac_adata.var.loc[peak1]
            row2 = atac_adata.var.loc[peak2]
            if row1["chrom"] == row2["chrom"]:
                mid1 = (row1["start"] + row1["end"]) // 2
                mid2 = (row2["start"] + row2["end"]) // 2
                distance_str = f"{abs(mid1 - mid2):,} bp"

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(f"Pair {i+1}: {peak1} ↔ {peak2}  |  Distance: {distance_str}", fontsize=12)

        if peak1 in atac_adata.var_names:
            sq.pl.spatial_scatter(
                atac_adata, color=peak1, ax=axes[0], size=size, shape=None
            )
            axes[0].set_title("Peak 1")
        else:
            axes[0].set_title("Peak 1 (not found)")

        if peak2 in atac_adata.var_names:
            sq.pl.spatial_scatter(
                atac_adata, color=peak2, ax=axes[1], size=size, shape=None
            )
            axes[1].set_title("Peak 2")
        else:
            axes[1].set_title("Peak 2 (not found)")

        plt.tight_layout()
        plt.show()




def plot_peak_accessibility(
    peak_name: str,
    atac: anndata.AnnData,
    threshold: float = 0.7,
    size: int = 5,
    cmap: str = "viridis"
):
    """
    Plot spatial accessibility of a peak, greying out spots below a threshold.

    Parameters:
    - peak_name (str): Name of the peak (must be in atac.var_names).
    - atac (AnnData): ATAC AnnData object with spatial coordinates in .obsm['spatial'].
    - threshold (float): Minimum accessibility value to color a spot (else greyed out).
    - size (int): Dot size in the plot.
    - cmap (str): Colormap for non-zero accessibility.

    Returns:
    - None
    """
    if peak_name not in atac.var_names:
        raise ValueError(f"Peak {peak_name} not found in atac.var_names.")

    # Get peak values
    peak_idx = atac.var_names.get_loc(peak_name)
    values_sparse = atac.X[:, peak_idx]
    values = np.array(values_sparse.todense()).flatten()

    # Get spatial coordinates
    coords = atac.obsm["spatial"]

    # Masks
    is_low = values < threshold
    is_high = values >= threshold

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(coords[is_low, 0], coords[is_low, 1], c="lightgray", s=size, label="low")
    sc = ax.scatter(coords[is_high, 0], coords[is_high, 1], c=values[is_high],
                    cmap=cmap, s=size)
    plt.colorbar(sc, ax=ax).set_label("Peak Accessibility")
    ax.set_title(peak_name)
    ax.set_xlabel("spatial1")
    ax.set_ylabel("spatial2")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


def get_promoter_overlapping_peaks(peaks_df, promoter_ccres):
    """
    Filter peaks that overlap with promoter-like cCREs (PLS).
    
    Parameters:
    - peaks_df: pd.DataFrame with index = peak names and columns ['chrom', 'start', 'end']
    - promoter_ccres: pd.DataFrame of cCREs with 'chrom', 'start', 'end'

    Returns:
    - List of overlapping peak names
    """
    overlaps = peaks_df.reset_index().merge(
        promoter_ccres, on="chrom", how="inner", suffixes=("_peak", "_ccre")
    )

    overlaps = overlaps[
        (overlaps["start_peak"] <= overlaps["end_ccre"]) &
        (overlaps["end_peak"] >= overlaps["start_ccre"])
    ]
    return overlaps["index"].unique().tolist()


def get_correlated_peaks_near_peaks(
    peak_names,
    atac,
    p_p_similarity,
    max_distance=100_000,
    top_n_per_peak=10
):
    """
    Identify spatially correlated and genomically proximal peaks for each input peak.

    Parameters:
    - peak_names (list): List of input peak names (e.g., 'chr1:12345-12456').
    - atac (AnnData): ATAC AnnData object with peak annotations in .var['chrom', 'start', 'end'].
    - p_p_similarity (np.ndarray): Peak–peak similarity matrix (shape: peaks x peaks).
    - max_distance (int): Max genomic distance in bp.
    - top_n_per_peak (int): Number of top correlated nearby peaks to return per peak.

    Returns:
    - dict: Mapping from each peak to a list of nearby correlated peaks.
    """
    peak_to_index = {name: i for i, name in enumerate(atac.var_names)}
    results = {}

    for peak_name in peak_names:
        if peak_name not in peak_to_index:
            continue

        peak_idx = peak_to_index[peak_name]
        peak_row = atac.var.loc[peak_name]
        peak_chr = peak_row["chrom"]
        peak_mid = (peak_row["start"] + peak_row["end"]) // 2

        correlations = p_p_similarity[peak_idx]
        top_peak_indices = np.argsort(correlations)[::-1]

        filtered_peaks = []

        for other_idx in top_peak_indices:
            if other_idx == peak_idx:
                continue  # skip self

            other_peak = atac.var_names[other_idx]
            other_row = atac.var.loc[other_peak]

            if other_row["chrom"] != peak_chr:
                continue

            other_mid = (other_row["start"] + other_row["end"]) // 2
            distance = abs(peak_mid - other_mid)

            if distance <= max_distance:
                filtered_peaks.append(other_peak)

            if len(filtered_peaks) == top_n_per_peak:
                break

        if filtered_peaks:
            results[peak_name] = filtered_peaks

    return results


def analyze_gene_peak_relationship(expr, atac, gene, peak, subclass_key="subclass", plot=False, method="spearman"):
    """
    Analyze the relationship between a gene's expression and a peak's accessibility across subclasses.
    Returns the Spearman or Pearson correlation coefficient between mean expression and accessibility.

    Parameters:
        expr: AnnData object with gene expression.
        atac: AnnData object with peak accessibility.
        gene: str, gene name (must be in expr.var_names).
        peak: str, peak name (must be in atac.var_names).
        subclass_key: str, column in .obs indicating cell types.
        plot: bool, whether to show violin and scatter plots.
        method: str, one of {"spearman", "pearson"}

    Returns:
        rho: Correlation coefficient between mean expression and accessibility across subclasses.
    """
    # Extract and store vectors
    expr.obs[f"{gene}_expr"] = expr[:, gene].X.toarray().flatten()
    atac.obs[f"access_{gene}_peak"] = atac[:, peak].X.toarray().flatten()

    # Means per subclass
    mean_expr = expr.obs.groupby(subclass_key)[f'{gene}_expr'].mean()
    mean_access = atac.obs.groupby(subclass_key)[f'access_{gene}_peak'].mean()
    comparison_df = pd.DataFrame({'mean_expr': mean_expr, 'mean_access': mean_access}).dropna()

    # Choose correlation method
    if method == "spearman":
        from scipy.stats import spearmanr
        rho, _ = spearmanr(comparison_df['mean_expr'], comparison_df['mean_access'])
        label = "Spearman ρ"
    elif method == "pearson":
        from scipy.stats import pearsonr
        rho, _ = pearsonr(comparison_df['mean_expr'], comparison_df['mean_access'])
        label = "Pearson r"
    else:
        raise ValueError("Method must be 'spearman' or 'pearson'.")

    # Plotting
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))
        sns.violinplot(data=atac.obs, x=subclass_key, y=f'access_{gene}_peak')
        plt.xticks(rotation=90)
        plt.title(f'Accessibility of {peak} (linked to {gene})')
        plt.show()

        plt.figure(figsize=(12, 4))
        sns.violinplot(data=expr.obs, x=subclass_key, y=f'{gene}_expr')
        plt.xticks(rotation=90)
        plt.title(f'Expression of {gene}')
        plt.show()

        sns.regplot(data=comparison_df, x='mean_expr', y='mean_access')
        plt.xlabel(f'Mean {gene} Expression')
        plt.ylabel(f'Mean {gene} Promoter Accessibility')
        plt.title(f'{label} = {rho:.2f}')
        plt.show()

    return rho
