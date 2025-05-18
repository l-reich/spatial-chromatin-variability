import numpy as np
import pandas as pd
import anndata


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
    size=30
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