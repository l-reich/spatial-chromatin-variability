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