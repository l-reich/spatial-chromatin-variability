import requests
import gzip
import io
import re
import pandas as pd
import numpy as np # Needed for future potential coordinate handling, good to include

# Helper function to parse GTF attributes string
# It's good to define this outside the main function if it might be reused,
# or just before it if specific to this processing.
# Let's define it outside for clarity.
def _parse_gtf_attributes(attribute_string):
    """Parses the attributes string from a GTF file line."""
    attributes = {}
    # Split by semicolon and trim whitespace
    pairs = [pair.strip() for pair in attribute_string.split(';') if pair.strip()]
    for pair in pairs:
        # Split by the first space to separate key from value
        parts = pair.split(' ', 1)
        if len(parts) == 2:
            key = parts[0]
            # Remove quotes from the value
            value = parts[1].strip('"')
            attributes[key] = value
    return attributes

# Main function to load and process the GTF file
def load_gencode_gtf_genes(gtf_url: str, feature_type: str = 'gene', exclude_chroms: list = None) -> pd.DataFrame:
    """
    Downloads, parses, and processes a GENCODE GTF file to extract gene coordinates.

    Args:
        gtf_url: URL of the gzipped GTF file.
        feature_type: The feature type to filter for (e.g., 'gene', 'transcript').
                      Defaults to 'gene'.
        exclude_chroms: A list of chromosome names to exclude (e.g., ['chrY']).

    Returns:
        A pandas DataFrame with gene coordinates and annotations, indexed by gene_name,
        including 'chrom', 'start', 'end', 'Strand', 'gene_ids', and 'interval' columns.

    Raises:
        requests.exceptions.RequestException: If the file cannot be downloaded.
        Exception: If there's an error processing the GTF file.
    """
    if exclude_chroms is None:
        exclude_chroms = []

    # --- Download and Read the GTF file ---
    try:
        response = requests.get(gtf_url)
        response.raise_for_status() # Raise an exception for bad status codes

        # Read the gzipped content
        with gzip.open(io.BytesIO(response.content), 'rt') as f:
            # Read line by line to skip comment lines (though comment='# is used below)
            # This step might not be strictly necessary depending on read_csv,
            # but explicitly filtering can prevent issues.
            gtf_lines = [line for line in f if not line.startswith('#')]

    except requests.exceptions.RequestException as e:
        print(f"Error downloading or reading file from {gtf_url}: {e}")
        # Re-raise the exception so the caller can handle it
        raise e
    except Exception as e:
        print(f"An unexpected error occurred during file download or read: {e}")
        raise e

    # --- Parse the GTF file ---
    try:
        # Read the filtered lines into a pandas DataFrame
        # Specify column names as GTF doesn't have a header
        gtf_df = pd.read_csv(
            io.StringIO("".join(gtf_lines)),
            sep='\t',
            header=None,
            comment='#', # Use comment to skip lines starting with #
            names=[
                'seqname', 'source', 'feature', 'start', 'end',
                'score', 'Strand', 'frame', 'attribute'
            ]
        )

        # Filter for the specified feature type
        features_df = gtf_df[gtf_df['feature'] == feature_type].copy()

        # Parse the attributes column
        attribute_data = features_df['attribute'].apply(_parse_gtf_attributes)

        # Extract gene_id and gene_name
        features_df['gene_id'] = attribute_data.apply(lambda x: x.get('gene_id'))
        features_df['gene_name'] = attribute_data.apply(lambda x: x.get('gene_name'))

        # Select and rename relevant columns
        # Ensure columns are in appropriate data types (start/end as integers)
        features_df['start'] = features_df['start'].astype(int)
        features_df['end'] = features_df['end'].astype(int)

        gene_coords_df = features_df[[
            'seqname', 'start', 'end', 'Strand', 'gene_name', 'gene_id'
        ]].rename(columns={
            'seqname': 'chrom',
            'gene_id': 'gene_ids'
        }).copy()

        # Filter out specified chromosomes
        if exclude_chroms:
             gene_coords_df = gene_coords_df[~gene_coords_df["chrom"].isin(exclude_chroms)]


        # Set gene_name as index
        gene_coords_df = gene_coords_df.set_index('gene_name')

        # Create the 'interval' column
        gene_coords_df['interval'] = gene_coords_df['chrom'].astype(str) + ':' + \
                                     gene_coords_df['start'].astype(str) + '-' + \
                                     gene_coords_df['end'].astype(str)

        # Optional: Handle potential duplicate gene_names if groupby agg is needed later
        # For simplicity here, we assume gene_name is unique or min/max is handled by caller

        print(f"Successfully processed GTF from {gtf_url}.")
        print(f"Number of '{feature_type}' features found: {len(gene_coords_df)}")
        if exclude_chroms:
             print(f"Excluded chromosomes: {exclude_chroms}")

        return gene_coords_df

    except Exception as e:
        print(f"An error occurred during GTF parsing or processing: {e}")
        # Re-raise the exception
        raise e

# Example usage (if this were in a script, not typically in a notebook):
# gtf_url = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M10/gencode.vM10.annotation.gtf.gz"
# try:
#     gencode_genes = load_gencode_gtf_genes(gtf_url, exclude_chroms=['chrY'])
#     print(gencode_genes.head())
# except Exception as e:
#     print("Processing failed.")