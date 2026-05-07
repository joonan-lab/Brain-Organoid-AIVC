#!/usr/bin/env python3
"""
Cluster Enrichment Analysis using Fisher's Exact Test.
ASD185_EAGLE only version.

This script performs enrichment analysis for:
- ASD185_EAGLE (known ASD genes) only

Method: Fisher's exact test with FDR correction
Significant cluster: FDR < 0.05 AND Odds Ratio > 1
"""
import warnings
warnings.filterwarnings('ignore')

import os, glob
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import scipy.stats as st
from statsmodels.stats.multitest import fdrcorrection
import argparse



def load_annotation_data(annot_path):
    """Load gene annotation data."""
    print("Loading annotation data...")
    annot_df = pd.read_excel(annot_path)
    
    return annot_df


def load_cluster_data(adata_path):
    """Load cluster assignments."""
    adata = sc.read_h5ad(adata_path)
    df = adata.obs.copy()    
    cluster_key = [x for x in df.columns if x.startswith('leiden_')][0]
    
    df['cluster'] = df[cluster_key].astype(str).copy()
    df['gene'] = [x.split('_perturbed')[0] for x in df.index] if 'gene' not in df.columns else df['gene']
    
    clust_counts = pd.DataFrame(df['cluster'].value_counts()).reset_index()
    valid_clusters = clust_counts.loc[clust_counts['count']>=30, 'cluster'].tolist()
    
    valid_df = df.loc[df['cluster'].isin(valid_clusters)]
    
    return valid_df


def perform_fisher_test(data_df, feature_mask, feature_name):
    """
    Perform Fisher's exact test for enrichment analysis per cluster.
    """
    rows = []

    for cl in sorted(data_df['cluster'].unique(), key=lambda x: int(x) if x.isdigit() else x):
        in_cluster = data_df['cluster'] == cl
        feature_in = feature_mask & in_cluster
        non_feature_in = in_cluster.sum() - feature_in.sum()

        out_cluster = ~in_cluster
        feature_out = feature_mask & out_cluster
        non_feature_out = out_cluster.sum() - feature_out.sum()

        contingency = [[feature_in.sum(), non_feature_in],
                       [feature_out.sum(), non_feature_out]]

        odds_ratio, p_val = st.fisher_exact(contingency, alternative='greater')

        rows.append({
            'cluster': cl,
            f'{feature_name}_in_cluster': feature_in.sum(),
            'total_in_cluster': in_cluster.sum(),
            f'{feature_name}_pct_in_cluster': feature_in.sum() / in_cluster.sum() * 100 if in_cluster.sum() > 0 else 0,
            'OR': odds_ratio,
            'p_val': p_val,
        })

    enrichment_df = pd.DataFrame(rows)

    # FDR correction
    enrichment_df['fdr'] = fdrcorrection(enrichment_df['p_val'])[1]

    # Get significant clusters (FDR < 0.05 AND Odds Ratio > 1)
    significant = enrichment_df[
        (enrichment_df['fdr'] < 0.05) &
        (enrichment_df['OR'] > 1)
    ]
    effective_clusters = significant['cluster'].tolist()

    return enrichment_df, effective_clusters


def analyze_model(model_name, adata_path, annot_df, feature_name):
    """Perform enrichment analysis for a single model."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*60}")

    # Load cluster data
    cluster_df = load_cluster_data(adata_path)
    print(f"Loaded {len(cluster_df)} genes with {cluster_df['cluster'].nunique()} clusters")

    # Merge with annotation
    if feature_name in cluster_df.columns:
        merged_df = cluster_df.copy()
        merged_df[feature_name] = merged_df[feature_name].tolist()
        merged_df[feature_name] = merged_df[feature_name].map({'TRUE':1, 'FALSE':0})
    else:
        merged_df = cluster_df.merge(annot_df[['gene', feature_name]],
                                    on='gene', how='left')

    # Create feature mask (ASD185_EAGLE only)
    feature_mask = merged_df[feature_name].fillna(0).astype(bool)

    # Perform enrichment analysis
    enrichment_df, effective_clusters = perform_fisher_test(merged_df, feature_mask, feature_name)

    results = {
        'enrichment_df': enrichment_df,
        'effective_clusters': effective_clusters,
        'n_positive': feature_mask.sum(),
    }
    print(f"  - {feature_name}: {feature_mask.sum()} positive genes, {len(effective_clusters)} significant clusters")

    return results, merged_df


def generate_report(model_name, results, merged_df, output_file, feature_name):
    """Generate enrichment report for a model."""
    lines = []

    lines.append("=" * 80)
    lines.append(f"CLUSTER ENRICHMENT ANALYSIS for {feature_name}: {model_name}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Method: Fisher's exact test (one-sided, alternative='greater')")
    lines.append("Significance: FDR < 0.05 AND Odds Ratio > 1")
    lines.append("Feature: ASD185_EAGLE (known ASD genes) only")
    lines.append("")

    # Summary
    lines.append("-" * 80)
    lines.append("SUMMARY")
    lines.append("-" * 80)
    lines.append(f"Total genes: {len(merged_df)}")
    lines.append(f"Number of clusters: {merged_df['cluster'].nunique()}")
    lines.append(f"{feature_name} positive genes: {results['n_positive']}")
    lines.append("")
    lines.append(f"Significant clusters: {', '.join(results['effective_clusters']) if results['effective_clusters'] else 'None'}")
    lines.append("")

    # Detailed results
    lines.append("=" * 80)
    lines.append("DETAILED RESULTS: ASD185_EAGLE")
    lines.append("=" * 80)
    lines.append("")

    df = results['enrichment_df']
    
    # Significant clusters
    sig_df = df[(df['fdr'] < 0.05) & (df['OR'] > 1)]
    lines.append(f"Significant clusters ({len(sig_df)}):")
    if len(sig_df) > 0:
        display_cols = ['cluster', f'{feature_name}_in_cluster', 'total_in_cluster',
                       f'{feature_name}_pct_in_cluster', 'OR', 'fdr']
        lines.append(sig_df[display_cols].to_string(index=False))
    else:
        lines.append("  None")
    lines.append("")

    # Non-significant clusters
    nonsig_df = df[~((df['fdr'] < 0.05) & (df['OR'] > 1))]
    lines.append(f"Non-significant clusters ({len(nonsig_df)}):")
    if len(nonsig_df) > 0:
        display_cols = ['cluster', f'{feature_name}_in_cluster', 'total_in_cluster',
                       f'{feature_name}_pct_in_cluster', 'OR', 'fdr']
        lines.append(nonsig_df[display_cols].to_string(index=False))
    lines.append("")

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    return results['effective_clusters']


def main(parser):
    args = parser.parse_args()
    
    # Directories
    CLUSTER_DIR = args.cluster_dir #"<PROJECT_ROOT>/results_clustering"
    OUTPUT_DIR = args.save_dir # f"{CLUSTER_DIR}/Effective_clusters"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Annotation file
    annot_path = args.annot_file # Path("<DATA_ROOT>/resources/ASD_risk_genes_ASD185_EAGLE_ASD_NDD664.250901.xlsx")
    feature_name = args.feature_name # 'ASD185_EAGLE'

    # Cluster files with NEW naming convention
    adata_files = glob.glob(CLUSTER_DIR + '/*.h5ad')
    model_names = [os.path.basename(x).split(args.split_word)[0] for x in adata_files]
    adata_list = {k: v for k, v in zip(model_names, adata_files)}
    
    
    """Main analysis function."""
    print("=" * 60)
    print(f"CLUSTER ENRICHMENT ANALYSIS ({feature_name}) for {model_names}")
    print("=" * 60)

    # Load annotation data
    annot_df = load_annotation_data(annot_path)

    # Store summary across all models
    all_model_results = {}

    # Analyze each model
    for model_name, adata_path in adata_list.items():
        results, merged_df = analyze_model(model_name, adata_path, annot_df, feature_name)

        # Generate report
        output_file = Path(f"{OUTPUT_DIR}/enrichment_results/{model_name}_{feature_name}_enrichment_result.txt")
        effective_clusters = generate_report(model_name, results, merged_df, output_file, feature_name)
        print(f"  - Report saved to: {output_file.name}")

        all_model_results[model_name] = {
            'results': results,
            'effective_clusters': effective_clusters,
        }
        meta_file = f"{OUTPUT_DIR}/metadata/{model_name}_perturbed_gene_metadata.csv"
        merged_df.to_csv(meta_file, sep=",")
        print(f"  - Annotated perturbation gene metadata file saved to: {meta_file}")

    # Generate cross-model summary
    print("\n" + "=" * 60)
    print("GENERATING CROSS-MODEL SUMMARY")
    print("=" * 60)

    summary_lines = []
    summary_lines.append("=" * 100)
    summary_lines.append("CROSS-MODEL ENRICHMENT SUMMARY (ASD185_EAGLE only)")
    summary_lines.append("=" * 100)
    summary_lines.append("")
    summary_lines.append("Method: Fisher's exact test (one-sided)")
    summary_lines.append("Significance: FDR < 0.05 AND Odds Ratio > 1")
    summary_lines.append("Feature: ASD185_EAGLE (known ASD genes) only")
    summary_lines.append("")

    # Summary table
    summary_lines.append("-" * 100)
    summary_lines.append("SIGNIFICANT CLUSTERS PER MODEL")
    summary_lines.append("-" * 100)
    summary_lines.append("")

    summary_data = []
    for model_name, data in all_model_results.items():
        effective = data['effective_clusters']
        summary_data.append({
            'Model': model_name,
            'ASD185_EAGLE_Significant_Clusters': ', '.join(sorted(effective, key=lambda x: int(x) if x.isdigit() else x)) if effective else '-',
            'N_Significant': len(effective),
        })

    summary_df = pd.DataFrame(summary_data)
    summary_lines.append(summary_df.to_string(index=False))
    summary_lines.append("")

    # Save summary
    summary_file = f"{OUTPUT_DIR}/enrichment_summary_all_models.txt"
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"Cross-model summary saved to: {summary_file}")

    # Also save as CSV
    summary_csv = f"{OUTPUT_DIR}/enrichment_summary_all_models.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary CSV saved to: {summary_csv}")





    print(f"\n{'='*60}")
    print(f"All results saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster_dir', type=str, required=True, help='Name for perturbation dataset')
    parser.add_argument('--split_word', type=str, default='_processed', help='Word for split names based model')
    parser.add_argument('--annot_file', type=Path, required=True, help='Gene annotation file path')
    parser.add_argument('--feature_name', type=str, default='ASD185_EAGLE')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save trained models')

    main(parser)