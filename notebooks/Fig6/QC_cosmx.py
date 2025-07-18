#!/usr/bin/env python3
"""
qc_cosmx.py

Quality‑control pipeline for CosMX slides:
  - loads metadata, FOV positions, and expression matrix
  - assigns tissue labels
  - computes per‑FOV and per‑cell summaries
  - filters low‑quality cells/genes
  - generates spatial & histogram/box plots
  - saves updated metadata & summary tables

Usage:
  python qc_cosmx.py \
    --meta metadata.csv \
    --fov fov_positions.csv \
    --counts_dir /path/to/nanostring_dir \
    --outdir qc_results
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt
import seaborn as sns

# reproducibility & style
np.random.seed(42)
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (6, 6)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--meta',      required=True, help="Slide metadata CSV")
    p.add_argument('--fov',       required=True, help="FOV positions CSV")
    p.add_argument('--counts_dir',required=True,
                   help="dir containing nanostring counts, meta & fov files")
    p.add_argument('--outdir',    default='qc_results', help="output directory")
    return p.parse_args()


def load_tables(meta_path, fov_path):
    meta = pd.read_csv(meta_path)
    fov  = pd.read_csv(fov_path)
    return meta, fov


def plot_spatial_cells(meta, outdir):
    plt.figure()
    plt.scatter(meta["CenterX_global_px"], meta["CenterY_global_px"],
                s=meta["Mean.DAPI"]/meta["Mean.DAPI"].max()*200, 
                c='navy', alpha=0.4)
    plt.title("All Cells (size ∝ DAPI intensity)")
    plt.gca().invert_yaxis()
    plt.savefig(outdir / "cells_spatial.png", dpi=150)
    plt.close()


def plot_fov_positions(fov, outdir):
    fig, ax = plt.subplots()
    ax.scatter(fov["x_global_px"], fov["y_global_px"], c='crimson')
    for _, row in fov.iterrows():
        ax.text(row["x_global_px"]*1.01, row["y_global_px"]*1.01,
                str(int(row["fov"])), fontsize=8)
    ax.set_title("FOV centers")
    ax.invert_yaxis()
    fig.savefig(outdir / "fov_positions.png", dpi=150)
    plt.close()


def assign_tissue(fov):
    bins = [(1,16,'NBT004'), (17,38,'NBT005'),
            (39,78,'NBT006'), (79,82,'NBT004'),
            (83,85,'NBT005'), (86,999,'NBT006')]
    fov['tissue'] = 'Unknown'
    for lo, hi, label in bins:
        mask = fov['fov'].between(lo, hi)
        fov.loc[mask, 'tissue'] = label
    return fov


def plot_tissue_fov(fov, outdir):
    fig = sns.scatterplot(data=fov, x='x_global_px', y='y_global_px',
                          hue='tissue', palette='Set2', s=100)
    sns.move_legend(fig, "upper left", bbox_to_anchor=(1,1))
    fig.figure.savefig(outdir / "fov_by_tissue.png", dpi=150, bbox_inches='tight')
    plt.close()


def crosstab_cells(meta, outdir):
    ct = pd.crosstab(meta['fov'], meta['Tissue_ID'])
    ct['cells_per_fov'] = ct.sum(axis=1)
    ct.to_csv(outdir / "cells_per_fov.csv")
    return ct


def plot_cells_per_fov(ct, fov, outdir):
    fov2 = fov.merge(ct['cells_per_fov'], left_on='fov', right_index=True)
    # size by cell count
    plt.figure()
    plt.scatter(fov2.x_global_px, fov2.y_global_px,
                s=fov2.cells_per_fov/5, c='steelblue', alpha=0.6)
    plt.title("Cells per FOV (size ∝ count)")
    plt.gca().invert_yaxis()
    plt.savefig(outdir / "cells_count_spatial.png", dpi=150)
    plt.close()
    # hue by count
    fig = sns.scatterplot(data=fov2, x='x_global_px', y='y_global_px',
                          hue='cells_per_fov', palette='Spectral_r', s=100)
    sns.move_legend(fig, "upper left", bbox_to_anchor=(1,1))
    fig.figure.savefig(outdir / "cells_count_hue.png", dpi=150, bbox_inches='tight')
    plt.close()


def boxplot_cells_by_tissue(fov, outdir):
    plt.figure()
    sns.boxplot(x='tissue', y='cells_per_fov', data=fov,
                palette='tab10')
    sns.despine(offset=10, trim=True)
    plt.title("Cells per FOV by Tissue")
    plt.savefig(outdir / "cells_boxplot_by_tissue.png", dpi=150)
    plt.close()


def load_anndata(counts_dir, meta_fname, fov_fname):
    adata = sq.read.nanostring(
        path=counts_dir,
        counts_file=Path(meta_fname).name.replace("metadata","exprMat"),
        meta_file=Path(meta_fname).name,
        fov_file=Path(fov_fname).name
    )
    return adata


def calculate_qc(adata):
    adata.var['NegPrb'] = adata.var_names.str.startswith("NegPrb")
    sc.pp.calculate_qc_metrics(adata, qc_vars=['NegPrb'], inplace=True)
    return adata


def plot_qc_histograms(adata, outdir):
    fig, axs = plt.subplots(1,4, figsize=(16,4))
    titles = ["Total counts","Unique genes","Counts per FOV","Cell Area"]
    data = [adata.obs.total_counts,
            adata.obs.n_genes_by_counts,
            adata.obs.groupby('fov').total_counts.sum(),
            adata.obs.Area]
    for ax, vals, title in zip(axs, data, titles):
        sns.histplot(vals, kde=False, ax=ax)
        ax.set_title(title)
    fig.savefig(outdir / "histograms_qc.png", dpi=150)
    plt.close()


def filter_cells_and_genes(adata, outdir):
    # drop cells with extreme NegPrb%
    adata = adata[adata.obs.pct_counts_NegPrb <= 5].copy()
    sc.pp.filter_cells(adata, min_genes=25, max_genes=300)
    min_cells = int(adata.n_obs * 0.01)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    # report
    stats = {
        'n_cells': adata.n_obs,
        'n_genes': adata.n_vars
    }
    pd.Series(stats).to_csv(outdir / "filtered_dims.csv")
    return adata


def basic_dimensionality(adata, outdir):
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, svd_solver='arpack')
    sc.pl.pca_variance_ratio(adata, log=True, 
                             show=False)
    plt.savefig(outdir / "pca_variance.png", dpi=150)
    plt.close()
    sc.pp.neighbors(adata, n_pcs=15)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.7)
    sc.pl.umap(adata, color=['leiden','Tissue_ID'], 
               size=5, show=False)
    plt.savefig(outdir / "umap_clusters.png", dpi=150)
    plt.close()


def save_updated_meta(adata, outdir):
    df = adata.obs.copy()
    df.to_csv(outdir / "metadata_updated.csv")


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    meta, fov = load_tables(args.meta, args.fov)
    plot_spatial_cells(meta, outdir)
    plot_fov_positions(fov, outdir)

    fov = assign_tissue(fov)
    plot_tissue_fov(fov, outdir)

    ct = crosstab_cells(meta, outdir)
    plot_cells_per_fov(ct, fov, outdir)
    boxplot_cells_by_tissue(fov, outdir)

    adata = load_anndata(args.counts_dir, args.meta, args.fov)
    adata = calculate_qc(adata)
    plot_qc_histograms(adata, outdir)
    adata = filter_cells_and_genes(adata, outdir)
    basic_dimensionality(adata, outdir)
    save_updated_meta(adata, outdir)

    print("QC pipeline complete — results in:", outdir)


if __name__ == '__main__':
    main()
