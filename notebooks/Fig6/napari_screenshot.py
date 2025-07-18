#!/usr/bin/env python3
# apply_custom_colors.py
# Harmonised UMAP-inspired colours per fine cell type for Gemini/Napari
# Author: fsumayyamohamed
# Date: 2025-07-18
# License: MIT
# Description: Apply custom color mapping to cell types in Gemini, capture screenshot.

import os
import json
import argparse
import logging
from pathlib import Path

import imageio
import gem

def load_color_map(path=None):
    """
    Load color map from JSON file or use built-in defaults.
    """
    default_colors = {
        # Luminal-like
        "Cancer.LumA.SC": "#3d7822",
        "Cancer.Cycling": "#3d7822",
        # Luminal
        "Mature.Luminal": "#8fbc8f",
        "Luminal.Progenitors": "#3d7822",
        # Myoepithelial
        "Myoepithelial": "#bc255c",
        # Endothelial
        "Endothelial.ACKR1": "#cc7d00",
        "Endothelial.CXCL12": "#cc7d00",
        "Endothelial.Lymphatic.LYVE1": "#cc7d00",
        "Endothelial.RGS5": "#cc7d00",
        # Fibroblasts
        "CAFs.MSC.iCAF-like.s1": "#193773",
        "CAFs.MSC.iCAF-like.s2": "#193773",
        "CAFs.Transitioning.s3": "#193773",
        "CAFs.myCAF.like.s4": "#193773",
        "CAFs.myCAF.like.s5": "#193773",
        "undefined": "#193773",
        # Myeloid
        "Myeloid_c10_Macrophage_1_EGR1": "#708090",
        "Cycling_Myeloid": "#bc255c",
        "Myeloid_c11_cDC2_CD1C": "#708090",
        "Myeloid_c2_LAM2_APOE": "#708090",
        "Myeloid_c3_cDC1_CLEC9A": "#708090",
        "Myeloid_c4_DCs_pDC_IRF7": "#708090",
        # T cells
        "T_cells_c11_MKI67": "#87CEEB",
        # B/Plasma
        "B.cells.Memory": "#4393c3",
        "Plasmablasts": "#4393c3",
        # PVL
        "PVL.Immature.s1": "#f9d057",
        "PVL_Immature.s2": "#f9d057",
        "PVL.Differentiated.s3": "#f9d057"
    }
    if path:
        with open(path) as f:
            return json.load(f)
    return default_colors


def apply_colors(celltype_col, color_map):
    """
    Apply the color map to the Gemini viewer based on metadata column.
    """
    gem.color_cells(celltype_col, color=color_map)


def save_screenshot(outpath, dpi=800):
    """
    Capture the current Gemini canvas and save to image file.
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(outpath), mode='I', fps=1) as writer:
        img = gem.viewer.screenshot(canvas_only=True)
        writer.append_data(img)
    logging.info(f"Screenshot saved to: {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply custom cell-type colors in Gemini and save screenshot"
    )
    parser.add_argument('-c', '--celltype_col', required=True,
                        help="Metadata column for cell-type labels")
    parser.add_argument('-m', '--colors_json', default=None,
                        help="Optional path to JSON file defining colors")
    parser.add_argument('-o', '--outdir', default='figures',
                        help="Directory to save outputs")
    parser.add_argument('-n', '--name', default='umap_colors.png',
                        help="Filename for the saved screenshot")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s] %(message)s')
    logging.info("Loading color map...")
    cmap = load_color_map(args.colors_json)
    
    logging.info("Applying colors to column '%s'...", args.celltype_col)
    apply_colors(args.celltype_col, cmap)

    outpath = Path(args.outdir) / args.name
    logging.info("Saving screenshot to %s...", outpath)
    save_screenshot(outpath)

    logging.info("Done.")

if __name__ == '__main__':
    main()
