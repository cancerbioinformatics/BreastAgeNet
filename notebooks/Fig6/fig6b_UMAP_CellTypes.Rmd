---
title: "Grouping & Marker Detection Analysis"
author: "fsumayyamohamed"
date: "18.07.25"
output:
  html_document:
    toc: true
    toc_depth: 2
params:
  seurat_rds:      "data/seuratObject.rds"
  out_fig_png:     "figures/umap_grouped_celltypes.png"
  out_fig_pdf:     "figures/umap_grouped_celltypes.pdf"
  markers_csv:     "figures/group_markers.csv"
---

```{r setup, include=FALSE}
# Load libraries and set chunk options
library(here)
library(dplyr)
library(Seurat)
library(ComplexHeatmap)
library(circlize)
library(ggplot2)
knitr::opts_chunk$set(
  echo = TRUE,
  message = FALSE,
  warning = FALSE
)
```

## 1. Load Data

```{r load-data}
# Read Seurat object from relative path
seu <- readRDS(here(params$seurat_rds))
```

## 2. Recode Metadata into Grouped Cell Types

```{r recode-metadata}
# Define mapping from original clusters to grouped types
group_map <- c(
  "Mature.Luminal"                = "LHS",
  "Luminal.Progenitors"           = "LASP",
  "Myoepithelial"                 = "Myoepithelial",
  "Cancer.LumA.SC"                = "LASP",
  "Cancer.Cycling"                = "LASP",
  "Endothelial.ACKR1"             = "Endothelial",
  "Endothelial.CXCL12"            = "Endothelial",
  "Endothelial.Lymphatic.LYVE1"   = "Endothelial",
  "Endothelial.RGS5"              = "Endothelial",
  "CAFs.MSC.iCAF-like.s1"         = "Fibroblasts",
  "CAFs.MSC.iCAF-like.s2"         = "Fibroblasts",
  "CAFs.Transitioning.s3"         = "Fibroblasts",
  "CAFs.myCAF.like.s4"            = "Fibroblasts",
  "CAFs.myCAF.like.s5"            = "Fibroblasts",
  "undefined"                     = "Fibroblasts",
  "Myeloid_c10_Macrophage_1_EGR1" = "Myeloid",
  "Cycling_Myeloid"               = "Myeloid",
  "Myeloid_c11_cDC2_CD1C"         = "Myeloid",
  "Myeloid_c2_LAM2_APOE"          = "Myeloid",
  "Myeloid_c3_cDC1_CLEC9A"        = "Myeloid",
  "Myeloid_c4_DCs_pDC_IRF7"       = "Myeloid",
  "T_cells_c11_MKI67"             = "T Cells",
  "B.cells.Memory"                = "B Cells / Plasma",
  "Plasmablasts"                  = "B Cells / Plasma",
  "PVL.Immature.s1"               = "Periovascular-like",
  "PVL_Immature.s2"               = "Periovascular-like",
  "PVL.Differentiated.s3"         = "Periovascular-like"
)
# Add grouped cell type metadata
seu <- AddMetaData(
  seu,
  metadata = recode(
    seu@meta.data$RNA_fathima_Cell.Typing.InSituType.1_1_clusters,
    !!!group_map,
    .default = "Other"
  ),
  col.name = "cell_type_group"
)
```

## 3. Set Identities and Find Markers

```{r markers}
# Use the new grouping as identities
Idents(seu) <- seu@meta.data$cell_type_group

# Find cluster markers (only positive log-fold changes)
group_markers <- FindAllMarkers(
  seu,
  assay            = "RNA",
  slot             = "data",
  only.pos         = TRUE,
  logfc.threshold  = 0.25,
  min.pct          = 0.10
)

# Save markers to CSV
write.csv(
  group_markers,
  here(params$markers_csv),
  row.names = FALSE
)
```

## 4. Dimensionality Reduction & UMAP Visualization

```{r umap-plot}
# Normalize, find variable features, scale, PCA, and UMAP
seu <- NormalizeData(seu)
seu <- FindVariableFeatures(seu, selection.method = "vst", nfeatures = 2000)
seu <- ScaleData(seu, features = VariableFeatures(seu))
seu <- RunPCA(seu, features = VariableFeatures(seu), npcs = 30, verbose = FALSE)
seu <- RunUMAP(seu, dims = 1:20, verbose = FALSE)

# Define custom colors for each group
my_cols <- c(
  "LASP"               = "#3d7822",
  "LHS"                = "#8fbc8f",
  "Myoepithelial"      = "#bc255c",
  "Endothelial"        = "#cc7d00",
  "Periovascular-like" = "#f9d057",
  "B Cells / Plasma"   = "#4393c3",
  "Fibroblasts"        = "#193773",
  "Myeloid"            = "#708090",
  "T Cells"            = "#87CEEB"
)

# Plot UMAP with labels and custom palette
Idents(seu) <- "cell_type_group"
p_umap <- DimPlot(
  seu,
  reduction = "umap",
  cols      = my_cols,
  label     = TRUE,
  label.size= 4,
  pt.size   = 0.5
) +
  ggtitle("UMAP of Grouped Cell Types") +
  theme_minimal(base_size = 14)

# Save as PNG
ggsave(
  here(params$out_fig_png),
  plot  = p_umap,
  width = 8, height = 6,
  units = "in", dpi = 300,
  bg    = "white"
)

# Save as PDF
ggsave(
  here(params$out_fig_pdf),
  plot  = p_umap,
  width = 8, height = 6,
  units = "in", dpi = 300,
  bg    = "white"
)

# Display plot
p_umap
```

## 5. Session Information

```{r session-info}
# Record session info for reproducibility
sessionInfo()
