---
title: "UMAP and Grouped Cell Typing Analysis"
author: "fsumayyamohamed"
date: "18.07.25"
output:
  html_document:
    toc: true
    toc_depth: 2
params:
  seurat_rds:         "data/seurat_objects_withcelltypes/seuratObject_query_with_celltypes.RDS"
  out_umap_orig_png:  "figures/umap_original_clusters.png"
  out_umap_group_png: "figures/umap_grouped_celltypes.png"
  markers_csv:        "results/group_markers_late.csv"
---

```{r setup, include=FALSE}
library(here)
library(Seurat)
library(dplyr)
library(ggplot2)
knitr::opts_chunk$set(
  echo = TRUE,
  message = FALSE,
  warning = FALSE
)
```

## 1. Load Seurat Object

```{r load-data}
seu <- readRDS(here(params$seurat_rds))
print(colnames(seu@meta.data))
unique_vals <- unique(seu@meta.data$cell_type)
message("Unique cell_type values: ", paste(unique_vals, collapse=", "))
```

## 2. Preprocessing & Original UMAP

```{r preprocess-umap}
# Normalize, select HVGs, scale, PCA, UMAP
seu <- NormalizeData(seu)
seu <- FindVariableFeatures(seu, selection.method="vst", nfeatures=2000)
seu <- ScaleData(seu, features = VariableFeatures(seu))
seu <- RunPCA(seu, features = VariableFeatures(seu), npcs = 30, verbose = FALSE)
seu <- RunUMAP(seu, dims = 1:20, verbose = FALSE)

# Plot original assignments
p_orig <- DimPlot(
  seu,
  reduction = "umap",
  group.by  = "cell_type",
  label     = TRUE,
  pt.size   = 0.5
) +
  ggtitle("UMAP: Original cell_type assignments") +
  theme_minimal(base_size = 14)

# Save original UMAP
ggsave(
  here(params$out_umap_orig_png),
  plot = p_orig,
  width = 8, height = 6,
  units = "in", dpi = 300
)

p_orig
```

## 3. Recode Grouped Cell Types

```{r recode-group}
# Define mapping to grouped cell types
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
# Add grouped metadata
seu <- AddMetaData(
  seu,
  metadata = recode(
    seu@meta.data$cell_type,
    !!!group_map,
    .default = "Other"
  ),
  col.name = "cell_type_group"
)
```

## 4. Find Markers for Grouped Types

```{r find-markers}
# Set Idents to new grouping
Idents(seu) <- "cell_type_group"
# Find markers (only positive logFC)
group_markers <- FindAllMarkers(
  seu,
  assay         = "RNA",
  slot          = "data",
  only.pos      = TRUE,
  logfc.threshold = 0.25,
  min.pct       = 0.10
)
# Save markers
write.csv(
  group_markers,
  here(params$markers_csv),
  row.names = FALSE
)
```

## 5. UMAP of Grouped Cell Types

```{r plot-grouped-umap}
# Plot grouped UMAP
p_grp <- DimPlot(
  seu,
  reduction = "umap",
  group.by  = "cell_type_group",
  label     = TRUE,
  pt.size   = 0.5
) +
  ggtitle("UMAP: Grouped cell types") +
  theme_minimal(base_size = 14)

# Save grouped UMAP
ggsave(
  here(params$out_umap_group_png),
  plot = p_grp,
  width = 8, height = 6,
  units = "in", dpi = 300
)

p_grp
```

## 6. Session Information

```{r session-info}
sessionInfo()
