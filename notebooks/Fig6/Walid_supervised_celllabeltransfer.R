---
title: "Cell Type Label Transfer using Seurat"
author: "fsumayyamohamed"
date: "18.07.25"
output:
  html_document:
    toc: true
    toc_depth: 2
params:
  query_rds:        "data/seurat_objects_withcelltypes/combined/seuratObject_query.RDS"
  ref_h5ad:         "data/Whaled_celltyping/Whaled_reference.h5ad"
  ref_h5seurat:     "data/Whaled_celltyping/Whaled_reference.h5seurat"
  output_rds:       "data/Whaled_celltyping/seurat_transferred_labels.RDS"
---

```{r setup, include=FALSE}
# Load libraries and set chunk options
library(here)
library(Seurat)
library(dplyr)
knitr::opts_chunk$set(
  echo = TRUE,
  message = FALSE,
  warning = FALSE
)
```

## 1. Parameters & Paths

```{r params}
# Paths to files (adjust if needed)
query_rds    <- here(params$query_rds)    # Original Seurat object (query)
ref_h5ad     <- here(params$ref_h5ad)     # Reference AnnData file (.h5ad)
ref_h5seurat <- here(params$ref_h5seurat) # Converted Seurat reference (.h5seurat)
output_rds   <- here(params$output_rds)   # Path to save updated Seurat object
```

## 2. Load Required Packages

```{r libraries}
# Core libraries
library(Seurat)
library(dplyr)
# For conversion from .h5ad
if (!requireNamespace("SeuratDisk", quietly = TRUE)) {
  install.packages("SeuratDisk")
}
library(SeuratDisk)
if (!requireNamespace("zellkonverter", quietly = TRUE)) {
  BiocManager::install("zellkonverter")
}
library(zellkonverter)
```

## 3. Read Query and Prepare Reference

```{r load-data}
# 3.1 Read query Seurat object
sem <- readRDS(query_rds)

# 3.2 Convert .h5ad to .h5seurat if needed
if (!file.exists(ref_h5seurat)) {
  message("Converting reference from .h5ad to .h5seurat...")
  Convert(
    input     = ref_h5ad,
    dest      = ref_h5seurat,
    source    = "anndata",
    overwrite = TRUE
  )
}

# 3.3 Load converted Seurat reference
ref <- LoadH5Seurat(ref_h5seurat)
```

## 4. Preprocess Reference

```{r preprocess-ref}
ref <- NormalizeData(ref, verbose = FALSE)
ref <- FindVariableFeatures(ref, selection.method = "vst", verbose = FALSE)
ref <- ScaleData(ref, features = VariableFeatures(ref), verbose = FALSE)
ref <- RunPCA(ref, features = VariableFeatures(ref), verbose = FALSE)
```

## 5. Preprocess Query

```{r preprocess-query}
sem <- NormalizeData(sem, verbose = FALSE)
sem <- FindVariableFeatures(sem, selection.method = "vst", verbose = FALSE)
sem <- ScaleData(sem, features = VariableFeatures(sem), verbose = FALSE)
sem <- RunPCA(sem, features = VariableFeatures(sem), verbose = FALSE)
```

## 6. Find Transfer Anchors

```{r anchors}
anchors <- FindTransferAnchors(
  reference = ref,
  query     = sem,
  dims      = 1:30
)
```

## 7. Transfer Cell Type Labels

```{r transfer}
# Ensure reference has 'cell_type' metadata column
if (!"cell_type" %in% colnames(ref@meta.data)) {
  stop("Reference object missing 'cell_type' metadata column")
}

predictions <- TransferData(
  anchorset = anchors,
  refdata   = ref@meta.data$cell_type,
  dims      = 1:30
)
# Add transferred labels
sem <- AddMetaData(sem, metadata = predictions)
```

## 8. (Optional) QC Plots

```{r visualize}
# Violin of maximum prediction score
VlnPlot(sem, features = "prediction.score.max", group.by = "predicted.id", pt.size = 0)

# UMAP colored by predicted labels
sem <- RunUMAP(sem, dims = 1:30, verbose = FALSE)
DimPlot(sem, group.by = "predicted.id", label = TRUE)
```

## 9. Save Updated Seurat Object

```{r save-object}
saveRDS(sem, file = output_rds)
message("Updated Seurat object saved to: ", output_rds)
```

## 10. Session Information

```{r session-info}
sessionInfo()
