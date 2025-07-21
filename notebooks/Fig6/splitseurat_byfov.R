---
title: "Split Seurat Objects by FOV"
author: "fsumayyamohamed"
date: "18.07.25"
output:
  html_document:
    toc: true
    toc_depth: 2
params:
  rds_dir:     "data/seurat_objects_withcelltypes"
  output_dir:  "data/seurat_objects_withcelltypes/split_seurats"
---

```{r setup, include=FALSE}
# Load libraries and set chunk options
library(here)
library(Seurat)
knitr::opts_chunk$set(
  echo = TRUE,
  message = FALSE,
  warning = FALSE
)
```

## 1. Define Paths & Mapping

```{r mapping}
# Input directory with combined Seurat RDS files
rds_dir    <- here(params$rds_dir)
# Directory to save split Seurat objects
out_dir    <- here(params$output_dir)
# Create output directory if it doesn't exist
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Mapping of filename -> sample_id -> FOV indices
mapping <- list(
  "seuratObject_21001338_22002462_21002305_with_celltypes.RDS" = list(
    "21001338" = 1:77,
    "22002462" = c(78:249, 276:286),
    "21002305" = 250:275
  ),
  "seuratObject_21002312_22000466_21002306_with_celltypes.RDS" = list(
    "21002306" = 1:38,
    "22000466" = 39:217,
    "21002312" = 218:371
  ),
  "seuratObject_22001589_19001626_17063106_with_celltypes.RDS" = list(
    "22001589" = 1:32,
    "19001626" = 33:180,
    "17063106" = 181:195
  ),
  "seuratObject_22002145_19001625_17063451_with_celltypes.RDS" = list(
    "22002145" = 1:167,
    "19001625" = 168:295,
    "17063451" = 296:350
  ),
  "seuratObject_22003460_22002320_22003431_with_celltypes.RDS" = list(
    "22003431" = c(37:51, 74:83, 116:123, 200:203),
    "22003460" = c(1:27, 52:64, 96:115, 124:178),
    "22002320" = c(28:36, 65:73, 84:95, 179:199)
  )
)
```

## 2. Split and Save by Sample ID

```{r split-save}
for (rds_fn in names(mapping)) {
  message("Reading: ", rds_fn)
  full_path <- file.path(rds_dir, rds_fn)
  so <- readRDS(full_path)

  if (!"fov" %in% colnames(so@meta.data)) {
    stop("Column 'fov' not found in metadata of ", rds_fn)
  }

  # Iterate sample IDs in this mapping
  for (sample_id in names(mapping[[rds_fn]])) {
    fovs <- mapping[[rds_fn]][[sample_id]]
    # Subset by FOV
    sub_so <- subset(so, subset = fov %in% fovs)
    sub_so$sample_id <- sample_id

    # Save to output directory
    out_file <- file.path(out_dir, paste0("seurat_", sample_id, ".RDS"))
    message("  → writing ", out_file,
            " (nCells = ", ncol(sub_so), ")")
    saveRDS(sub_so, file = out_file)
  }
}
```

## 3. Session Information

```{r session-info}
sessionInfo()
