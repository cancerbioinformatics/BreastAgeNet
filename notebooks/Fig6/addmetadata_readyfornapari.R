#!/usr/bin/env Rscript
# cleanup_seurat_metadata.R
# Clean Seurat object by removing unwanted metadata slots, extract key metadata, and optionally restore specific columns
# Author: fsumayyamohamed
# Date: 2025-07-18
# License: MIT
# Description: General script to strip specified patterns from Seurat object slots,
#              extract selected metadata columns into CSV, and optionally restore columns.

# ==== I) Command-line arguments ===========================================
suppressPackageStartupMessages(library(optparse))

option_list <- list(
  make_option(c("-i","--input"), type="character", help="Path to input Seurat RDS", metavar="file"),
  make_option(c("-o","--output_meta"), type="character", help="Path to output metadata CSV", metavar="file"),
  make_option(c("-s","--save_seurat"), type="character", default=NULL,
              help="Optional path to save cleaned Seurat RDS", metavar="file"),
  make_option(c("-r","--restore_from"), type="character", default=NULL,
              help="Optional path to original Seurat RDS for restoring columns", metavar="file"),
  make_option(c("-n","--run_name"), type="character", required=TRUE,
              help="Value of Run_Tissue_name to filter cells by", metavar="string"),
  make_option(c("-c","--columns"), type="character", default=NULL,
              help="Comma-separated metadata columns to extract", metavar="cols"),
  make_option(c("-p","--remove_patterns"), type="character", default="backup",
              help="Comma-separated regex patterns for slots/metadata columns to remove [default: 'backup']", metavar="patterns")
)
opt <- parse_args(OptionParser(option_list=option_list))

# Ensure required args
if (is.null(opt$input) || is.null(opt$output_meta) || is.null(opt$columns)) {
  print_help(OptionParser(option_list=option_list))
  stop("--input, --output_meta, and --columns are required.", call.=FALSE)
}

# ==== II) Load libraries ==================================================
suppressPackageStartupMessages({
  library(Seurat)
  library(dplyr)
})

# ==== III) Read Seurat object =============================================
message("Reading Seurat object: ", opt$input)
seu <- readRDS(opt$input)

# ==== IV) Define removal patterns ========================================
patterns <- strsplit(opt$remove_patterns, ",")[[1]]
# helper to test if a name matches any pattern
matches_pattern <- function(names_vec, patterns) {
  keep_idx <- rep(FALSE, length(names_vec))
  for (pat in patterns) {
    keep_idx <- keep_idx | grepl(pat, names_vec)
  }
  return(keep_idx)
}

# ==== V) Remove slots and metadata =======================================
message("Removing slots/columns matching patterns: ", toString(patterns))
# metadata columns
md_names <- colnames(seu@meta.data)
drop_md   <- md_names[matches_pattern(md_names, patterns)]
if (length(drop_md)>0) {
  seu@meta.data <- seu@meta.data[, !md_names %in% drop_md]
  message("  dropped metadata columns: ", toString(drop_md))
} else message("  no metadata columns to drop")
# assays
assays    <- names(seu@assays)
drop_ass  <- assays[matches_pattern(assays, patterns)]
seu@assays     <- seu@assays[! assays %in% drop_ass]
# reductions
reds      <- names(seu@reductions)
drop_red  <- reds[matches_pattern(reds, patterns)]
seu@reductions <- seu@reductions[! reds %in% drop_red]
# graphs
grs       <- names(seu@graphs)
drop_gr   <- grs[matches_pattern(grs, patterns)]
seu@graphs     <- seu@graphs[! grs %in% drop_gr]
# commands
cmds      <- names(seu@commands)
drop_cmd  <- cmds[matches_pattern(cmds, patterns)]
seu@commands   <- seu@commands[! cmds %in% drop_cmd]

# ==== VI) Extract specified metadata ======================================
ct_cols <- strsplit(opt$columns, ",")[[1]]
message("Filtering cells with Run_Tissue_name == ", opt$run_name)
meta_df <- seu@meta.data %>%
  filter(Run_Tissue_name == opt$run_name) %>%
  select(all_of(ct_cols)) %>%
  mutate(cell_ID = rownames(.)) %>%
  { rownames(.) <- NULL; . } %>%
  relocate(cell_ID)

# write metadata CSV
message("Writing metadata CSV: ", opt$output_meta)
write.csv(meta_df, opt$output_meta, row.names=FALSE, quote=FALSE)

# ==== VII) Optionally restore columns ====================================
if (!is.null(opt$restore_from)) {
  message("Restoring columns from: ", opt$restore_from)
  orig <- readRDS(opt$restore_from)
  all_orig <- orig@meta.data
  for (pat in patterns) {
    cols_to_restore <- grep(pat, colnames(all_orig), value=TRUE)
    for (col in cols_to_restore) {
      seu@meta.data[[col]] <- all_orig[[col]]
      message("  restored column: ", col)
    }
  }
}

# ==== VIII) Optionally save cleaned Seurat object ========================
if (!is.null(opt$save_seurat)) {
  message("Saving cleaned Seurat object to: ", opt$save_seurat)
  saveRDS(seu, opt$save_seurat)
}

# ==== IX) Session info ==================================================
message("Done. Session info:")
print(sessionInfo())
