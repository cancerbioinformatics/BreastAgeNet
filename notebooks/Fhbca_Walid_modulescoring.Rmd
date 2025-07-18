---
title: "HBCA Module Scores & Spatial Visualization"
author: "fsumayyamohamed"
date: "18.07.25"
output:
  html_document:
    toc: true
    toc_depth: 2
params:
  input_dir:           "data/seurat_objects_withcelltypes/split_seurats"
  gene_lists_script:   "scripts/hbca_gene_lists.R"
  out_summary_csv:     "results/mean_module_scores_by_condition.csv"
  out_bar_png:         "figures/hbca_module_scores_barplot.png"
  spatial_dir:         "figures/spatial_plots"
---

```{r setup, include=FALSE}
# Load libraries and set options
library(here)
library(Seurat)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggrepel)
library(tibble)
knitr::opts_chunk$set(
  echo = TRUE,
  message = FALSE,
  warning = FALSE
)
```

## 1. Load Gene Lists & Data

```{r load-gene-lists}
# Source predefined HBCA gene sets
source(here(params$gene_lists_script))  # defines lasp2_genes, lhs2_genes, bmyo1_genes

# Input directory of Seurat objects
seurat_dir <- here(params$input_dir)
seurat_files <- list.files(seurat_dir, pattern="^seurat_\\d+\\.RDS$", full.names=TRUE)
sample_ids <- gsub("^.*seurat_(\\d+)\\.RDS$", "\\1", seurat_files)

# Define condition mapping per sample
condition_map <- c(
  "21001338" = "<35 BRCA, accelerated ageing",
  "22003431" = "<35 BRCA, accelerated ageing",
  "22002312" = "<35 BRCA, accelerated ageing",
  "22000466" = "<35 BRCA, concordant ageing",
  "22002145" = "<35 BRCA, concordant ageing",
  "22002462" = "<35 BRCA, concordant ageing"
)
```

## 2. Compute Module Scores

```{r compute-scores}
# Initialize storage
all_metadata   <- list()
summary_scores <- list()

# Loop over samples
for (i in seq_along(seurat_files)) {
  file <- seurat_files[i]
  sid  <- sample_ids[i]
  cond <- condition_map[sid]

  message("Processing sample: ", sid, " (", cond, ")")
  so <- readRDS(file)
  so$sample_id <- sid
  so$condition <- cond

  # Compute module scores
  so <- AddModuleScore(so, features = list(lasp2_genes), name = "LASP2")
  so <- AddModuleScore(so, features = list(lhs2_genes), name = "LHS2")
  so <- AddModuleScore(so, features = list(bmyo1_genes), name = "BMYO1")

  # Extract metadata
  md <- so@meta.data %>%
    transmute(
      sample_id,
      condition,
      cell_type,
      LASP2 = LASP21,
      LHS2  = LHS21,
      BMYO1 = BMYO11,
      x = x_slide_mm,
      y = y_slide_mm
    )

  all_metadata[[sid]]   <- md

  # Summarize mean scores per cell_type and condition
  summary_scores[[sid]] <- md %>%
    group_by(cell_type, condition) %>%
    summarise(
      LASP2 = mean(LASP2, na.rm=TRUE),
      LHS2  = mean(LHS2, na.rm=TRUE),
      BMYO1 = mean(BMYO1, na.rm=TRUE),
      .groups = "drop"
    ) %>%
    mutate(sample_id = sid)
}
```

## 3. Save Summary Scores & Barplot

```{r save-summary}
# Combine and save CSV
summary_df <- bind_rows(summary_scores)
write.csv(summary_df, here(params$out_summary_csv), row.names=FALSE)

# Prepare long format for plotting
bar_df <- summary_df %>%
  pivot_longer(
    cols = c(LASP2, LHS2, BMYO1),
    names_to = "module",
    values_to = "score"
  )

# Barplot
p_bar <- ggplot(bar_df, aes(x=cell_type, y=score, fill=condition)) +
  geom_bar(stat="identity", position="dodge") +
  facet_wrap(~ module, scales="free_y") +
  theme_minimal(base_size=14) +
  labs(
    title = "Mean HBCA Module Scores by Cell Type & Condition",
    x = "Cell Type",
    y = "Mean Module Score",
    fill = "Condition"
  ) +
  theme(axis.text.x = element_text(angle=45, hjust=1))

# Save barplot
ggsave(here(params$out_bar_png), p_bar,
       width = 10, height = 6, units = "in", dpi = 300)

p_bar
```

## 4. Spatial Feature Plots

```{r spatial-plots}
# Create output directory
dir.create(here(params$spatial_dir), showWarnings = FALSE, recursive = TRUE)

# Loop through each sample and module
for (sid in names(all_metadata)) {
  md <- all_metadata[[sid]]
  for (feature in c("LASP2", "LHS2", "BMYO1")) {
    p <- ggplot(md, aes(x=x, y=y, color=.data[[feature]])) +
      geom_point(size=0.4) +
      coord_fixed() +
      scale_color_viridis_c(option="plasma") +
      theme_void(base_size=12) +
      labs(
        title = paste0("Spatial: ", feature, " in sample ", sid),
        color = feature
      )
    fn <- here(params$spatial_dir, paste0("spatial_", feature, "_", sid, ".pdf"))
    ggsave(fn, p, width=6, height=5, units="in", dpi=300)
  }
}
```

## 5. Session Information

```{r session-info}
sessionInfo()
