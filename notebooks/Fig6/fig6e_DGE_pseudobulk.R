---
title: "Pseudobulk DGE: <35 BRCA Accelerated vs Concordant"
author: "fsumayyamohamed"
date: "18.07.25"
output:
  html_document:
    toc: true
    toc_depth: 2
params:
  input_dir:    "data/seurat_objects_withcelltypes/split_seurats"
  out_csv:      "results/dge_results_brca_acc_vs_concordant.csv"
  out_volcano:  "figures/volcano_plot_brca_acc_vs_concordant.pdf"
---

```{r setup, include=FALSE}
# Load libraries and set chunk options
library(here)
library(Seurat)
library(dplyr)
library(tidyr)
library(Matrix)
library(edgeR)
library(ggplot2)
library(ggrepel)
library(tibble)
knitr::opts_chunk$set(
  echo = TRUE,
  message = FALSE,
  warning = FALSE
)
```

## 1. Define Inputs & Mappings

```{r params}
# Directory containing seurat_<sample_id>.RDS
data_dir <- here(params$input_dir)

# Condition mapping per sample
condition_map <- c(
  "22003431" = "<35 BRCA, accelerated ageing",
  "22003460" = "other",
  "22003420" = "other",
  "21001338" = "<35 BRCA, accelerated ageing",
  "22002462" = "<35 BRCA, concordant ageing",
  "21002305" = ">55 BRCA, concordant ageing",
  "21002306" = ">55 BRCA, concordant ageing",
  "22000466" = "<35 BRCA, concordant ageing",
  "22002312" = "<35 BRCA, accelerated ageing",
  "22002145" = "<35 BRCA, concordant ageing",
  "17063451" = ">55 NON BRCA",
  "19001625" = "<35 NON BRCA",
  "19001626" = "<35 NON BRCA",
  "22001589" = ">55 BRCA, concordant ageing",
  "17063106" = ">55 NON BRCA",
  "21002312" = "<35 BRCA, accelerated ageing",
  "22002320" = "other"
)

# Slide/batch mapping per sample
slide_map <- c(
  "22003431" = "Slide1", "22003460" = "Slide1", "22003420" = "Slide1",
  "21001338" = "Slide3", "22002462" = "Slide3", "21002305" = "Slide3",
  "21002306" = "Slide4", "22000466" = "Slide4", "22002312" = "Slide4",
  "22002145" = "Slide5", "17063451" = "Slide5", "19001625" = "Slide5",
  "19001626" = "Slide6", "22001589" = "Slide6", "17063106" = "Slide6",
  "21002312" = "Slide4", "22002320" = "Slide1"
)
```

## 2. Load Seurat Objects & Filter

```{r load-seurats}
files <- list.files(data_dir, pattern = "^seurat_\\d+\\.RDS$", full.names = TRUE)
samples <- lapply(files, readRDS)
names(samples) <- gsub("^.*seurat_(\\d+)\\.RDS$", "\\1", basename(files))

# Drop any Slide2 if present
drop <- names(slide_map)[slide_map == "Slide2"]
samples <- samples[! names(samples) %in% drop]
condition_map <- condition_map[! names(condition_map) %in% drop]
slide_map     <- slide_map[! names(slide_map) %in% drop]
```

## 3. Build Pseudobulk Count Matrix

```{r pseudobulk}
# Extract counts and metadata
data_list <- lapply(samples, function(so) {
  list(
    counts = GetAssayData(so, assay = "RNA", slot = "counts"),
    meta   = so@meta.data %>% select(sample_id, cell_type)
  )
})
# Sum counts by sample_celltype groups
pb_list <- lapply(data_list, function(x) {
  mat    <- x$counts
n   groups <- paste(x$meta$sample_id, x$meta$cell_type, sep = "_")
  ug     <- unique(groups)
  do.call(cbind, lapply(ug, function(g) rowSums(mat[, groups == g, drop = FALSE]))) %>%
    `colnames<-`(ug)
})
pb_counts <- do.call(cbind, pb_list)

# Build lib_info dataframe\llib_info <- tibble(
  sample_celltype = colnames(pb_counts)
) %>%
  mutate(
    sample_id = sub("_.*$", "", sample_celltype),
    cell_type = sub("^[^_]+_", "", sample_celltype),
    condition = condition_map[sample_id],
    batch     = slide_map[sample_id]
  ) %>%
  stopifnot(!anyNA(.$condition), !anyNA(.$batch)) %>%
  mutate(
    condition = factor(condition),
    batch     = factor(batch)
  )

# Reorder lib_info to match columns
lib_info <- lib_info[match(colnames(pb_counts), lib_info$sample_celltype), ]
```

## 4. Design & Fit edgeR Model

```{r edgeR-model}
# Create design matrix
design <- model.matrix(~ 0 + condition + batch, data = lib_info)

# Build DGEList and fit
dge <- DGEList(counts = pb_counts, samples = lib_info)
dge <- calcNormFactors(dge)
dge <- estimateDisp(dge, design)
fit <- glmQLFit(dge, design)

# Contrast: accelerated vs concordant ageing
cnames   <- colnames(design)
contrast <- numeric(length(cnames))
contrast[which(cnames == "condition<35 BRCA, accelerated ageing")] <-  1
contrast[which(cnames == "condition<35 BRCA, concordant ageing")]  <- -1

qlf <- glmQLFTest(fit, contrast = contrast)
```

## 5. Extract & Save DGE Results

```{r extract-results}
res <- topTags(qlf, n = Inf)$table %>%
  rownames_to_column("gene") %>%
  mutate(
    regulation = case_when(
      FDR < 0.05 & logFC >  0.5  ~ "Up",
      FDR < 0.05 & logFC < -0.5  ~ "Down",
      TRUE                       ~ "NS"
    )
  )
# Save results
write.csv(res, here(params$out_csv), row.names = FALSE)
print(table(res$regulation))
```

## 6. Volcano Plot

```{r volcano-plot}
p <- ggplot(res, aes(x=logFC, y=-log10(FDR), color=regulation)) +
  geom_point(alpha=0.6, size=1) +
  scale_color_manual(values = c(Up="blue", Down="red", NS="grey70"), name="Regulation") +
  geom_vline(xintercept = c(-0.5,0.5), linetype="dashed") +
  geom_hline(yintercept = -log10(0.05), linetype="dashed") +
  geom_text_repel(
    data = filter(res, regulation %in% c("Up","Down")),
    aes(label=gene), size=3, max.overlaps=10
  ) +
  theme_minimal(base_size=14) +
  labs(
    title = "<35 BRCA Accelerated vs Concordant Ageing",
    x = expression(Log[2]~FC),
    y = expression(-Log[10]~FDR)
  ) +
  theme(legend.position="bottom")
# Save volcano
ggsave(here(params$out_volcano), plot = p, width=8, height=6, units="in", dpi=300)
p
```

## 7. Session Info

```{r session-info}
sessionInfo()
