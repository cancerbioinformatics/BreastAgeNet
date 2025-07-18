---
title: "Stromal Pseudobulk DGE: <35 BRCA Accelerated vs Concordant"
author: "fsumayyamohamed"
date: "18.07.25"
output:
  html_document:
    toc: true
    toc_depth: 2
params:
  input_dir:    "data/seurat_objects_withcelltypes/split_seurats"
  out_csv:      "results/dge_stromal_acc_vs_concordant.csv"
  out_volcano:  "figures/volcano_stromal_acc_vs_concordant.pdf"
---

```{r setup, include=FALSE}
# Load required libraries
library(here)
library(Seurat)
library(dplyr)
library(tidyr)
library(edgeR)
library(ggplot2)
library(ggrepel)
library(tibble)
knitr::opts_chunk$set(echo=TRUE, message=FALSE, warning=FALSE)
```

## 1. Read Seurat Objects

```{r load-seurats}
# Read all seurat_<sample_id>.RDS
data_dir <- here(params$input_dir)
fns <- list.files(data_dir, "^seurat_\\d+\\.RDS$", full.names=TRUE)
samples <- lapply(fns, readRDS)
names(samples) <- gsub("^.*seurat_(\\d+)\\.RDS$", "\\1", fns)
```

## 2. Define Sample Mappings

```{r maps}
# Condition and slide/batch maps
condition_map <- c(
  "22003431" = "<35 BRCA, accelerated ageing",
  "22003460" = "other",
  "22002320" = "other",
  "21001338" = "<35 BRCA, accelerated ageing",
  "22002462" = "<35 BRCA, concordant ageing",
  "21002305" = ">55 BRCA, concordant ageing",
  "21002306" = ">55 BRCA, concordant ageing",
  "22000466" = "<35 BRCA, concordant ageing",
  "21002312" = "<35 BRCA, accelerated ageing",
  "22002145" = "<35 BRCA, concordant ageing",
  "17063451" = ">55 NON BRCA",
  "19001625" = "<35 NON BRCA",
  "19001626" = "<35 NON BRCA",
  "22001589" = ">55 BRCA, concordant ageing",
  "17063106" = ">55 NON BRCA"
)
slide_map <- c(
  "22003431" = "Slide1", "22003460" = "Slide1", "22002320" = "Slide1",
  "21001338" = "Slide3", "22002462" = "Slide3", "21002305" = "Slide3",
  "21002306" = "Slide4", "22000466" = "Slide4", "21002312" = "Slide4",
  "22002145" = "Slide5", "17063451" = "Slide5", "19001625" = "Slide5",
  "19001626" = "Slide6", "22001589" = "Slide6", "17063106" = "Slide6"
)
# Drop Slide2 if present
drop <- names(slide_map)[slide_map == "Slide2"]
samples <- samples[! names(samples) %in% drop]
condition_map <- condition_map[! names(condition_map) %in% drop]
slide_map     <- slide_map[! names(slide_map)     %in% drop]
```

## 3. Build Pseudobulk by Compartment

```{r pseudobulk-comp}
# Extract counts & metadata
all_pb <- lapply(samples, function(so) {
  list(counts = GetAssayData(so, assay="RNA", slot="counts"),
       meta   = so@meta.data %>% select(sample_id, cell_type))
})
# sum counts by sample_celltype
pb_list <- lapply(all_pb, function(x) {
  mat <- x$counts
  groups <- paste(x$meta$sample_id, x$meta$cell_type, sep="_")
  ug <- unique(groups)
  do.call(cbind, lapply(ug, function(g) rowSums(mat[,groups==g,drop=FALSE]))) %>%
    `colnames<-`(ug)
})
pb_counts <- do.call(cbind, pb_list)

# Build lib_info
lib_info <- tibble(sample_celltype = colnames(pb_counts)) %>%
  separate(sample_celltype, into=c("sample_id","cell_type"), sep="_") %>%
  mutate(
    condition = condition_map[sample_id],
    batch     = slide_map[sample_id],
    compartment = ifelse(
      cell_type %in% c("Mature Luminal","Luminal secretory",
                       "Luminal Progenitors","Myoepithelial"),
      "Epithelial","Stromal"
    )
  ) %>%
  filter(!is.na(condition), !is.na(batch), compartment=="Stromal")
# Subset counts for stromal only
pb_str <- pb_counts[, lib_info$sample_celltype]
```

## 4. DGE with edgeR

```{r edgeR-stromal}
# Build DGEList
dge_str <- DGEList(counts=pb_str, samples=lib_info)
dge_str <- calcNormFactors(dge_str)
# Design: accelerated vs concordant, with batch
conds <- c("<35 BRCA, accelerated ageing","<35 BRCA, concordant ageing")
lib_info <- lib_info %>% mutate(condition=factor(condition, levels=conds),
                                batch=factor(batch))
design <- model.matrix(~0+condition+batch, data=lib_info)
dge_str <- estimateDisp(dge_str, design)
fit_str <- glmQLFit(dge_str, design)
# Contrast accelerated - concordant
contrast <- c(1,-1, rep(0, ncol(design)-2))
qlf_str <- glmQLFTest(fit_str, contrast=contrast)
res_str <- topTags(qlf_str, n=Inf)$table %>%
  rownames_to_column("gene") %>%
  mutate(sig = FDR < 0.05)
# Save results
write.csv(res_str, here(params$out_csv), row.names=FALSE)
print(table(res_str$sig))
```

## 5. Volcano Plot

```{r volcano}
p <- ggplot(res_str, aes(x=logFC, y=-log10(FDR), color=sig)) +
  geom_point(alpha=0.6, size=1.5) +
  scale_color_manual(values=c("FDR < 0.05"="#b2182b","NS"="grey70")) +
  geom_vline(xintercept=c(-1,1), linetype="dashed") +
  geom_hline(yintercept=-log10(0.05), linetype="dashed") +
  geom_text_repel(data=subset(res_str, sig), aes(label=gene), size=3, max.overlaps=10) +
  theme_minimal(base_size=14) +
  labs(title="Stromal: <35 BRCA accelerated vs concordant ageing",
       x=expression(Log[2]~FC), y=expression(-Log[10]~FDR), color="") +
  theme(legend.position="bottom")
# Save plot
ggsave(here(params$out_volcano), plot=p, width=8, height=6, units="in", dpi=300)
p
```

## 6. Session Info

```{r session-info}
sessionInfo()
