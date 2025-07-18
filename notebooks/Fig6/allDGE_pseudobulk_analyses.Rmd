---
title: "Comprehensive Pseudobulk & Pseudocompartment DGE Analyses"
author: "fsumayyamohamed"
date: "18.07.25"
output:
  html_document:
    toc: true
    toc_depth: 3
params:
  input_dir:   "data/seurat_objects_withcelltypes/split_seurats"
  output_dir:  "results/pseudobulk_dge"
  min_fc:      0.5
  fdr_thr:     0.05
---

```{r setup, include=FALSE}
# Load libraries and create output directory
library(here); library(dplyr); library(tidyr)
library(edgeR); library(ggplot2); library(ggrepel); library(pheatmap)
knitr::opts_chunk$set(echo=TRUE, message=FALSE, warning=FALSE)
dir.create(here(params$output_dir), recursive=TRUE, showWarnings=FALSE)
```

# 1. Build Pseudobulk Matrix and Metadata

```{r pseudobulk}
# Read split Seurat objects
fns     <- list.files(here(params$input_dir), "^seurat_\\d+\\.RDS$", full.names=TRUE)
samples <- lapply(fns, readRDS)
names(samples) <- gsub("^.*seurat_(\\d+)\\.RDS$", "\\1", fns)

# Construct pseudobulk counts (sample_id × cell_type)
pb_list  <- lapply(samples, function(so) {
  mat    <- GetAssayData(so, assay="RNA", slot="counts")
  groups <- paste(so@meta.data$sample_id, so@meta.data$cell_type, sep="_")
  ug     <- unique(groups)
  out    <- sapply(ug, function(g) rowSums(mat[, groups==g, drop=FALSE]))
  colnames(out) <- ug
  out
})
pb_counts <- do.call(cbind, pb_list)

# Build sample‐level metadata
df_meta  <- tibble(sample_celltype = colnames(pb_counts)) %>%
  separate(sample_celltype, into=c("sample_id","cell_type"), sep="_") %>%
  mutate(
    condition = recode(sample_id,
      "22003431" = "<35 BRCA, accelerated ageing",
      "21002312" = "<35 BRCA, accelerated ageing",
      "22002462" = "<35 BRCA, concordant ageing",
      "22002145" = "<35 BRCA, concordant ageing",
      "17063106" = ">55 NON BRCA",
      "19001625" = "<35 NON BRCA",
      "21002305" = ">55 BRCA, concordant ageing",
      "21002306" = ">55 BRCA, concordant ageing",
      "22000466" = "<35 BRCA, concordant ageing",
      "22003460" = "other",
      "22002320" = "other",
      "17063106" = ">55 NON BRCA"
    ),
    batch = recode(sample_id,
      "22003431" = "Slide1","22003460"="Slide1","22002320"="Slide1",
      "21001338" = "Slide3","22002462"="Slide3","21002305"="Slide3",
      "21002306" = "Slide4","22000466"="Slide4","21002312"="Slide4",
      "22002145" = "Slide5","17063451"="Slide5","19001625"="Slide5",
      "19001626" = "Slide6","22001589"="Slide6","17063106"="Slide6"
    )
  ) %>%
  filter(!is.na(condition), !is.na(batch))
stopifnot(nrow(df_meta)==ncol(pb_counts))
```

# 2. Build edgeR Objects & Design

```{r design}
# Create DGEList and compute normalization factors
dge    <- DGEList(counts=pb_counts, samples=df_meta)
dge    <- calcNormFactors(dge)
# Design: ~0 + condition + batch
design <- model.matrix(~0 + condition + batch, data=df_meta)
colnames(design)
```

# 3. Global Pseudobulk DGE: Accelerated vs Concordant

```{r global-dge}
contrast <- rep(0, ncol(design)); names(contrast) <- colnames(design)
contrast[paste0("condition<35 BRCA, accelerated ageing")] <-  1
contrast[paste0("condition<35 BRCA, concordant ageing")]  <- -1

fit    <- glmQLFit(dge, design)
qlf    <- glmQLFTest(fit, contrast=contrast)
res    <- topTags(qlf, n=Inf)$table %>% rownames_to_column("gene") %>%
  mutate(regulation = case_when(
    FDR < params$fdr_thr & logFC >  params$min_fc ~ "Up",
    FDR < params$fdr_thr & logFC < -params$min_fc ~ "Down",
    TRUE                                        ~ "NS"
  ))
write.csv(res, here(params$output_dir, "global_acc_vs_con.csv"), row.names=FALSE)
# Top 20 table
knitr::kable(head(res,20), caption="Global Top 20 DEGs: accelerated vs concordant")
# Volcano
ggplot(res, aes(logFC, -log10(FDR), color=regulation)) +
  geom_point(alpha=0.6,size=1)+
  scale_color_manual(values=c(Up='blue',Down='red',NS='grey70'))+
  geom_vline(xintercept=c(-params$min_fc,params$min_fc),linetype='dashed')+
  geom_hline(yintercept=-log10(params$fdr_thr),linetype='dashed')+
  geom_text_repel(data=filter(res, regulation!='NS'), aes(label=gene), size=2.5)+
  theme_minimal() + labs(title='Global: <35 BRCA accelerated vs concordant')
```

# 4. Compartmental Analyses

## 4.1 Define Compartments

```{r compartments}
epithelial <- c("Mature Luminal","Luminal secretory","Luminal Progenitors","Myoepithelial")
df_meta <- df_meta %>% mutate(
  compartment = ifelse(cell_type %in% epithelial, 'Epithelial','Stromal')
)
```

## 4.2 Pseudobulk by Compartment
group compartments and sum counts
```{r pseudobulk-comp}
groups   <- paste(df_meta$sample_id, df_meta$compartment, sep="_")
pb_comp  <- t(rowsum(t(pb_counts), group=groups, reorder=FALSE))
# metadata for pb_comp
meta_comp<- tibble(sample_comp=colnames(pb_comp)) %>%
  separate(sample_comp, into=c('sample_id','compartment'), sep='_') %>%
  mutate(condition=df_meta$condition[match(sample_comp, paste(df_meta$sample_id,df_meta$cell_type,sep='_'))],
         batch    = df_meta$batch[    match(sample_comp, paste(df_meta$sample_id,df_meta$cell_type,sep='_'))])
```

## 4.3 DGE: Stromal Only

```{r stromal-dge}
keep_str <- meta_comp$compartment=='Stromal'
dge_str  <- DGEList(counts=pb_comp[,keep_str], samples=meta_comp[keep_str,]) %>% calcNormFactors()
design_s <- model.matrix(~0+condition+batch, data=meta_comp[keep_str,])
fit_s    <- glmQLFit(dge_str, design_s)
# same contrast vector names in design_s
cn_s     <- colnames(design_s)
vec_s    <- rep(0, length(cn_s)); names(vec_s)<-cn_s
vec_s[paste0('condition<35 BRCA, accelerated ageing')] <-1
vec_s[paste0('condition<35 BRCA, concordant ageing')]  <--1
qlf_s    <- glmQLFTest(fit_s, contrast=vec_s)
res_s    <- topTags(qlf_s,n=Inf)$table %>% rownames_to_column('gene') %>% mutate(sig=FDR<params$fdr_thr)
write.csv(res_s, here(params$output_dir,'stromal_acc_vs_con.csv'),row.names=FALSE)
knitr::kable(head(res_s,20), caption='Stromal Top 20 DEGs')
# Volcano & heatmap
# ... volcano and heatmap code as above
```

# 5. Per‐Cell‐Type Analyses

```{r per-celltype}
cell_types <- unique(df_meta$cell_type)
for(ct in cell_types) {
  meta_ct <- df_meta %>% filter(cell_type==ct)
  keep    <- paste(meta_ct$sample_id,ct,sep='_')
  if (sum(keep %in% colnames(pb_counts))<3) next
  dge_ct  <- DGEList(counts=pb_counts[,keep], samples=meta_ct) %>% calcNormFactors()
  des_ct  <- model.matrix(~0+condition+batch, data=meta_ct)
  fit_ct  <- glmQLFit(dge_ct, des_ct)
  vec_ct  <- rep(0,ncol(des_ct)); names(vec_ct)<-colnames(des_ct)
  vec_ct[paste0('condition<35 BRCA, accelerated ageing')] <-1
  vec_ct[paste0('condition<35 BRCA, concordant ageing')]  <--1
  qlf_ct  <- glmQLFTest(fit_ct, contrast=vec_ct)
  res_ct  <- topTags(qlf_ct,n=Inf)$table %>% rownames_to_column('gene') %>% mutate(sig=FDR<params$fdr_thr)
  # save
  write.csv(res_ct, here(params$output_dir,paste0('DE_',ct,'.csv')),row.names=FALSE)
  # top20
  top20   <- head(res_ct,20)
  print(knitr::kable(top20, caption=paste(ct,'Top 20 DEGs')))
  # volcano
  p <- ggplot(res_ct, aes(logFC,-log10(FDR),color=sig)) + geom_point(alpha=0.6) +
    scale_color_manual(values=c('TRUE'='red','FALSE'='grey70')) +
    labs(title=paste(ct,'DE')) + theme_minimal()
  ggsave(here(params$output_dir,paste0('volcano_',ct,'.pdf')),p,width=6,height=4)
}
```

# 6. Session Information

```{r session-info}
sessionInfo()
```
