---
title: "FOV Fraction Analysis & Statistical Testing"
author: "fsumayyamohamed"
date: "18.07.25"
output:
  html_document:
    toc: true
    toc_depth: 2
params:
  input_dir:           "data/seurat_objects_withcelltypes/split_seurats"
  out_wilcox_csv:      "results/wilcox_all_celltype_comparisons.csv"
  out_mix_csv:         "results/mixed_effects_results.csv"
  out_boxplot_png:     "figures/boxplot_celltype_fractions.png"
  out_stack_png:       "figures/stacked_bar_mean_composition.png"
  out_pair_png:        "figures/stacked_bar_pairwise.png"
---

```{r setup, include=FALSE}
# Load libraries and set options
library(here)
library(dplyr)
library(tidyr)
library(ggplot2)
library(purrr)
library(stringr)
library(tibble)
library(rstatix)
library(ggpubr)
library(lme4)
library(lmerTest)
library(broom)
knitr::opts_chunk$set(
  echo = TRUE,
  message = FALSE,
  warning = FALSE
)
```

## 1. Define Sample Metadata & Input Directory

```{r metadata}
# Relative input path
data_dir <- here(params$input_dir)

# Map sample IDs to conditions
condition_map <- tribble(
  ~sample_id, ~condition,
  "22003431", "<35 BRCA, accelerated ageing",
  "22003460", "other",
  "22002320", "other",
  "21001338", "<35 BRCA, accelerated ageing",
  "22002462", "<35 BRCA, concordant ageing",
  "21002305", ">55 BRCA, concordant ageing",
  "21002306", ">55 BRCA, concordant ageing",
  "22000466", "<35 BRCA, concordant ageing",
  "21002312", "<35 BRCA, accelerated ageing",
  "22002145", "<35 BRCA, concordant ageing",
  "17063451", ">55 NON BRCA",
  "19001625", "<35 NON BRCA",
  "19001626", "<35 NON BRCA",
  "22001589", ">55 BRCA, concordant ageing",
  "17063106", ">55 NON BRCA"
)
```

## 2. Load Seurat Objects

```{r load-seurats}
# Function to read all matching RDS and tag sample_id
load_seurats <- function(dir, pattern = "^seurat_\\d+\\.RDS$") {
  files <- list.files(dir, pattern = pattern, full.names = TRUE)
  map(files, ~ {
    so <- readRDS(.x)
    sid <- str_extract(basename(.x), "\\d+")
    so$sample_id <- sid
    if (!"fov" %in% colnames(so@meta.data)) {
      stop("Missing 'fov' metadata for ", sid)
    }
    so
  }) %>%
    set_names(map_chr(files, ~ str_extract(basename(.x), "\\d+")))
}

samples <- load_seurats(data_dir)
```

## 3. Compute FOV × Cell-Type Fractions

```{r compute-fractions}
prop_fov_df <- samples %>%
  imap_dfr(~ {
    md   <- .x@meta.data
    # collapse luminal subtypes
    md2  <- md %>% mutate(
      cell_type = case_when(
        cell_type %in% c("Luminal Progenitors","Luminal secretory") ~ "LASP",
        cell_type == "Mature Luminal"                              ~ "LHS",
        cell_type %in% c("T cells","T Cells")                   ~ "T cells",
        cell_type == "PVL"                                         ~ "Periovascular-like",
        TRUE                                                         ~ cell_type
      )
    )
    tibble(FOV = md2$fov, cell_type = md2$cell_type) %>%
      count(FOV, cell_type, name = "count") %>%
      group_by(FOV) %>%
      mutate(
        fraction  = count / sum(count),
        sample_id = .y
      ) %>%
      ungroup()
  }) %>%
  left_join(condition_map, by = "sample_id") %>%
  mutate(
    condition = factor(condition, levels = unique(condition_map$condition))
  )
```

## 4. Wilcoxon Tests per Cell Type

```{r wilcox-tests}
wilcox_res <- prop_fov_df %>%
  group_by(cell_type) %>%
  wilcox_test(fraction ~ condition) %>%
  adjust_pvalue(method = "BH") %>%
  add_significance("p.adj") %>%
  select(cell_type, group1, group2, n1, n2, statistic, p, p.adj, p.adj.signif)

# Save results
write.csv(
  wilcox_res,
  here(params$out_wilcox_csv),
  row.names = FALSE
)
```

## 5. Mixed-Effects Models per Cell Type

```{r mixed-effects}
mix_res <- prop_fov_df %>%
  group_by(cell_type) %>%
  do({
    df  <- .
    mod <- lmer(fraction ~ condition + (1 | sample_id), data = df)
    coefs <- summary(mod)$coefficients
    tibble(
      term      = rownames(coefs),
      Estimate  = coefs[, "Estimate"],
      StdError  = coefs[, "Std. Error"],
      df        = coefs[, "df"],
      t_value   = coefs[, "t value"],
      p_value   = coefs[, "Pr(>|t|)"],
      cell_type = df$cell_type[1]
    )
  }) %>% ungroup()

# Save mixed-effects summary
write.csv(
  mix_res,
  here(params$out_mix_csv),
  row.names = FALSE
)
```

## 6. Boxplot of Cell-Type Fractions

```{r boxplot}
# Generate boxplot with p-values
p_box <- ggplot(prop_fov_df, aes(condition, fraction)) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(width = 0.2, size = 1, alpha = 0.5) +
  facet_wrap(~ cell_type, scales = "free_y") +
  labs(title = "Cell-type fractions by condition",
       x = NULL, y = "Fraction per FOV") +
  theme_minimal(base_size = 13) +
  theme(
    axis.text.x      = element_text(angle = 45, hjust = 1),
    strip.background = element_rect(fill = "grey90", color = NA)
  ) +
  stat_pvalue_manual(
    wilcox_res,
    label      = "p.adj.signif",
    inherit.aes = FALSE,
    x          = "group1",
    xend       = "group2",
    y.position = max(prop_fov_df$fraction) * 1.05
  )

# Save boxplot
ggsave(
  here(params$out_boxplot_png),
  plot = p_box,
  width = 10, height = 8,
  units = "in", dpi = 300
)
```

## 7. Stacked Bar: Mean Composition by Condition

```{r stacked-bar}
# Define order
keep <- c(
  "<35 NON BRCA",
  "<35 BRCA, concordant ageing",
  ">55 NON BRCA",
  ">55 BRCA, concordant ageing",
  "<35 BRCA, accelerated ageing"
)
epithelial_ct <- c("LASP","LHS","Myoepithelial")
all_ct <- unique(prop_fov_df$cell_type)
cell_order <- c(epithelial_ct, setdiff(all_ct, epithelial_ct))

stack_df <- prop_fov_df %>%
  filter(condition %in% keep) %>%
  group_by(condition, FOV) %>%
  complete(cell_type = cell_order, fill = list(fraction = 0)) %>%
  ungroup() %>%
  group_by(condition, cell_type) %>%
  summarise(mean_frac = mean(fraction), .groups = "drop") %>%
  mutate(
    condition = factor(condition, levels = keep),
    cell_type = factor(cell_type, levels = cell_order)
  )

p_stack <- ggplot(stack_df, aes(x = condition, y = mean_frac, fill = cell_type)) +
  geom_col(color = "white", size = 0.1, position = position_stack(reverse = TRUE)) +
  scale_y_continuous(labels = scales::percent_format(), expand = c(0,0)) +
  scale_fill_brewer(palette = "Spectral") +
  labs(title = "Average cell-type composition by condition",
       x = NULL, y = "Fraction of cells", fill = "Cell Type") +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid.major.x = element_blank()
  )

# Save stacked bar
ggsave(
  here(params$out_stack_png),
  plot = p_stack,
  width = 10, height = 6,
  units = "in", dpi = 300
)
```

## 8. Pairwise Stacked Bar for Two Conditions

```{r pairwise-bar}
# Conditions of interest
conds <- c("<35 BRCA, concordant ageing", "<35 BRCA, accelerated ageing")

# Recompute summary for these two
small_df <- stack_df %>%
  filter(condition %in% conds) %>%
  mutate(condition = factor(condition, levels = conds))

# Pairwise Wilcox tests
pair_w <- prop_fov_df %>%
  filter(condition %in% conds) %>%
  group_by(cell_type) %>%
  wilcox_test(fraction ~ condition) %>%
  adjust_pvalue(method = "BH") %>%
  add_significance("p.adj") %>%
  select(cell_type, star = p.adj.signif)

# Compute label positions on accelerated bar
label_df <- small_df %>%
  filter(condition == conds[2]) %>%
  arrange(cell_type) %>%
  mutate(cum_frac = cumsum(mean_frac), ypos = cum_frac + 0.02) %>%
  left_join(pair_w, by = "cell_type") %>%
  filter(star != "ns") %>%
  mutate(x = conds[2])

p_pair <- ggplot(small_df, aes(x = condition, y = mean_frac, fill = cell_type)) +
  geom_col(color = "white", position = position_stack(reverse = TRUE)) +
  scale_y_continuous(labels = scales::percent_format(), expand = c(0,0)) +
  scale_fill_brewer(palette = "Spectral") +
  coord_cartesian(clip = "off") +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid.major.x = element_blank()
  ) +
  geom_text(
    data = label_df,
    aes(x = x, y = ypos, label = star),
    inherit.aes = FALSE,
    size = 5
  )

# Save pairwise bar
ggsave(
  here(params$out_pair_png),
  plot = p_pair,
  width = 8, height = 6,
  units = "in", dpi = 300
)
```

# Example: permutation test for one cell type in one contrast
cell_of_interest <- "LASP"
contrast        <- c("<35 BRCA, concordant ageing", "<35 BRCA, accelerated ageing")
n_perms         <- 5000  # number of permutations

# 1. Subset your data
df0 <- prop_fov_df %>%
  filter(cell_type == cell_of_interest,
         condition %in% contrast) %>%
  select(FOV, fraction, condition)

# 2. Compute the observed Wilcoxon W‑statistic
obs_stat <- wilcox_test(fraction ~ condition, data = df0)$statistic

# 3. Permutation loop
perm_stats <- replicate(n_perms, {
  shuffled <- df0 %>%
    mutate(condition = sample(condition))    # shuffle labels
  wilcox_test(fraction ~ condition, data = shuffled)$statistic
})

# 4. Calculate empirical p‑value
#    Count how many permuted stats are ≥ (for one‑sided) or | | ≥ | | (two‑sided)
perm_pval <- mean(abs(perm_stats - mean(perm_stats)) >= abs(obs_stat - mean(perm_stats)))

cat("Observed W =", obs_stat, "\n")
cat("Permutation p‑value =", perm_pval, "\n")


## 9. Session Info

```{r session-info}
sessionInfo()
