#!/usr/bin/env Rscript
# TWOMBLI Metrics Analysis Pipeline
# Author: fsumayyamohamed
# Date: 2025-07-18
# License: MIT
# Description: Load TWOMBLI ROI metrics for BRCA samples, assign groups,
#              compute summary statistics, generate heatmaps and boxplots.

# ==== I) Command-line arguments ===========================================
suppressPackageStartupMessages(library(optparse))

option_list <- list(
  make_option(c("-i", "--input"), type="character", required=TRUE,
              help="Path to Excel file with TWOMBLI results", metavar="file"),
  make_option(c("-o", "--output_dir"), type="character", default=".",
              help="Directory for output figures and tables [default: %default]", metavar="dir"),
  make_option(c("-m", "--metrics"), type="character", default=NULL,
              help="Comma-separated list of metrics to include (default: all except ROI)", metavar="metrics"),
  make_option(c("-g", "--groups"), type="character", default=NULL,
              help="Definition of groups as R formula, e.g. '22000466,22002462:<35 young;21001338,21002312:<35 discord'", metavar="string")
)
opt <- parse_args(OptionParser(option_list=option_list))

# Prepare output directory
if (!dir.exists(opt$output_dir)) dir.create(opt$output_dir, recursive=TRUE)

# ==== II) Install and load libraries ======================================
required_pkgs <- c("readxl", "dplyr", "tidyr", "pheatmap", "ggplot2", "ggpubr")
for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly=TRUE)) install.packages(pkg, repos="https://cloud.r-project.org")
  library(package = pkg, character.only = TRUE)
}

# ==== III) Read data & assign groups ======================================
message("Loading data from: ", opt$input)
df <- read_excel(opt$input)

# Extract Sample ID and assign Group
message("Assigning sample IDs and groups...")
df <- df %>%
  mutate(
    Sample = sub("_.*", "", ROI),
    Group = case_when(
      Sample %in% c("22000466","22002462") ~ "<35 young BRCA",
      Sample %in% c("21001338","21002312") ~ "<35 BRCA discord",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(Group))

# Define metrics to analyze
if (is.null(opt$metrics)) {
  metrics <- setdiff(names(df), c("ROI", "Sample", "Group", "TotalImageArea"))
} else {
  metrics <- strsplit(opt$metrics, ",")[[1]]
}

# Create directory to save outputs
fig_dir <- file.path(opt$output_dir, "figures")
if (!dir.exists(fig_dir)) dir.create(fig_dir)

# ==== IV) Summary statistics =============================================
message("Computing summary statistics...")
long_df <- df %>%
  pivot_longer(all_of(metrics), names_to = "Metric", values_to = "Value")

stats_tbl <- long_df %>%
  group_by(Metric) %>%
  summarise(
    n         = n(),
    meanA     = mean(Value[Group == "<35 young BRCA"], na.rm=TRUE),
    meanB     = mean(Value[Group == "<35 BRCA discord"], na.rm=TRUE),
    sdA       = sd(Value[Group == "<35 young BRCA"], na.rm=TRUE),
    sdB       = sd(Value[Group == "<35 BRCA discord"], na.rm=TRUE),
    t_p.value = t.test(Value ~ Group)$p.value,
    .groups   = "drop"
  ) %>%
  mutate(p.adj = p.adjust(t_p.value, method = "fdr"))

write.csv(stats_tbl, file.path(opt$output_dir, "metric_stats.csv"), row.names=FALSE)
message("Saved statistics to metric_stats.csv")

# ==== V) Heatmap of mean metrics =========================================
message("Generating heatmap...")
mean_mat <- df %>%
  group_by(Sample) %>%
  summarise(across(all_of(metrics), mean, na.rm=TRUE), .groups="drop") %>%
  column_to_rownames("Sample")

pheatmap(
  mat            = as.matrix(mean_mat),
  scale          = "row",
  cluster_rows   = FALSE,
  cluster_cols   = FALSE,
  main           = "Mean Metrics per Sample",
  annotation_row = df %>% select(Sample, Group) %>% distinct() %>% column_to_rownames("Sample"),
  filename       = file.path(fig_dir, "mean_metrics_heatmap.pdf")
)
message("Saved heatmap to figures/mean_metrics_heatmap.pdf")

# ==== VI) Boxplots with significance =====================================
message("Plotting boxplots with significance annotations...")
p <- ggplot(long_df, aes(x=Group, y=Value, fill=Group)) +
  geom_boxplot(outlier.shape=NA, width=0.6, alpha=0.7, show.legend=FALSE) +
  geom_jitter(width=0.15, size=1, aes(color=Group), show.legend=FALSE) +
  stat_compare_means(method="t.test", label="p.signif", hide.ns=TRUE) +
  facet_wrap(~ Metric, scales="free_y", ncol=4) +
  theme_bw(base_size=12) +
  theme(
    axis.text.x = element_text(angle=25, hjust=1),
    strip.text  = element_text(size=9)
  ) +
  labs(title="Comparison of <35 BRCA Groups Across Metrics", x=NULL, y="Value")

ggsave(
  filename = file.path(fig_dir, "metrics_boxplots.pdf"),
  plot     = p,
  device   = "pdf",
  width    = 12,
  height   = 8,
  units    = "in"
)
message("Saved boxplots to figures/metrics_boxplots.pdf")

# ==== VII) Session info ==================================================
message("Analysis complete. Session info:")
print(sessionInfo())
