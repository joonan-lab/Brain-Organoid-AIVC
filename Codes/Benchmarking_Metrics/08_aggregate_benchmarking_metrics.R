library(readxl)
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)
library(RColorBrewer)
library(extrafont)
library(purrr)

select <- dplyr::select
loadfonts()
par(fontfamily = "Arial")

outdir <- "<PROJECT_ROOT>/model_performances"
figdir <- "<PROJECT_ROOT>/model_performances/figures"
date   <- "260326"

# ------------------------------------------------------------------
# Final integrated benchmarking framework
# ------------------------------------------------------------------
# 1. Prediction metrics: delta-response correlations plus profile errors
# 2. Embedding metrics: cv_knn_dist and mean_knn_dist_ref
# 3. Differential-expression recovery: AUPRC@50 and CentroidAcc
# 4. Biological coherence: FDR-filtered PCA-space Moran's I
# 5. Group-weighted rank aggregation: Pred 30%, Moran 20%, Embed 25%, DE 25%
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# 1. 데이터 로드
# ------------------------------------------------------------------
raw_mat <- readxl::read_xlsx(
  sprintf("%s/AIVC_model_benchmarking_result.260202.xlsx", outdir)
) %>% filter(!Model %in% c("Telen-Mouse", "Telen-NOCAP-Lite"))

# Embedding 지표 (새 파일: cv_knn_dist + mean_knn_dist_ref)
embedding_mat <- read.csv(
  sprintf("%s/graph_metrics_all_models/embedding_metrics_all_models.csv", outdir)
) %>%
  select(Model, cv_knn_dist, mean_knn_dist_ref)

# DE Quality 지표 (AUPRC@50, CentroidAcc)
de_mat <- read.csv(
  sprintf("%s/deg_ranking_metrics/de_metrics_all_models_summary.csv", outdir)
) %>%
  select(Model, `AUPRC.50`, CentroidAcc) %>%
  rename(`AUPRC@50` = `AUPRC.50`)

comb_mat = read_excel(sprintf('%s/AIVC_model_benchmarking_comprehensive_result.%s.xlsx', outdir, date))
comb_mat

models = comb_mat$Model %>% unique()
# ------------------------------------------------------------------
# 2. metric 정의
# ------------------------------------------------------------------
# Delta-only prediction metrics (raw Pearson/Spearman 제외)
metrics_pred_corr  <- c("pearson_delta", "pearson_de_delta",
                        "spearmanr_delta", "spearmanr_de_delta")
metrics_pred_error <- c("rmse", "rmse_de", "mae", "mae_de")
all_pred_metrics   <- c(metrics_pred_corr, metrics_pred_error)

# Embedding: cv_knn_dist↓ better, mean_knn_dist_ref↑ better
metrics_embed_error <- c("cv_knn_dist")
metrics_embed_corr  <- c("mean_knn_dist_ref")

# DE Quality: both ↑ better
metrics_de <- c("AUPRC@50", "CentroidAcc")

moran_metric <- "Moran_I_ASD185_EAGLE"
alpha        <- 0.05

# ------------------------------------------------------------------
# 3. Moran's I 유의성 정의
# ------------------------------------------------------------------
df_scores <- comb_mat %>%
  mutate(
    moran_sig = !is.na(.data[[moran_metric]]) &
      !is.na(FDR) &
      FDR < alpha &
      .data[[moran_metric]] > 0
  ) %>%
  select(Model,
         all_of(all_pred_metrics),
         all_of(moran_metric), moran_sig,
         all_of(metrics_embed_corr), all_of(metrics_embed_error),
         all_of(metrics_de))

# ------------------------------------------------------------------
# 4. long format
# ------------------------------------------------------------------
# Prediction
df_long_pred <- df_scores %>%
  select(Model, all_of(all_pred_metrics)) %>%
  pivot_longer(cols = -Model, names_to = "metric", values_to = "value")

# Moran's I (유의하지 않으면 NA)
df_moran <- df_scores %>%
  transmute(
    Model,
    metric = moran_metric,
    value  = if_else(moran_sig, .data[[moran_metric]], NA_real_)
  )

# Embedding
df_embed <- df_scores %>%
  select(Model, all_of(metrics_embed_corr), all_of(metrics_embed_error)) %>%
  pivot_longer(cols = -Model, names_to = "metric", values_to = "value")

# DE Quality
df_de <- df_scores %>%
  select(Model, all_of(metrics_de)) %>%
  pivot_longer(cols = -Model, names_to = "metric", values_to = "value")

df_long <- bind_rows(df_long_pred, df_moran, df_embed, df_de)

# ------------------------------------------------------------------
# 5. metric group / label
# ------------------------------------------------------------------
df_long <- df_long %>%
  mutate(
    metric_group = case_when(
      metric %in% all_pred_metrics    ~ "Prediction",
      metric == moran_metric          ~ "Moran's I",
      metric %in% c(metrics_embed_corr, metrics_embed_error) ~ "Embedding",
      metric %in% metrics_de          ~ "DE Quality",
      TRUE ~ "Other"
    ),
    metric_label = recode(
      metric,
      pearson_delta      = "Pearson-\u0394",
      pearson_de_delta   = "Pearson-DE\u0394",
      spearmanr_delta    = "Spearman-\u0394",
      spearmanr_de_delta = "Spearman-DE\u0394",
      rmse               = "RMSE",
      rmse_de            = "RMSE-DE",
      mae                = "MAE",
      mae_de             = "MAE-DE",
      Moran_I_ASD185_EAGLE = "Moran's I",
      mean_knn_dist_ref  = "kNN Dist",
      cv_knn_dist        = "CV kNN Dist",
      `AUPRC@50`         = "AUPRC@50",
      CentroidAcc        = "CentroidAcc"
    )
  )

# ------------------------------------------------------------------
# 6. rank 계산 (metric별, higher/lower is better 처리)
# ------------------------------------------------------------------
higher_is_better <- c(
  "pearson_delta", "pearson_de_delta", "spearmanr_delta", "spearmanr_de_delta",
  "Moran_I_ASD185_EAGLE", "mean_knn_dist_ref",
  "AUPRC@50", "CentroidAcc"
)
lower_is_better <- c(
  "rmse", "rmse_de", "mae", "mae_de",
  "cv_knn_dist"
)

df_long <- df_long %>%
  group_by(metric) %>%
  mutate(
    rank = case_when(
      metric %in% higher_is_better ~
        rank(-value, ties.method = "average", na.last = "keep"),
      metric %in% lower_is_better ~
        rank( value, ties.method = "average", na.last = "keep"),
      TRUE ~ NA_real_
    )
  ) %>%
  ungroup()

# ------------------------------------------------------------------
# 7. bar 길이 = rank 기반 (1=best → bar 가장 길게)
# ------------------------------------------------------------------
df_long <- df_long %>%
  group_by(metric) %>%
  mutate(
    max_rank = max(rank, na.rm = TRUE),
    bar_len  = (max_rank - rank + 1) / max_rank
  ) %>%
  ungroup()

# ------------------------------------------------------------------
# 8. color score 정의 (0=worst, 1=best)
# ------------------------------------------------------------------
df_long <- df_long %>%
  mutate(
    color_score = case_when(
      metric %in% higher_is_better ~ if_else(is.na(value), NA_real_,
                                             rescale(value, to = c(0, 1))),
      metric %in% lower_is_better  ~ if_else(is.na(value), NA_real_,
                                             1 - rescale(value, to = c(0, 1))),
      TRUE ~ NA_real_
    )
  )

# ------------------------------------------------------------------
# 9. color mapping (group별 palette)
# ------------------------------------------------------------------
df_long_split <- df_long %>%
  split(list(df_long$metric, df_long$metric_group), drop = TRUE)

df_long_colored <- map_df(df_long_split, function(g) {
  if (all(is.na(g$color_score))) {
    g$fill_col <- NA_character_
    return(g)
  }
  pal_name <- case_when(
    unique(g$metric_group) == "Prediction" ~ "YlGnBu",
    unique(g$metric_group) == "Embedding"  ~ "PuBuGn",
    unique(g$metric_group) == "DE Quality" ~ "Purples",
    unique(g$metric_group) == "Moran's I"  ~ "Reds",
    TRUE ~ "Greys"
  )
  pal <- colorRampPalette(brewer.pal(9, pal_name))(100)
  idx <- round(rescale(g$color_score, to = c(1, 100)))
  g$fill_col <- pal[idx]
  g
})

df_long <- df_long_colored

# ------------------------------------------------------------------
# 10. Final model ranking — group-weighted rank
#
#   Prediction  : 30%  (delta-only 8 metrics)
#   Moran's I   : 20%  (FDR-filtered; non-sig → excluded from group)
#   Embedding   : 25%  (cv_knn_dist↓ + mean_knn_dist_ref↑)
#   DE Quality  : 25%  (AUPRC@50↑ + CentroidAcc↑)
#
#   Moran's I non-sig → NaN group mean rank → weight redistributed
#   Final sort: moran_sig DESC → weighted_score ASC
# ------------------------------------------------------------------
group_weights <- c(
  "Prediction" = 0.30,
  "Moran's I"  = 0.20,
  "Embedding"  = 0.25,
  "DE Quality" = 0.25
)

method_rank_tbl <- df_long %>%
  filter(metric_group %in% names(group_weights)) %>%
  group_by(Model, metric_group) %>%
  summarise(group_mean_rank = mean(rank, na.rm = TRUE), .groups = "drop") %>%
  # NaN if all ranks were NA (Moran non-sig case)
  mutate(group_mean_rank = if_else(is.nan(group_mean_rank), NA_real_, group_mean_rank)) %>%
  group_by(Model) %>%
  summarise(
    moran_sig_any  = any(metric_group == "Moran's I" & !is.na(group_mean_rank)),
    weighted_score = {
      present <- metric_group[!is.na(group_mean_rank)]
      gmr     <- group_mean_rank[!is.na(group_mean_rank)]
      w       <- group_weights[present]
      sum(gmr * w) / sum(w)
    },
    .groups = "drop"
  ) %>%
  arrange(desc(moran_sig_any), weighted_score) %>%
  mutate(final_rank = row_number())

model_order <- method_rank_tbl$Model

cat("\n=== Final Model Ranking (group-weighted metric ranks) ===\n")
print(method_rank_tbl %>% select(final_rank, Model, moran_sig_any, weighted_score),
      n = 20)

# ------------------------------------------------------------------
# 10b. Total (종합 순위) column 추가
# ------------------------------------------------------------------
N_models  <- nrow(method_rank_tbl)
total_pal <- colorRampPalette(c("#FFF7BC", "#D95F0E"))(100)  # yellow → dark orange

df_total <- method_rank_tbl %>%
  mutate(
    metric       = "Total",
    metric_group = "Total",
    metric_label = "Total",
    value        = weighted_score,
    rank         = as.numeric(final_rank),
    max_rank     = N_models,
    bar_len      = (N_models - final_rank + 1) / N_models,
    color_score  = (N_models - final_rank) / (N_models - 1),
    fill_col     = total_pal[round(rescale(color_score, to = c(1, 100)))]
  ) %>%
  select(Model, metric, metric_group, metric_label, value,
         rank, max_rank, bar_len, color_score, fill_col)

df_long <- bind_rows(df_long, df_total)

df_long <- df_long %>%
  mutate(Model = factor(Model, levels = rev(model_order)))

# ------------------------------------------------------------------
# 11. facet 순서
# ------------------------------------------------------------------
metric_level_order <- c(
  "Pearson-\u0394", "Pearson-DE\u0394", "Spearman-\u0394", "Spearman-DE\u0394",
  "RMSE", "RMSE-DE", "MAE", "MAE-DE",
  "kNN Dist", "CV kNN Dist",
  "AUPRC@50", "CentroidAcc",
  "Moran's I",
  "Total"
)

df_long$metric_label <- factor(df_long$metric_label, levels = metric_level_order)

# metric_group 순서 고정
df_long$metric_group <- factor(
  df_long$metric_group,
  levels = c("Prediction", "Embedding", "DE Quality", "Moran's I", "Total")
)

# ------------------------------------------------------------------
# 12. 모델 색상 (sig/non-sig 구분)
# ------------------------------------------------------------------
sig_models <- method_rank_tbl %>%
  filter(moran_sig_any) %>%
  pull(Model)

# ------------------------------------------------------------------
# 13. plot
# ------------------------------------------------------------------
p <- ggplot(df_long, aes(x = bar_len, y = Model)) +
  geom_col(aes(fill = fill_col), width = 0.7, na.rm = TRUE) +
  geom_text(aes(label = ifelse(!is.na(rank), as.integer(round(rank)), "")),
            hjust = -0.25, size = 2.5, na.rm = TRUE, color='black', family='Arial') +
  scale_fill_identity() +
  facet_grid(. ~ metric_group + metric_label,
             scales = "free_x", space = "free_x") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(
    x    = NULL,
    y    = NULL,
    title = "Model Benchmarking Ranking",
    subtitle = "Ranking Weights: Pred 30% | Embed 25% | DE 25% | Moran 20%"
  ) +
  theme_minimal(base_size=12) +
  theme(
    plot.title = element_text(size=14, color='black', face='bold', family='Arial'),
    plot.subtitle = element_text(size=8, color='gray30', family='Arial'),
    panel.grid = element_blank(),
    axis.text.x = element_blank(),
    axis.text.y = element_text(size=10, color="black", family='Arial'),
    strip.text = element_text(size=10, color='black', face="bold", family='Arial'),
    strip.background = element_rect(fill="grey90", colour=NA),
    legend.position = "right"
  )
p

# ------------------------------------------------------------------
# 14. 그룹 경계선을 위한 separator plot (선택)
# ------------------------------------------------------------------
# metric_group 경계에 세로선을 추가하기 위해 vline layer 추가 가능
# (facet_grid의 세로선은 자동으로 표시됨)

### Save ###
ggsave(
  sprintf("%s/AIVC_model_benchmarking_result_plot.%s.pdf", figdir, date),
  p, width = 14, height = 7
)

#write.xlsx(comb_mat, sprintf("%s/AIVC_model_benchmarking_comprehensive_result.%s.xlsx", outdir, date))

