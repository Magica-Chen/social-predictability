library("data.table")
library("dplyr")
library("ggplot2")
library("reshape2")
library("latex2exp")

theme_set(
  theme_bw() +
    theme(legend.position = "right")
)

colors_10 <- c(
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
)
catscale10 <- scale_colour_manual(values = colors_10)
catscale10_2 <- scale_fill_manual(values = colors_10)


### """DataSet Check"""
wp_CE <- read.csv("temp_data/wp-150/wp-150-H/wp_CE_over1.csv")
bk_CE <- read.csv("temp_data/bk-150/bk-150-H/bk_CE_over1.csv")
gw_CE <- read.csv("temp_data/gws-150/gws-150-H/gws_CE_over1.csv")


wp_CE <- wp_CE[!is.na(wp_CE$CE_alter), ]
bk_CE <- bk_CE[!is.na(bk_CE$CE_alter), ]
gw_CE <- gw_CE[!is.na(gw_CE$CE_alter), ]


wp_CE$dataset <- "Weeplace"
bk_CE$dataset <- "BrightKite"
gw_CE$dataset <- "Gowalla"

df_CE <- rbindlist(list(wp_CE, bk_CE, gw_CE))
# add a N_previous requirement M_previous >=150
df_CE <- within(df_CE, group[N_previous<150] <- 'useless')


### Histgram for these dataset

ggplot(df_CE, aes(x = CE_alter)) +
  geom_histogram(aes(fill = group),
    alpha = 0.8, position = "identity", bins = 100
  ) +
  theme(
    legend.title = element_blank(),
    legend.spacing.x = unit(0.5, "char"),
    strip.text = element_text(size = 15),
    legend.text = element_text(
      face = "bold",
      size = 12
    ),
    axis.text = element_text(
      face = "bold",
      size = 12
    ),
    axis.title = element_text(
      face = "bold",
      size = 12
    ),
    # plot.title=element_text(face='bold', size=12,hjust = 0.5)
  ) +
  scale_fill_manual(
    # labels = unname(TeX(c("Useful", "$Useless"))),
    values = colors_10[8:9]
  ) +
  # scale_color_manual(labels=c('A', 'B'), values = colors_10[1:2]) +
  facet_wrap(~dataset, scales = "free") +
  labs(x = "LZ cross-entropy (bit)")

ggsave(
  filename = "hist_CE_Np.pdf", device = "pdf",
  width = 9.90, height = 2.66,
  path = "fig/"
)


### Only focus on usefull group's cross-predictability
df_CE_useful <- df_CE[df_CE$group == "useful", ]
D = df_CE_useful %>%
  group_by(dataset) %>%
  mutate(n_ego = n_distinct(userid_x), n_alters = n_distinct(userid_y))

ggplot(df_CE_useful, aes(x = Pi_alter)) +
  geom_histogram(aes(fill = group),
    alpha = 0.8, position = "identity", bins = 100
  ) +
  theme(
    legend.position = "none",
    strip.text = element_text(size = 15),
    axis.text = element_text(
      face = "bold",
      size = 12
    ),
    axis.title = element_text(
      face = "bold",
      size = 12
    ),
    # plot.title=element_text(face='bold', size=12,hjust = 0.5)
  ) +
  scale_fill_manual(
    values = colors_10[5]
  ) +
  facet_wrap(~dataset, scales = "free") +
  labs(x = "Cross-predictability")

ggsave(
  filename = "hist_cross_predictability_Np.pdf", device = "pdf",
  width = 9.90, height = 2.66,
  path = "fig/"
)

### Focus on relative cross-predictability
df_CE_useful <- df_CE[df_CE$group == "useful", ]
df_CE_useful$Pi_alter_rate <- (df_CE_useful$Pi_alter) / (df_CE_useful$Pi)
df_CE_useful <- df_CE_useful[df_CE_useful$Pi_alter_rate < 5, ]

ggplot(df_CE_useful, aes(x = Pi_alter_rate)) +
  geom_histogram(aes(fill = group),
    alpha = 0.8, position = "identity", bins = 100
  ) +
  theme(
    legend.position = "none",
    strip.text = element_text(size = 15),
    axis.text = element_text(
      face = "bold",
      size = 12
    ),
    axis.title = element_text(
      face = "bold",
      size = 12
    ),
    # plot.title=element_text(face='bold', size=12,hjust = 0.5)
  ) +
  scale_fill_manual(
    values = colors_10[6]
  ) +
  facet_wrap(~dataset, scales = "free") +
  labs(x = "Relative cross-predictability")

ggsave(
  filename = "hist_relative_cross_predictability_Np.pdf", device = "pdf",
  width = 9.90, height = 2.66,
  path = "fig/"
)
