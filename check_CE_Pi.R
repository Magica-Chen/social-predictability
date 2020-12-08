library("data.table")
library("dplyr")
library("ggplot2")
library("reshape2")
library("latex2exp")
library("magrittr")
library("ggpubr")

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

#--------------------
### """DataSet Check for co-locatorship"""
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
df_CE$group %<>% factor(levels= c("useless","useful"))

#---------## Histgram for these dataset
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


#--------------------
### """DataSet Check for TFN"""
wp_true_CE <- read.csv("temp_data/wp-150/wp-true_CE.csv")
bk_true_CE <- read.csv("temp_data/bk-150/bk-true_CE.csv")
gw_true_CE <- read.csv("temp_data/gws-150/gws-true_CE.csv")


wp_true_CE <- wp_true_CE[!is.na(wp_true_CE$CE_alter), ]
bk_true_CE <- bk_true_CE[!is.na(bk_true_CE$CE_alter), ]
gw_true_CE <- gw_true_CE[!is.na(gw_true_CE$CE_alter), ]


wp_true_CE$dataset <- "Weeplace"
bk_true_CE$dataset <- "BrightKite"
gw_true_CE$dataset <- "Gowalla"

df_true_CE <- rbindlist(list(wp_true_CE, bk_true_CE, gw_true_CE))
# add a N_previous requirement M_previous >=150
df_true_CE <- within(df_true_CE, group[N_previous<150] <- 'useless')
df_true_CE$group %<>% factor(levels= c("useless","useful"))

### Histgram for these dataset
ggplot(df_true_CE, aes(x = CE_alter)) +
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
    values = c(colors_10[8], colors_10[10])
  ) +
  # scale_color_manual(labels=c('A', 'B'), values = colors_10[1:2]) +
  facet_wrap(~dataset, scales = "free") +
  labs(x = "LZ cross-entropy (bit)")

ggsave(
  filename = "hist_true_CE_Np.pdf", device = "pdf",
  width = 9.90, height = 2.66,
  path = "fig/"
)

#------------------------------------------------------
# check how many alters as useful in co-locatorship and TFN

df_CE_useful <- df_CE[df_CE$group == "useful", ]
D = df_CE_useful %>%
  group_by(dataset) %>%
  mutate(n_ego = n_distinct(userid_x), n_alters = n_distinct(userid_y))

df_true_CE_useful <- df_true_CE[df_true_CE$group == "useful", ]
E = df_true_CE_useful %>%
  group_by(dataset) %>%
  mutate(n_ego = n_distinct(userid_x), n_alters = n_distinct(userid_y))

#------------------------------------------------------
# Only focus on usefull group's cross-predictability--
df_CE_useful$category <- 'Co-locationship'
df_CE_useful$meetup <- NULL
df_true_CE_useful$category <- 'Social relationship'
df_useful <- rbindlist(list(df_CE_useful, df_true_CE_useful))

p <- ggplot(df_useful, aes(x = Pi_alter)) +
  geom_density(aes(fill = category,  color=category),
    alpha = 0.6, position = "identity"
  ) +
  theme(
    legend.title = element_blank(),
    legend.position="top",
    legend.text = element_text(size=12),
    legend.spacing.x = unit(0.5, "char"),
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
  scale_colour_manual(
    values = colors_10[1:2]
  ) + 
  scale_fill_manual(
    values = colors_10[1:2]
  ) +
  facet_wrap(~dataset, scales = "free") +
  labs(x = unname(TeX(c("Cross-predictability $\\Pi_{ego, alter}"))))
print(p)

ggsave(
  filename = "hist_cross_predictability_Np.pdf", device = "pdf",
  width = 9, height = 3,
  path = "fig/"
)

### Focus on relative cross-predictability
# df_CE_useful <- df_CE[df_CE$group == "useful", ]

df_useful$Pi_alter_rate <- (df_useful$Pi_alter) / (df_useful$Pi)
df_useful <- df_useful[df_useful$Pi_alter_rate < 5, ]

q <- ggplot(df_useful, aes(x = Pi_alter_rate)) +
  geom_density(aes(fill = category, color=category),
    alpha = 0.6, position = "identity") +
  theme(
    legend.title = element_blank(),
    legend.position="top",
    legend.text = element_text(size=12),
    legend.spacing.x = unit(0.5, "char"),
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
  scale_colour_manual(
    values = colors_10[1:2]
  ) + 
  scale_fill_manual(
    values = colors_10[1:2]
  ) +
  facet_wrap(~dataset, scales = "free") +
  labs(x = unname(TeX(c("Relative cross-predictability $R_{alter|ego}"))))

print(q)

ggsave(
  filename = "hist_relative_cross_predictability_Np.pdf", device = "pdf",
  width = 9, height = 3,
  path = "fig/"
)


#--------------
# put p and q together
ggarrange(
  p, q, labels = c("A", "B"), nrow = 2, ncol = 1,
  common.legend = TRUE, legend = "top"
)

ggsave(
  filename = "density_CP_RCP_Np.pdf", device = "pdf",
  width = 9, height = 5,
  path = "fig/"  
)


#--------------
# only useful both CE and PI over three datasets.

df_useful_both <- df_useful %>% select(CE_alter, Pi_alter, dataset, category)

ggplot(df_useful_both, aes(x = CE_alter)) +
  geom_density(aes(fill = dataset, color=dataset),
               alpha = 0.6, position = "identity"
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
    values = colors_10[1:3]
  ) +
  scale_color_manual(values = colors_10[1:3]) +
  facet_wrap(~category, scales = "free") + 
  labs(x = "LZ cross-entropy (bit)")

ggsave(
  filename = "density_CE_dataset_CLN_VS_SRN.pdf", device = "pdf",
  width = 9.00, height = 2.96,
  path = "fig/"
)