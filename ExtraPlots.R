library("data.table")
library("dplyr")
library("ggplot2")
library("reshape2")
library("latex2exp")
library("magrittr")
library("ggpubr")
library(tikzDevice)
options(tikzMetricPackages = c("\\usepackage[utf8]{inputenc}",
                               "\\usepackage[T1]{fontenc}", "\\usetikzlibrary{calc}",
                               "\\usepackage{amssymb}"))

theme_set(
  theme_bw() +
    theme(legend.position = "top",
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          strip.background = element_blank(),
          panel.border = element_rect(colour = "black", fill = NA),
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
    )
)

colors_10 <- c(
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
)
catscale10 <- scale_colour_manual(values = colors_10)
catscale10_2 <- scale_fill_manual(values = colors_10)

# ------------------------------------------------
# Entropy and Predictability
wp <- read.csv("final/wp-150/wp-dataset-basic.csv")
bk <- read.csv("final/bk-150/bk-dataset-basic.csv")
gw <- read.csv("final/gws-150/gws-dataset-basic.csv")

wp <- na.omit(wp)
bk <- na.omit(bk)
gw <- na.omit(gw)

wp$dataset <- "Weeplaces"
bk$dataset <- "BrightKite"
gw$dataset <- "Gowalla"
bk[, 1] <- sapply(bk[, 1], as.factor)
gw[, 1] <- sapply(gw[, 1], as.factor)

df <- do.call("rbind", list(wp, bk, gw))


df_entropy <- df %>% select(userid, Shannon.Entropy, LZ.Entropy, dataset)
df_entropy_melt <- melt(df_entropy,
                        id.vars = c("userid", "dataset")
)

df_predictability <- df %>% select(userid, Shannon.Pi, LZ.Pi, dataset)
df_predictability_melt <- melt(df_predictability,
                               id.vars = c("userid", "dataset")
)


p_ent <- ggplot(df_entropy[df_entropy$dataset=='Weeplaces', ], 
                aes(x = LZ.Entropy)) +
  geom_density(size=2, alpha=.4, color=colors_10[8],
  ) +
  catscale10+
  # scale_color_manual(labels=c('A', 'B'), values = colors_10[1:2]) +
  labs(x = unname(TeX(c("Entropy  $\\hat{S}_A$ (bits)"))))

print(p_ent)

p_pred <- ggplot(df_predictability[df_predictability$dataset=='Weeplaces',], 
                 aes(x = LZ.Pi)) +
  geom_density(size=2, alpha=.4, color=colors_10[8],
  ) +
  catscale10+
  scale_x_continuous(labels = scales::percent) +
  # scale_color_manual(labels=c('A', 'B'), values = colors_10[1:2]) +
  labs(x = unname(TeX(c("Predictability $\\Pi_{A}$"))))

print(p_pred)

# -----------------------
# social ties vs colocator, CE and CP

NSCLN<- read.csv('final/wp-150/non_social_CLN_network_details_H.csv')
SRN<- read.csv('final/wp-150/SRN_network_details_H.csv')

common <- intersect(NSCLN$userid_x, SRN$userid_x)

NSCLN$Pi_alters_ratio <- NSCLN$Pi_alters / NSCLN$Pi
SRN$Pi_alters_ratio <- SRN$Pi_alters / SRN$Pi

SRN_1 <- SRN %>% filter(userid_x %in% common, Rank == '1') 
SRN_1$category <- 'Top social tie'

NSCLN_1 <- NSCLN %>% filter(userid_x %in% common, Rank == '1') 
NSCLN_1$category <- 'Top non-social co-locator'

NSCLN_3 <- NSCLN %>% filter(userid_x %in% common, Rank == '3') 
NSCLN_3$category <- 'Top 3 non-social co-locators'

df_compare <- do.call("rbind", list(SRN_1, NSCLN_1, NSCLN_3)) 
# df_compare$Pi_alters_ratio <- df_compare$Pi_alters / df_compare$Pi

p_CE <- ggplot(df_compare, 
                aes(x = CCE_alters)) +
  geom_density(size=2, aes(color=category),show.legend=FALSE)+
  stat_density(aes(colour=category), size=2,
               geom="line",position="identity") +
  scale_colour_manual(values = c(colors_10[3], colors_10[1], colors_10[2])) +
  theme(
    legend.position = c(0.3, 0.85),
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    strip.background = element_blank(),
    panel.border = element_rect(colour = "black", fill = NA),
    legend.title = element_blank(),
    legend.spacing.x = unit(0.5, "char"),
    strip.text = element_text(size = 15),
    legend.text = element_text(
      face = "bold",
      size = 10
    )
        # plot.title=element_text(face='bold', size=12,hjust = 0.5)
  ) + 
  # scale_color_manual(labels=c('A', 'B'), values = colors_10[1:2]) +
  labs(x = unname(TeX(c("Entropy  $\\hat{S}_A$ (bits)"))))
print(p_CE)


p_CP <- ggplot(df_compare, 
               aes(x = Pi_alters)) +
  geom_density(size=2, aes(color=category),,show.legend=FALSE)+
  stat_density(aes(colour=category), size=2,
               geom="line",position="identity") +
  scale_colour_manual(values = c(colors_10[3], colors_10[1], colors_10[2])) +
  theme(
    legend.position = c(0.72, 0.85),
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    strip.background = element_blank(),
    panel.border = element_rect(colour = "black", fill = NA),
    legend.title = element_blank(),
    legend.spacing.x = unit(0.5, "char"),
    strip.text = element_text(size = 15),
    legend.text = element_text(
      face = "bold",
      size = 10
    )
    # plot.title=element_text(face='bold', size=12,hjust = 0.5)
  ) + 
  scale_x_continuous(labels = scales::percent) +
  # scale_color_manual(labels=c('A', 'B'), values = colors_10[1:2]) +
  labs(x = unname(TeX(c("Predictability $\\Pi_{A}$"))))
print(p_CP)


#-------

top_alter <-inner_join(SRN_1, NSCLN_1, by="userid_x")
ggplot(top_alter, aes(x=Pi_alters.x, y=Pi_alters.y)) +
  geom_point(size=2)



SRN$Rank <- as.numeric(SRN$Rank)
NSCLN$Rank <- as.numeric(NSCLN$Rank)

# top10_alters <- inner_join(x=SRN %>% filter(userid_x %in% common, Rank <=10) , 
#                            y=NSCLN %>% filter(userid_x %in% common, Rank <=10), 
#                            by=c("userid_x", "Rank")
# )

top10_alters <- inner_join(x=SRN %>% filter(userid_x %in% common, Rank ==1) , 
                           y=NSCLN %>% filter(userid_x %in% common, Rank ==3), 
                           by=c("userid_x")
)


# top10_alters$Rank <- as.factor(top10_alters$Rank)

p_scatter_pred <- ggplot(top10_alters, aes(x=Pi_alters.x, y=Pi_alters.y)) +
  geom_point(size=2, color=colors_10[5]) +
  theme(
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    panel.border = element_rect(colour = "black", fill = NA),
    legend.title = element_blank(),
    legend.text = element_text(
      face = "bold",
      size = 10
    ),
    axis.text = element_text(
      face = "bold",
      size = 12
    ),
    axis.title = element_text(
      face = "bold",
      size = 12
    )
    ) +
  scale_x_continuous(labels = scales::percent) +
  scale_y_continuous(labels = scales::percent) +
  labs(x = 'Predictability: Top Social tie', 
       y='Predictability: Top 3 Non-social co-locators') + 
  geom_abline(intercept =0, slope = 1, size=1.2)

print(p_scatter_pred)


p_scatter_ent <- ggplot(top10_alters, aes(x=CCE_alters.x, y=CCE_alters.y)) +
  geom_point(size=2, color=colors_10[5]) +
  theme(
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    panel.border = element_rect(colour = "black", fill = NA),
    legend.title = element_blank(),
    legend.text = element_text(
      face = "bold",
      size = 10
    ),
    axis.text = element_text(
      face = "bold",
      size = 12
    ),
    axis.title = element_text(
      face = "bold",
      size = 12
    )
  ) +
  labs(x = 'Entropy: Top Social tie', 
       y='Entropy: Top 3 Non-social co-locators') + 
  geom_abline(intercept =0, slope = 1, size=1.2)

print(p_scatter_ent)

#---------------------------------

wp_stats <- read.csv("final/extra/wp_stats_non_social_vs_social.csv")
wp_stats$Rank <- as.factor(wp_stats$Rank)


p_CCE_alters <- ggplot(wp_stats, aes(x=Rank, y=mean_CCE_alters, 
                                  # shape=category,
                                  color=category)) + 
  geom_point(size=3) +
  theme(
    legend.position = c(0.6,0.87),
    legend.title = element_blank(),
    # legend.position = "none",
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
  ) +  catscale10 + 
  geom_errorbar(aes(ymin=lower_CCE_alters, ymax=upper_CCE_alters), 
                width=.2,
                position=position_dodge(0.05)) + 
  labs(x = "Included number of alters", 
       y = unname(TeX(c("$\\hat{S}_{ego|alters}")))
  )

print(p_CCE_alters)


p_Pi_alters <- ggplot(wp_stats, aes(x=Rank, y=mean_Pi_alters, 
                                     # shape=category,
                                     color=category)) + 
  geom_point(size=3) +
  theme(
    legend.position = c(0.6,0.17),
    legend.title = element_blank(),
    # legend.position = "none",
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
  ) +  catscale10 + 
  scale_y_continuous(labels = scales::percent) + 
  geom_errorbar(aes(ymin=lower_Pi_alters, ymax=upper_Pi_alters), 
                width=.2,
                position=position_dodge(0.05)) + 
  labs(x = "Included number of alters", 
       y = unname(TeX(c("$\\Phi_{ego|alters}")))
  )

print(p_Pi_alters)



#---------------------------------

ggarrange(
  p_ent, p_CE, p_scatter_ent, p_CCE_alters,
  p_pred, p_CP, p_scatter_pred, p_Pi_alters,
  labels = c("A", "C", "E", "G", "B", "D", "F", "H"), nrow = 2, ncol = 4
)

ggsave(
  filename = "combined_vs.pdf", 
  device = "pdf",
  width = 20, height = 8,
  path = "fig/"
)

