library("data.table")
library("dplyr")
library("ggplot2")
library("reshape2")
library("latex2exp")
library("magrittr")
library("ggpubr")
library("tikzDevice")


## ------------Global Settings-------------------------

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

## ---------Fig 1, entropy and predictability-----------

wp <- read.csv("final/wp-150/wp-dataset-basic.csv")
bk <- read.csv("final/bk-150/bk-dataset-basic.csv")
gw <- read.csv("final/gws-150/gws-dataset-basic.csv")
bzr <- read.csv("final/BZR-150/BZR-dataset-basic.csv")

wp <- na.omit(wp)
bk <- na.omit(bk)
gw <- na.omit(gw)
bzr <- na.omit(bzr)

wp$dataset <- "Weeplaces"
bk$dataset <- "BrightKite"
gw$dataset <- "Gowalla"
bzr$dataset <- 'Mobile Phone'
bk[, 1] <- sapply(bk[, 1], as.factor)
gw[, 1] <- sapply(gw[, 1], as.factor)

df <- do.call("rbind", list(wp, bk, gw, bzr))


df_entropy <- df %>% select(userid, Shannon.Entropy, LZ.Entropy, dataset)
df_entropy_melt <- melt(df_entropy,
                        id.vars = c("userid", "dataset")
)

df_predictability <- df %>% select(userid, Shannon.Pi, LZ.Pi, dataset)
df_predictability_melt <- melt(df_predictability,
                               id.vars = c("userid", "dataset")
)

df_entropy$dataset <- factor(df_entropy$dataset, 
                            level=c('BrightKite',
                                    'Gowalla',
                                    'Weeplaces',
                                    'Mobile Phone'))

orig_order <- levels(df_entropy$dataset)
new_order <- c('Gowalla',
               'Weeplaces',
               'Mobile Phone',
               'BrightKite')
orig_color <- colors_10[1:4]
new_color <- c(colors_10[2:4], colors_10[1])

df_entropy$dataset <- factor(df_entropy$dataset, levels = new_order)

p_ent <- ggplot(df_entropy, aes(x = LZ.Entropy)) +
  geom_histogram(aes(fill = dataset),
                 alpha = 0.5, position = "identity", bins = 100
  ) +
  scale_fill_manual(
    values = orig_color, labels=orig_order, breaks = orig_order
  ) +
  scale_x_continuous(limits = c(0.5, 10.5)) + 
  # scale_color_manual(labels=c('A', 'B'), values = colors_10[1:2]) +
  labs(x = unname(TeX(c("Entropy  $\\hat{S}_A$ (bits)"))))

# df_predictability$dataset <- factor(df_predictability$dataset, 
#                              level=c('BrightKite',
#                                       'Gowalla',
#                                      'Weeplaces',
#                                      'Mobile Phone'))
df_predictability$dataset <- factor(df_predictability$dataset, levels = new_order)

p_pred <- ggplot(df_predictability, aes(x = LZ.Pi)) +
  geom_histogram(aes(fill = dataset),
                 alpha = 0.5, position = "identity", bins = 100
  ) +
  scale_fill_manual(
    values = orig_color, labels = orig_order, breaks = orig_order
  ) +
  scale_x_continuous(limits = c(0.05, 0.95)) + 
  # scale_color_manual(labels=c('A', 'B'), values = colors_10[1:2]) +
  labs(x = unname(TeX(c("Predictability $\\Pi_{A}$"))))


ggarrange(
  p_ent, p_pred, labels = c("A", "B"), nrow = 1, ncol = 2,
  common.legend = TRUE, legend = "top"
)

ggsave(
  filename = "Pi_hist_dataset_revised_05_front.pdf", device = "pdf",
  width = 5.50, height =2.5,
  path = "fig/"
)

## ---------Fig 2: social ties vs colocator, CE and CP (weeplace)-------------
NSCLN_wp<- read.csv('BZR/no_call_30/no_call_CLN_network_details_H.csv')
SRN_wp<- read.csv('BZR/30/BZR_reciprocal_30_SRN_network_details_H.csv')

common_wp <- intersect(NSCLN_wp$userid_x, SRN_wp$userid_x)

NSCLN_wp$Pi_alters_ratio <- NSCLN_wp$Pi_alters / NSCLN_wp$Pi
SRN_wp$Pi_alters_ratio <- SRN_wp$Pi_alters / SRN_wp$Pi


SRN_1 <- SRN_wp %>% filter(userid_x %in% common_wp, Rank == '1') 
SRN_1$category <- 'Top social tie'

NSCLN_1 <- NSCLN_wp %>% filter(userid_x %in% common_wp, Rank == '1') 
NSCLN_1$category <- 'Top non-social colocator'

NSCLN_2 <- NSCLN_wp %>% filter(userid_x %in% common_wp, Rank == '2') 
NSCLN_2$category <- 'Top 2 non-social colocators'

wp_compare <- do.call("rbind", list(SRN_1, NSCLN_1, NSCLN_2)) 
# df_compare$Pi_alters_ratio <- df_compare$Pi_alters / df_compare$Pi

### ---- CE, CP top 1 non-social, social, top 3 non-social, histogram ------
wp_compare$category <- factor(wp_compare$category, 
                              levels = c('Top social tie',
                                         'Top non-social colocator',
                                         'Top 2 non-social colocators'))
signif_CE <- wp_compare %>% group_by(category)%>% 
  summarise(Mean=mean(CCE_alters), Median=median(CCE_alters), Std=sd(CCE_alters))
signif_CP <- wp_compare %>% group_by(category)%>% 
  summarise(Mean=mean(Pi_alters), Median=median(Pi_alters), Std=sd(Pi_alters))

p_CE <- ggplot(wp_compare, 
               aes(x = CCE_alters)) +
  geom_density(size=2, aes(color=category),show.legend=FALSE)+
  stat_density(aes(colour=category), size=2,
               geom="line",position="identity") +
  geom_vline(data=signif_CE, aes(xintercept = Median, color = category),
             size=1, linetype = "dotdash") + 
  catscale10 + 
  theme(
    legend.position = c(0.7, 0.85),
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
    ),
    axis.text = element_text(
      face = "bold",
      size = 14
    ),
    axis.title = element_text(
      face = "bold",
      size = 14
    )
    # plot.title=element_text(face='bold', size=12,hjust = 0.5)
  ) + 
  # scale_color_manual(labels=c('A', 'B'), values = colors_10[1:2]) +
  # labs(x = unname(TeX(c("Cross-Entropy  $\\hat{S}_{A|B}$ (bits)"))))
  labs(x = "(Cumulative) Cross-Entropy", 
       y = "Density")
print(p_CE)


p_CP <- ggplot(wp_compare, 
               aes(x = Pi_alters)) +
  geom_density(size=2, aes(color=category),,show.legend=FALSE)+
  stat_density(aes(colour=category), size=2,
               geom="line",position="identity") +
  geom_vline(data=signif_CP, aes(xintercept = Median, color = category),
             size=1, linetype = "dotdash") + 
  catscale10 + 
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
    ),
    axis.text = element_text(
      face = "bold",
      size = 14
    ),
    axis.title = element_text(
      face = "bold",
      size = 14
    )
    # plot.title=element_text(face='bold', size=12,hjust = 0.5)
  ) + 
  scale_x_continuous(labels = scales::percent) +
  # scale_color_manual(labels=c('A', 'B'), values = colors_10[1:2]) +
  # labs(x = unname(TeX(c("Predictability $\\Pi_{A|B}$"))))
  labs(x = "(Cumulative) Cross-Predictability",
       y = "Density")
print(p_CP)

### -----CE, CP, top-1 social and top 3 non-social, scatter plot-----

SRN_wp$Rank <- as.numeric(SRN_wp$Rank)
NSCLN_wp$Rank <- as.numeric(NSCLN_wp$Rank)

wp_10alters <- inner_join(x=SRN_wp %>% filter(userid_x %in% common_wp, Rank <=10) ,
                          y=NSCLN_wp %>% filter(userid_x %in% common_wp, Rank <=10),
                          by=c("userid_x", "Rank")
)

wp_top10_alters <- inner_join(x=SRN_wp %>% filter(userid_x %in% common_wp, Rank ==1) , 
                              y=NSCLN_wp %>% filter(userid_x %in% common_wp, Rank ==2), 
                              by=c("userid_x")
)

p_scatter_pred <- ggplot(wp_top10_alters, aes(x=Pi_alters.x, y=Pi_alters.y)) +
  geom_point(size=2, color=colors_10[5]) +
  theme(
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    panel.border = element_rect(colour = "black", fill = NA),
    legend.title = element_blank(),
    legend.text = element_text(
      face = "bold",
      size = 12
    ),
    axis.text = element_text(
      face = "bold",
      size = 14
    ),
    axis.title = element_text(
      face = "bold",
      size = 14
    )
  ) +
  scale_x_continuous(labels = scales::percent) +
  scale_y_continuous(labels = scales::percent) +
  labs(x = 'Cross-Predictability', 
       y='Cumulative Cross-Predictability') + 
  geom_abline(intercept =0, slope = 1, size=1.2)

print(p_scatter_pred)


p_scatter_ent <- ggplot(wp_top10_alters, aes(x=CCE_alters.x, y=CCE_alters.y)) +
  geom_point(size=2, color=colors_10[5]) +
  theme(
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    panel.border = element_rect(colour = "black", fill = NA),
    legend.title = element_blank(),
    legend.text = element_text(
      face = "bold",
      size = 12
    ),
    axis.text = element_text(
      face = "bold",
      size = 14
    ),
    axis.title = element_text(
      face = "bold",
      size = 14
    )
  ) +
  labs(x = 'Cross-Entropy', 
       y='Cumulative Cross-Entropy') + 
  geom_abline(intercept =0, slope = 1, size=1.2)

print(p_scatter_ent)

### -------CCE vs Rank plot --------------------------

wp_stats <- read.csv("Brazil/no_call_150/BZR_reciprocal_30_stats_no_call_history_vs_social.csv")

wp_stats$category[wp_stats$category=="non-social co-location network"]<-"Non-social colocator(s)"
wp_stats$category[wp_stats$category=="social network"]<-"Social tie(s)"

wp_stats$Rank <- as.factor(wp_stats$Rank)

wp_stats_alters <- wp_stats %>% select(Rank, category, 
                                       lower_CCE_alters, mean_CCE_alters, upper_CCE_alters,
                                       lower_Pi_alters, mean_Pi_alters, upper_Pi_alters
) %>% 
  rename(lower_CCE = lower_CCE_alters, lower_Pi = lower_Pi_alters,
         mean_CCE = mean_CCE_alters, mean_Pi = mean_Pi_alters,
         upper_CCE = upper_CCE_alters, upper_Pi = upper_Pi_alters
  )
wp_stats_alters$type <- 'Alter(s) only'

wp_stats_ego_alters <- wp_stats %>% select(Rank, category, 
                                           lower_CCE_ego_alters, mean_CCE_ego_alters, upper_CCE_ego_alters,
                                           lower_Pi_ego_alters, mean_Pi_ego_alters, upper_Pi_ego_alters,
) %>% 
  rename(lower_CCE = lower_CCE_ego_alters, lower_Pi = lower_Pi_ego_alters, 
         mean_CCE = mean_CCE_ego_alters, mean_Pi = mean_Pi_ego_alters,
         upper_CCE = upper_CCE_ego_alters, upper_Pi = upper_Pi_ego_alters
  ) 
wp_stats_ego_alters$type <- 'Alter(s) and ego'


wp_stats_details <- do.call("rbind", list(wp_stats_alters, wp_stats_ego_alters))
wp_stats_details$category <- factor(wp_stats_details$category, 
                                    level=c('Social tie(s)',
                                            'Non-social colocator(s)'))

p_CCE_alters <- ggplot(wp_stats_details, aes(x=Rank, y=mean_CCE, 
                                             shape=type,color=category)) + 
  geom_point(size=3) + geom_hline(aes(yintercept = mean(wp_10alters$LZ_entropy.x),
                                      shape='Ego'),
                                  size =1) + 
  theme(
    legend.box="vertical", legend.margin=margin(),
    legend.position = 'top',
    legend.title = element_blank(),
    # legend.position = "none",
    strip.text = element_text(size = 15),
    legend.text = element_text(
      face = "bold",
      size = 14
    ),
    axis.text = element_text(
      face = "bold",
      size = 14
    ),
    axis.title = element_text(
      face = "bold",
      size = 14
    ),
    # plot.title=element_text(face='bold', size=12,hjust = 0.5)
  ) +  catscale10 + 
  geom_errorbar(aes(ymin=lower_CCE, ymax=upper_CCE), 
                width=.2,
                position=position_dodge(0.05)) + 
  labs(x = "Included number of alters", 
       y = "Cumulative Cross-Entropy"
  )

print(p_CCE_alters)
mean(wp_10alters$LZ_entropy.x)



p_Pi_alters <- ggplot(wp_stats_details, aes(x=Rank, y=mean_Pi, 
                                            shape=type,color=category)) + 
  geom_point(size=3) + geom_hline(yintercept = mean(wp_10alters$Pi.x),
                                  size =1) + 
  theme(
    # legend.box="vertical", legend.margin=margin(),
    legend.position = 'None',
    legend.title = element_blank(),
    # legend.position = "none",
    strip.text = element_text(size = 15),
    legend.text = element_text(
      face = "bold",
      size = 14
    ),
    axis.text = element_text(
      face = "bold",
      size = 14
    ),
    axis.title = element_text(
      face = "bold",
      size = 14
    ),
    # plot.title=element_text(face='bold', size=12,hjust = 0.5)
  ) +  catscale10 + 
  scale_y_continuous(labels = scales::percent) + 
  geom_errorbar(aes(ymin=lower_Pi, ymax=upper_Pi), 
                width=.2,
                position=position_dodge(0.05)) + 
  labs(x = "Included number of alters", 
       y = 'Cumulative Cross-Predictability'
  )

print(p_Pi_alters)
mean(wp_10alters$Pi.x)
##########---------------------------------

# ggarrange(
#   p_ent, p_CE, p_scatter_ent, p_CCE_alters,
#   p_pred, p_CP, p_scatter_pred, p_Pi_alters,
#   labels = c("A", "C", "E", "G", "B", "D", "F", "H"), nrow = 2, ncol = 4
# )

ggarrange(
  p_CE, p_scatter_ent, p_CCE_alters,
  p_CP, p_scatter_pred, p_Pi_alters,
  labels = c("A", "C", "E", "B", "D", "F"), nrow = 2, ncol = 3
)


ggsave(
  filename = "BZR_combined_CCP_social_vs_no_call_30.pdf",
  device = "pdf",
  width = 15, height = 8,
  path = "BZR/no_call_30/"
)







## ------Fig 3, Div in how many alters ------

rank_vs_ratio <- read.csv('final/extra/rank_vs_ratio_all.csv')

rank_vs_ratio$N_ties[rank_vs_ratio$N_ties==1]<-"Top 1 Social Tie"
rank_vs_ratio$N_ties[rank_vs_ratio$N_ties==2]<-"Top 2 Social Ties"
rank_vs_ratio$N_ties[rank_vs_ratio$N_ties==3]<-"Top 3 Social Ties"

rank_vs_ratio$Rank <- as.factor(rank_vs_ratio$Rank)
rank_vs_ratio$dataset <- factor(rank_vs_ratio$dataset, 
                             level=c('BrightKite',
                                     'Gowalla',
                                     'Weeplaces',
                                     'Mobile Phone'))

p_rank_ratio1 <- ggplot(rank_vs_ratio %>% filter(Rank==1) , 
                        aes(x=as.factor(N_ties), y=mean_equality_ratio, fill=dataset)) + 
  geom_bar(position=position_dodge(), stat="identity", colour='black') +
  theme(
    # legend.box="vertical", legend.margin=margin(),
    legend.position = c(0.15, 0.8),
    legend.title = element_blank(),
    # legend.position = "none",
    strip.text = element_text(size = 15),
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
      size = 10
    ),
    # plot.title=element_text(face='bold', size=12,hjust = 0.5)
  ) +  catscale10_2 + 
  scale_y_continuous(breaks = seq(1, 6, len = 6)) + 
  geom_errorbar(aes(ymin=lower_equality_ratio, 
                    ymax=upper_equality_ratio), 
                width=.2,
                position=position_dodge(0.9)) + 
  labs(x = " ", 
       y = 'Predictability ratio'
  )

print(p_rank_ratio1)


p_rank_ratio2 <- ggplot(rank_vs_ratio %>% filter(Rank!=1) , aes(x=Rank, y=mean_equality_ratio, 
                                                                color=dataset)) + 
  geom_point(size=3) +geom_hline(yintercept = 1,
                                 size =1) + 
  theme(
    # legend.box="vertical", legend.margin=margin(),
    legend.position = c(0.5, 0.9),
    legend.title = element_blank(),
    # legend.position = "none",
    strip.text = element_text(size = 12),
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
      size = 10
    ),
    # plot.title=element_text(face='bold', size=12,hjust = 0.5)
  ) +  catscale10 + 
  scale_y_continuous(breaks = seq(1, 3, len = 6)) + 
  facet_wrap(~N_ties, scales = 'free_y') + 
  geom_errorbar(aes(ymin=lower_equality_ratio, ymax=upper_equality_ratio), 
                width=.2,
                position=position_dodge(0.05)) + 
  labs(x = "The number of non-social colocators included", 
       y = 'Predictability ratio'
  )

print(p_rank_ratio2)

ggarrange(
  p_rank_ratio1, p_rank_ratio2,
  labels = c("A", "B"), nrow = 2, ncol = 1, common.legend = TRUE, legend = "top"
)


ggsave(
  filename = "combined_ratio_compare_revised.pdf", 
  device = "pdf",
  width = 6.5, height = 4.9,
  path = "fig/"
)

## ------Fig 4, OLDR and COLDR, overlap vs rank------
# this wp_stats contain info of BZR datasets
wp_stats['dataset'] <- 'Mobile Phone'

bk_stats <- read.csv("final/extra/bk_stats_non_social_vs_social.csv")
bk_stats$Rank <- as.factor(bk_stats$Rank)
bk_stats['dataset'] <- 'BrightKite'


bk_stats$category[bk_stats$category=="non-social co-location network"]<-"Non-social colocator(s)"
bk_stats$category[bk_stats$category=="social network"]<-"Social tie(s)"


gw_stats <- read.csv("final/extra/gw_stats_non_social_vs_social.csv")
gw_stats$Rank <- as.factor(gw_stats$Rank)
gw_stats['dataset'] <- 'Gowalla'

gw_stats$category[gw_stats$category=="non-social co-location network"]<-"Non-social colocator(s)"
gw_stats$category[gw_stats$category=="social network"]<-"Social tie(s)"

twp_stats <- read.csv("final/extra/wp_stats_non_social_vs_social.csv")
twp_stats$Rank <- as.factor(twp_stats$Rank)
twp_stats['dataset'] <- 'Weeplaces'

twp_stats$category[twp_stats$category=="non-social co-location network"]<-"Non-social colocator(s)"
twp_stats$category[twp_stats$category=="social network"]<-"Social tie(s)"


df_stats <- do.call("rbind", list(wp_stats, 
                                  bk_stats, 
                                  gw_stats,
                                  twp_stats))

df_stats$category <- factor(df_stats$category, 
                            level=c('Social tie(s)',
                                    'Non-social colocator(s)'))
df_stats$dataset <- factor(df_stats$dataset, 
                             level=c('BrightKite',
                                     'Gowalla',
                                     'Weeplaces',
                                     'Mobile Phone'))


p_ODLR <- ggplot(df_stats, aes(x=Rank, 
                               y=mean_ODLR,
                               color=category)) + 
  geom_point(size=1.5) +
  theme(
    # legend.box="vertical", legend.margin=margin(),
    legend.position = 'top',
    legend.title = element_blank(),
    # legend.position = "none",
    strip.text = element_text(size = 10),
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
  facet_wrap(~dataset, ncol = 4, scales = 'free_y') + 
  geom_errorbar(aes(ymin=lower_ODLR, ymax=upper_ODLR), 
                width=.2,
                position=position_dodge(0.05)) + 
  labs(x = "Rank", 
       y = 'ODLR'
  )

print(p_ODLR)

p_CODLR <- ggplot(df_stats, aes(x=Rank, 
                                y=mean_CODLR,
                                color=category)) + 
  geom_point(size=1.5) +
  theme(
    # legend.box="vertical", legend.margin=margin(),
    legend.position = 'top',
    legend.title = element_blank(),
    # legend.position = "none",
    strip.text = element_text(size = 10),
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
  facet_wrap(~dataset, ncol = 4, scales = 'free_y') + 
  geom_errorbar(aes(ymin=lower_CODLR, ymax=upper_CODLR), 
                width=.2,
                position=position_dodge(0.05)) + 
  labs(x = "Included number of alters", 
       y = 'CODLR'
  )

print(p_CODLR)


ggarrange(
  p_ODLR, p_CODLR,
  labels = c("A", "B"), nrow = 2, ncol = 1, common.legend = TRUE, legend = "top"
)


ggsave(
  filename = "ODLR_CODLR_social_vs_non_social_revised.pdf", 
  device = "pdf",
  width = 8, height = 4.8,
  path = "fig/"
)


## ------Fig S6, rank5 alters CE--------------
NSCLN_wp_full<- read.csv('extra/weeplace/non_social_CLN_network_details_H.csv')
SRN_wp_full<- read.csv('extra/weeplace/SRN_network_details_H.csv')
NSCLN_wp_full$dataset = 'Weeplaces'
SRN_wp_full$dataset = 'Weeplaces'
NSCLN_wp_full$category = 'Non-social co-locator'
SRN_wp_full$category = 'Social tie'

NSCLN_bk_full<- read.csv('extra/brightkite/non_social_CLN_network_details_H.csv')
SRN_bk_full<- read.csv('extra/brightkite/SRN_network_details_H.csv')
NSCLN_bk_full$dataset = 'BrightKite'
SRN_bk_full$dataset = 'BrightKite'
NSCLN_bk_full$category = 'Non-social co-locator'
SRN_bk_full$category = 'Social tie'

NSCLN_gw_full<- read.csv('extra/gowalla/non_social_CLN_network_details_H.csv')
SRN_gw_full<- read.csv('extra/gowalla/SRN_network_details_H.csv')
NSCLN_gw_full$dataset = 'Gowalla'
SRN_gw_full$dataset = 'Gowalla'
NSCLN_gw_full$category = 'Non-social co-locator'
SRN_gw_full$category = 'Social tie'

NSCLN_BZR_full<- read.csv('final/BZR-150/no_call_CLN_network_details_H.csv')
SRN_BZR_full<- read.csv('final/BZR-150/BZR_reciprocal_30_SRN_network_details_H.csv')
NSCLN_BZR_full$dataset = 'Mobile Phone'
SRN_BZR_full$dataset = 'Mobile Phone'
NSCLN_BZR_full$category = 'Non-social co-locator'
SRN_BZR_full$category = 'Social tie'

compare_full<- do.call("rbind", list(NSCLN_wp_full, SRN_wp_full,
                                     NSCLN_bk_full, SRN_bk_full,
                                     NSCLN_gw_full, SRN_gw_full,
                                     NSCLN_BZR_full, SRN_BZR_full
                                     ))
compare_full$category <- factor(compare_full$category, 
                                levels = c('Social tie',
                                           'Non-social co-locator'))
compare_full$dataset <- factor(compare_full$dataset, 
                             level=c('BrightKite',
                                     'Gowalla',
                                     'Weeplaces',
                                     'Mobile Phone'))

compare_full_5 <- compare_full %>% filter(Rank ==5)


p_CE_rank5 <- ggplot(compare_full_5, aes(x = CE_alter)) +
  geom_density(size=1, aes(color=category),,show.legend=FALSE)+
  stat_density(aes(colour=category), size=1.2,
               geom="line",position="identity") +
  # geom_vline(data=signif_CE5, aes(xintercept = Mean, color = dataset),
  #            size=1.5) + 
  # geom_density(aes(fill = dataset, color=dataset),
  #              alpha = 0.6, position = "identity"
  # ) +
  facet_wrap(~dataset, ncol = 4) + 
  scale_color_manual(values = colors_10[1:3]) +
  labs(x = unname(TeX(c("Cross-entropy  $\\hat{S}_{A|B}$ (bits)"))),
       y = "Density")

print(p_CE_rank5)

compare_full_10 <- compare_full %>% filter(Rank ==10)

p_CE_rank10 <- ggplot(compare_full_10 , aes(x = CE_alter)) +
  stat_density(aes(colour=category), size=1.2,
               geom="line",position="identity") +
  # geom_vline(data=signif_CE10, aes(xintercept = Mean, color = dataset),
  #            size=1.5) + 
  # geom_density(aes(fill = dataset, color=dataset),
  #              alpha = 0.6, position = "identity"
  # ) +
  facet_wrap(~dataset, ncol = 4) + 
  scale_color_manual(values = colors_10[1:3]) +
  labs(x = unname(TeX(c("Cross-entropy  $\\hat{S}_{A|B}$ (bits)"))),
       y = "Density")
# xlim(0,15)

print(p_CE_rank10)



ggarrange(
  p_CE_rank5,  p_CE_rank10, 
  labels = c("A", "B"), nrow = 2, ncol = 1,
  common.legend = TRUE, legend = "top"
)


ggsave(
  filename = "Histogram_CE_5_10_social_non_social_revised.pdf", device = "pdf",
  width = 7.5, height =4.8,
  path = "fig/"
)

## ------Fig S12/13/14/15 spatial correlation -Focus on 1--10: CODLR vs CCP-----
# social
ggscatter(SRN_wp %>% filter(Rank <=10), 
          x = "CODLR", y = "Pi_alters", color = colors_10[1],
          add = "reg.line", 
          add.params = list(color = "black", 
                            size = 1,
                            fill = "lightgray"),
          conf.int = TRUE, cor.coef = TRUE,
          cor.coeff.args = list(method = "pearson"),
          xlab = unname(TeX(c("$\\eta_{ego}(alters)"))),
          ylab = unname(TeX(c("$\\Pi_{ego|alters}")))) +
  theme(strip.background = element_blank(),
        panel.border = element_rect(colour = "black", fill = NA),
        legend.title = element_blank(),
        # legend.position = "none",
        # strip.text.x = element_text(size = 8),
        # strip.text = element_blank(),
  )+
  scale_x_continuous(labels = scales::percent)+
  scale_y_continuous(labels = scales::percent)+
  facet_wrap(~Rank,
             nrow = 3,
             ncol = 4,
             scales = 'free_y',
             # strip.position="right"
  )

ggsave(
  filename = "BZR_all_CODLR_CCP_social.pdf",
  device = "pdf",
  width = 9.39, height = 6.24,
  path = "fig/"
)

# non-social

ggscatter(NSCLN_wp %>% filter(Rank <=10), 
          x = "CODLR", y = "Pi_alters", color = colors_10[2],
          add = "reg.line", 
          add.params = list(color = "black", 
                            size = 1,
                            fill = "lightgray"),
          conf.int = TRUE, cor.coef = TRUE,
          cor.coeff.args = list(method = "pearson"),
          xlab = unname(TeX(c("$\\eta_{ego}(alters)"))),
          ylab = unname(TeX(c("$\\Pi_{ego|alters}")))) +
  theme(strip.background = element_blank(),
        panel.border = element_rect(colour = "black", fill = NA),
        legend.title = element_blank(),
        # legend.position = "none",
        # strip.text.x = element_text(size = 8),
        # strip.text = element_blank(),
  )+
  scale_x_continuous(labels = scales::percent)+
  scale_y_continuous(labels = scales::percent)+
  facet_wrap(~Rank,
             nrow = 3,
             ncol = 4,
             scales = 'free_y',
             # strip.position="right"
  )

ggsave(
  filename = "BZR_all_CODLR_CCP_non_social.pdf",
  device = "pdf",
  width = 9.39, height = 6.24,
  path = "fig/"
)


## -----FIg S11, homophily plot predictability of ego vs top social and non-social-----

wp_top_a_Pi <- read.csv('extra/weeplace/non_social_vs_social_predictability.csv')
bk_top_a_Pi <- read.csv('extra/brightkite/non_social_vs_social_predictability.csv')
gw_top_a_Pi <- read.csv('extra/gowalla/non_social_vs_social_predictability.csv')
BZR_top_a_Pi <- read.csv('BZR/no_call_30/no_call_vs_social_predictability.csv')

# wp_top_a_Pi <- read.csv('extra/weeplace/separate_non_social_vs_social_predictability.csv')
# bk_top_a_Pi <- read.csv('extra/brightkite/separate_non_social_vs_social_predictability.csv')
# gw_top_a_Pi <- read.csv('extra/gowalla/separate_non_social_vs_social_predictability.csv')
# BZR_top_a_Pi <- read.csv('BZR/no_call_30/separate_no_call_vs_social_predictability.csv')

wp_top_a_Pi$dataset <- 'Weeplaces' 
bk_top_a_Pi$dataset <- 'BrightKite' 
gw_top_a_Pi$dataset <- 'Gowalla'
BZR_top_a_Pi$dataset <- 'Mobile Phone'

top_a_Pi <- do.call("rbind", list(wp_top_a_Pi, 
                                  bk_top_a_Pi, 
                                  gw_top_a_Pi,
                                  BZR_top_a_Pi))

top_a_Pi$category[top_a_Pi$category=="non-social co-location network"]<-"Non-social co-locator"
top_a_Pi$category[top_a_Pi$category=="social network"]<-"Social tie"

top_a_Pi$category <- factor(top_a_Pi$category, 
                            level=c('Social tie',
                                    'Non-social co-locator'))
top_a_Pi$dataset <- factor(top_a_Pi$dataset, 
                             level=c('BrightKite',
                                     'Gowalla',
                                     'Weeplaces',
                                     'Mobile Phone'))


ggscatter(top_a_Pi, x = "Pi", y = "a_Pi", color='category',shape = 'dataset',
          conf.int = TRUE,
          cor.coef = TRUE,
          cor.coeff.args = list(method = "pearson", label.x.npc = 0.15, label.y.npc = 0.04),
          xlab = unname(TeX(c("$\\Pi_{ego}"))),
          ylab = unname(TeX(c("$\\Pi_{alter}"))), 
          # title = 'Predictability of Ego vs Predictability of Ego\'s top 1 alter'
) +
  geom_abline(intercept =0, slope = 1, show.legend=TRUE, size=1) + 
  # xlim(0.18, 0.9) +
  # ylim(0.25, 0.9) +
  theme(
    legend.title = element_blank(),
    strip.text = element_blank(),
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
    ))+
  catscale10 + catscale10_2 + 
  scale_x_continuous(breaks = c(0.1, 0.3, 0.5, 0.7, 0.9)) +
  # scale_y_continuous(labels = scales::percent) +
  facet_wrap(~category + dataset,
             nrow = 2,
             ncol = 4,
             scales = 'free',
             # strip.position="right"
  )

ggsave(
  filename = "homophily_all_dataset_revised.pdf",
  device = "pdf",
  width = 9.3, height = 5.3,
  path = "fig/"
)


## -----Fig S7 SW-1H vs CB-1H ---------------------------
CB_vs_SW_wp <- read.csv('final/extra/wp_stats_non_social_vs_SW1.csv')
CB_vs_SW_wp$dataset <- 'Weeplaces'
CB_vs_SW_bk <- read.csv('final/extra/bk_stats_non_social_vs_SW1.csv')
CB_vs_SW_bk$dataset <- 'BrightKite'
CB_vs_SW_gw <- read.csv('final/extra/gw_stats_non_social_vs_SW1.csv')
CB_vs_SW_gw$dataset <- 'Gowalla'
CB_vs_SW_BZR <- read.csv('Brazil/no_call_150/BZR_reciprocal_30_stats_no_call_history_vs_SW1.csv')
CB_vs_SW_BZR$dataset <- 'Mobile Phone'

CB_vs_SW <- do.call("rbind", list(CB_vs_SW_wp, 
                                  CB_vs_SW_bk, 
                                  CB_vs_SW_gw,
                                  CB_vs_SW_BZR))


CB_vs_SW$Rank <- as.factor(CB_vs_SW$Rank)
CB_vs_SW$dataset <- factor(CB_vs_SW$dataset, 
                             level=c('BrightKite',
                                     'Gowalla',
                                     'Weeplaces',
                                     'Mobile Phone'))


p_CCP_SW <- ggplot(CB_vs_SW, aes(x=Rank, y=mean_Pi_alters, 
                                 color=category)) + 
  geom_point(size=3) + 
  theme(
    legend.box="vertical", legend.margin=margin(),
    legend.position = 'top',
    legend.title = element_blank(),
    # legend.position = "none",
    strip.text = element_text(size = 15),
    legend.text = element_text(
      face = "bold",
      size = 14
    ),
    axis.text = element_text(
      face = "bold",
      size = 14
    ),
    axis.title = element_text(
      face = "bold",
      size = 14
    ),
    # plot.title=element_text(face='bold', size=12,hjust = 0.5)
  ) +  catscale10 +
  facet_wrap(~dataset,
             ncol = 4,
             scales = 'free_y') + 
  scale_y_continuous(labels = scales::percent) + 
  geom_errorbar(aes(ymin=lower_Pi_alters, ymax=upper_Pi_alters), 
                width=.2,
                position=position_dodge(0.01)) + 
  labs(x = "Included number of alters", 
       y = 'Cumulative Cross-Predictability'
  )

print(p_CCP_SW)

ggsave(
  filename = "CCP_CB_vs_SW_revised.pdf", device = "pdf",
  width = 10.2, height =2.8,
  path = "fig/"
)




## -----Compare 150 vs 75 in WP and BZR---------




