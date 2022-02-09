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


## -----Compare 150 vs 75 in WP and BZR---------

BRZ_stats_150 <- read.csv("Brazil/no_call_150/BZR_reciprocal_30_stats_no_call_history_vs_social.csv")
BRZ_stats_75 <- read.csv('Brazil/no_call_75/BZR_reciprocal_30_75_stats_non_social_vs_social.csv')
BRZ_stats_150$threshold <- '150'
BRZ_stats_75$threshold <- '75'
BRZ_stats <- do.call("rbind", list(BRZ_stats_150, BRZ_stats_75))
BRZ_stats$dataset <- 'Mobile Phone'


wp_stats_150 <- read.csv("final/extra/wp_stats_non_social_vs_social.csv")
wp_stats_75 <- read.csv('wp-75/wp_stats_non_social_vs_social_75.csv')
wp_stats_150$threshold <- '150'
wp_stats_75$threshold <- '75'
wp_stats <- do.call("rbind", list(wp_stats_150, wp_stats_75))
wp_stats$dataset <- 'Weeplaces'


df_stats <- do.call("rbind", list(wp_stats, BRZ_stats))

df_stats$category[df_stats$category=="non-social co-location network"]<-"Non-social colocator(s)"
df_stats$category[df_stats$category=="social network"]<-"Social tie(s)"

df_stats$Rank <- as.factor(df_stats$Rank)

df_stats_alters <- df_stats %>% select(Rank, category, 
                                       lower_CCE_alters, mean_CCE_alters, upper_CCE_alters,
                                       lower_Pi_alters, mean_Pi_alters, upper_Pi_alters,
                                       threshold, dataset
) %>% 
  rename(lower_CCE = lower_CCE_alters, lower_Pi = lower_Pi_alters,
         mean_CCE = mean_CCE_alters, mean_Pi = mean_Pi_alters,
         upper_CCE = upper_CCE_alters, upper_Pi = upper_Pi_alters
  )
df_stats_alters$type <- 'Alter(s) only'

df_stats_alters$category <- factor(df_stats_alters$category, 
                                    level=c('Social tie(s)',
                                            'Non-social colocator(s)'))
df_stats_alters$dataset <- factor(df_stats_alters$dataset, 
                                   level=c('Weeplaces',
                                           'Mobile Phone')) 
# df_stats_ego_alters <- df_stats %>% select(Rank, category, 
#                                            lower_CCE_ego_alters, mean_CCE_ego_alters, upper_CCE_ego_alters,
#                                            lower_Pi_ego_alters, mean_Pi_ego_alters, upper_Pi_ego_alters,
#                                            threshold, dataset
# ) %>% 
#   rename(lower_CCE = lower_CCE_ego_alters, lower_Pi = lower_Pi_ego_alters, 
#          mean_CCE = mean_CCE_ego_alters, mean_Pi = mean_Pi_ego_alters,
#          upper_CCE = upper_CCE_ego_alters, upper_Pi = upper_Pi_ego_alters
#   ) 
# df_stats_ego_alters$type <- 'Alter(s) and ego'
# 
# 
# df_stats_details <- do.call("rbind", list(df_stats_alters, df_stats_ego_alters))
# df_stats_details$category <- factor(df_stats_details$category, 
#                                     level=c('Social tie(s)',
#                                             'Non-social colocator(s)'))

p_CCE_alters <- ggplot(df_stats_alters, aes(x=Rank, y=mean_CCE, 
                                             shape=category, color=threshold)) + 
  geom_point(size=3)  + 
  theme(
    # legend.box="vertical", legend.margin=margin(),
    # legend.position = 'top',
    # legend.title = element_blank(),
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
  facet_wrap(~dataset, scales = 'free_y')+ 
  geom_errorbar(aes(ymin=lower_CCE, ymax=upper_CCE), 
                width=.2,
                position=position_dodge(0.05)) + 
  labs(x = "Included number of alters", 
       y = "Cumulative Cross-Entropy"
  )

print(p_CCE_alters)

p_Pi_alters <- ggplot(df_stats_alters, aes(x=Rank, y=mean_Pi, 
                                            shape=category, color=threshold)) + 
  geom_point(size=3) + 
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
  facet_wrap(~dataset, scales = 'free_y')+ 
  scale_y_continuous(labels = scales::percent) + 
  geom_errorbar(aes(ymin=lower_Pi, ymax=upper_Pi), 
                width=.2,
                position=position_dodge(0.05)) + 
  labs(x = "Included number of alters", 
       y = 'Cumulative Cross-Predictability'
  )

print(p_Pi_alters)

ggarrange(
  p_CCE_alters, p_Pi_alters, labels = c("A", "B"), nrow = 2, ncol = 1,
  common.legend = TRUE, legend = "top"
)

ggsave(
  filename = "WP_VS_BRZ_150_VS_75.pdf", device = "pdf",
  width = 8.5, height =8,
  path = "fig/"
)

## -----Fig 6--Time Lag (CCP vs Time Lag)-----------------------

BZR_time_lag <- read.csv('final/extra/BZR_common_time_lag.csv')
BZR_time_lag$interval <- as.numeric(BZR_time_lag$interval)/2

p_time_lag <- ggplot(BZR_time_lag %>% filter(type=='Pi'), 
                     aes(x=interval, 
                         y=mean)) + 
  geom_point(size=2, color = colors_10[2]) +
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
  ) + scale_colour_manual(values = colors_10[2]) + 
  scale_y_continuous(labels = scales::percent) +
  scale_x_continuous(breaks = c(1,6,12,18,24)) +
  facet_wrap(~Rank, nrow = 2, ncol = 5) + 
  geom_errorbar(aes(ymin=lower, ymax=upper), 
                width=.2,
                # color = colors_10[2],
                position=position_dodge(0.05)) + 
  labs(x = "Time lag (hour)", 
       y = '(Cumulative) Cross-Predictability'
  )

print(p_time_lag)


ggsave(
  filename = "BZR_time_lag_effec.pdf", 
  device = "pdf",
  width = 8, height = 4,
  path = "fig/"
)

## -----Fig S21 -- Time Lag (CCP vs Rank)-----------------------

time_lag_Pi <- BZR_time_lag %>% filter(type=='Pi', 
                                      interval %in% c(0.5, 3, 6, 12))
time_lag_Pi$Interval <- as.factor(time_lag_Pi$interval)
time_lag_Pi$Rank <- as.factor(time_lag_Pi$Rank)

p_time_lag_rank <- ggplot(time_lag_Pi, 
                          aes(x=Rank,
                              fill=Interval,
                              y=mean)) + 
  geom_bar(position=position_dodge(), stat="identity", colour='black') +
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
  ) +  catscale10  + catscale10_2 + 
  scale_y_continuous(labels = scales::percent) +
  # scale_x_continuous(breaks = c(1,6,12,18,24)) +
  geom_errorbar(aes(ymin=lower, ymax=upper), 
                width=.2,
                # color = colors_10[2],
                position=position_dodge(1)) + 
  labs(x = "Included number of alters", 
       y = '(Cumulative) Cross-Predictability'
  )

print(p_time_lag_rank)
ggsave(
  filename = "BZR_time_lag_rank_effec.pdf", 
  device = "pdf",
  width = 4.5, height = 3.5,
  path = "fig/"
)


## -----Fig S21 -- Time Lag (CoDLR vs Rank)-----------------------

time_lag_CODLR <- BZR_time_lag %>% filter(type=='CODLR', 
                                         interval %in% c(0.5, 3, 6, 12))
time_lag_CODLR$interval <- as.factor(time_lag_CODLR$interval)
time_lag_CODLR$Rank <- as.factor(time_lag_CODLR$Rank)

p_time_lag_CODLR <- ggplot(time_lag_CODLR, 
                           aes(x=Rank,
                               fill=interval,
                               y=mean)) + 
  geom_bar(position=position_dodge(), stat="identity", colour='black') +
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
  ) +  catscale10  + catscale10_2 + 
  scale_y_continuous(labels = scales::percent) +
  # scale_x_continuous(breaks = c(1,6,12,18,24)) +
  geom_errorbar(aes(ymin=lower, ymax=upper), 
                width=.2,
                # color = colors_10[2],
                position=position_dodge(1)) + 
  labs(x = "Included number of alters", 
       y = 'CODLR'
  )

print(p_time_lag_CODLR)
ggsave(
  filename = "BZR_time_lag_CODLR_effec.pdf", 
  device = "pdf",
  width = 4.5, height = 3.5,
  path = "fig/"
)



## ---------FinalData LZ entropy and Predictability------------

wp <- read.csv("final/extra/wp_common_top10_details.csv")
bk <- read.csv("final/extra/bk_common_top10_details.csv")
gw <- read.csv("final/extra/gw_common_top10_details.csv")
bzr <- read.csv("Brazil/no_call_150/BZR_reciprocal_30_no_call_common_top10_details.csv")

wp$dataset <- "Weeplaces"
bk$dataset <- "BrightKite"
gw$dataset <- "Gowalla"
bzr$dataset <- 'Mobile Phone'

df <- do.call("rbind", list(wp, bk, gw, bzr))



df$dataset <- factor(df$dataset, 
                             level=c('BrightKite',
                                     'Gowalla',
                                     'Weeplaces',
                                     'Mobile Phone'))

p_ent <- ggplot(df, aes(x = LZ_entropy)) +
  geom_histogram(aes(fill = dataset),
                 alpha = 0.7, position = "identity", bins = 100
  ) +
  scale_fill_manual(
    values = colors_10[1:4]
  ) +
  # scale_color_manual(labels=c('A', 'B'), values = colors_10[1:2]) +
  labs(x = unname(TeX(c("Entropy  $\\hat{S}_A$ (bits)"))))
print(p_ent)

p_pred <- ggplot(df, aes(x = Pi)) +
  geom_histogram(aes(fill = dataset),
                 alpha = 0.7, position = "identity", bins = 100
  ) +
  scale_fill_manual(
    values = colors_10[1:4]
  ) +
  # scale_color_manual(labels=c('A', 'B'), values = colors_10[1:2]) +
  labs(x = unname(TeX(c("Predictability $\\Pi_{A}$"))))
print(p_pred)

ggarrange(
  p_ent, p_pred, labels = c("A", "B"), nrow = 1, ncol = 2,
  common.legend = TRUE, legend = "top"
)


ggsave(
  filename = "Pi_hist_dataset_revised_final.pdf", device = "pdf",
  width = 5.50, height =2.5,
  path = "fig/"
)


## --------Shannon vs LZ--------------------
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


df_entropy_melt$dataset <- factor(df_entropy_melt$dataset, 
                             level=c('BrightKite',
                                     'Gowalla',
                                     'Weeplaces',
                                     'Mobile Phone'))
df_predictability_melt$dataset <- factor(df_predictability_melt$dataset, 
                                  level=c('BrightKite',
                                          'Gowalla',
                                          'Weeplaces',
                                          'Mobile Phone'))

p_ent <- ggplot(df_entropy_melt, aes(x = value)) +
  geom_histogram(aes(fill = variable),
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
    labels = unname(TeX(c("$\\hat{S}_{SN}", "$\\hat{S}_{LZ}"))),
    values = colors_10[1:2]
  ) +
  # scale_color_manual(labels=c('A', 'B'), values = colors_10[1:2]) +
  facet_wrap(~dataset, nrow = 1) +
  labs(x = "Entropy (bit)")

print(p_ent)

ggsave(
  filename = "Shannon_VS_LZ_Ent.pdf", device = "pdf",
  width = 7, height =3,
  path = "fig/"
)



p_pred <- ggplot(df_predictability_melt, aes(x = value)) +
  geom_histogram(aes(fill = variable),
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
    labels = unname(TeX(c("$\\Pi_{SN}", "$\\Pi_{LZ}"))),
    values = colors_10[3:4]
  ) +
  facet_wrap(~dataset, nrow = 1) +
  labs(x = "Predictability")

ggarrange(
  p_ent, p_pred, labels = c("A", "B"), nrow = 2, ncol = 1,
  common.legend = FALSE, legend = "right"
)

ggsave(
  filename = "Shannon_VS_LZ_Ent_Pi.pdf", device = "pdf",
  width = 12.5, height =5.5,
  path = "fig/"
)




##---------CCP vs Rank --BZR and WEEPLACES

f <- function(wp_stats){
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
  return(wp_stats_details)
}

wp_stats_with0 <- read.csv("final/extra/wp_stats_non_social_vs_social_no_filter_with0.csv")
wp_stats_details_with0 <- f(wp_stats_with0)
bzr_stats_with0 <- read.csv("final/extra/BZR_stats_no_call_30_vs_reciprocal_30_no_filter_with0.csv")
bzr_stats_details_with0 <- f(bzr_stats_with0)

wp_stats_details_with0$dataset <- 'Weeplaces'
bzr_stats_details_with0$dataset <- 'Mobile Phone'

df_stats_details <- do.call("rbind", list(wp_stats_details_with0, 
                                        bzr_stats_details_with0))

df_stats_details$dataset <- factor(df_stats_details$dataset, 
                                level=c('Weeplaces',
                                        'Mobile Phone')) 

ggplot(df_stats_details, 
       aes(x=Rank, 
           y=mean_Pi,
           shape=type,color=category)) + 
  geom_point(size=3) + 
  # geom_hline(yintercept = mean(wp_10alters$Pi.x),
  #                                 size =1) + 
  theme(
    # legend.box="vertical", legend.margin=margin(),
    legend.position = 'bottom',
    strip.text = element_text(
      face = "bold",
      size = 14),
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
  facet_wrap(~dataset, scales = 'free_y') + 
  scale_y_continuous(labels = scales::percent) + 
  geom_errorbar(aes(ymin=lower_Pi, ymax=upper_Pi), 
                width=.2,
                position=position_dodge(0.05)) + 
  labs(x = "Included number of alters", 
       y = 'Cumulative Cross-Predictability'
  )

ggsave(
  filename = "CCP_filter_vs_no_filter.pdf",
  device = "pdf",
  width = 8.8, height = 4.5,
  path = "fig"
)




## -------CCE/CCP vs Rank---no filter (wp) --------------
wp_stats_without0 <- read.csv("final/extra/wp_stats_non_social_vs_social_no_filter_without0.csv")
wp_stats_with0 <- read.csv("final/extra/wp_stats_non_social_vs_social_no_filter_with0.csv")
f <- function(wp_stats){
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
  return(wp_stats_details)
}

wp_stats_details_without0 <- f(wp_stats_without0)
wp_stats_details_with0 <- f(wp_stats_with0)

wp_stats_details_without0$fig <- "A"
wp_stats_details_with0$fig <- "B"

wp_stats_details <- do.call("rbind", list(wp_stats_details_without0, 
                                          wp_stats_details_with0))

p_Pi_alters <- ggplot(wp_stats_details, aes(x=Rank, y=mean_Pi, 
                      shape=type,color=category)) + 
  geom_point(size=3) + 
  # geom_hline(yintercept = mean(wp_10alters$Pi.x),
  #                                 size =1) + 
  theme(
    # legend.box="vertical", legend.margin=margin(),
    legend.position = 'bottom',
    # legend.title = element_blank(),
    strip.text = element_text(
      face = "bold",
      size = 14),
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
  facet_wrap(~fig) + 
  scale_y_continuous(labels = scales::percent) + 
  geom_errorbar(aes(ymin=lower_Pi, ymax=upper_Pi), 
                width=.2,
                position=position_dodge(0.05)) + 
  labs(x = "Included number of alters", 
       y = 'Cumulative Cross-Predictability'
  )

print(p_Pi_alters)
# mean(wp_10alters$Pi.x)


ggsave(
  filename = "wp_CCP_filter_vs_no_filter.pdf",
  device = "pdf",
  width = 8.8, height = 5,
  path = "fig"
)


## -------CCE/CCP vs Rank---no filter (BZR)-------------------
bzr_stats_without0 <- read.csv("final/extra/BZR_stats_no_call_30_vs_reciprocal_30_no_filter_without0.csv")
bzr_stats_with0 <- read.csv("final/extra/BZR_stats_no_call_30_vs_reciprocal_30_no_filter_with0.csv")

bzr_stats_details_without0 <- f(bzr_stats_without0)
bzr_stats_details_with0 <- f(bzr_stats_with0)

bzr_stats_details_without0$fig <- "A"
bzr_stats_details_with0$fig <- "B"

bzr_stats_details <- do.call("rbind", list(bzr_stats_details_without0, 
                                          bzr_stats_details_with0))



bzr_Pi_alters <- ggplot(bzr_stats_details, 
                                 aes(x=Rank, 
                                     y=mean_Pi,
                                     shape=type,color=category)) + 
  geom_point(size=3) + 
  # geom_hline(yintercept = mean(wp_10alters$Pi.x),
  #                                 size =1) + 
  theme(
    # legend.box="vertical", legend.margin=margin(),
    legend.position = 'bottom',
    strip.text = element_text(
      face = "bold",
      size = 14),
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
  facet_wrap(~fig) + 
  scale_y_continuous(labels = scales::percent) + 
  geom_errorbar(aes(ymin=lower_Pi, ymax=upper_Pi), 
                width=.2,
                position=position_dodge(0.05)) + 
  labs(x = "Included number of alters", 
       y = 'Cumulative Cross-Predictability'
  )

print(bzr_Pi_alters)
# mean(wp_10alters$Pi.x)


ggsave(
  filename = "BZR_CCP_filter_vs_no_filter.pdf",
  device = "pdf",
  width = 8.8, height = 5,
  path = "fig"
)


# --------0 CCP --no filter -------

wp_no_filter <- read.csv('final/extra/wp_no_filter_missing_pct.csv')
bzr_no_flter <- read.csv('final/extra/BZR_no_filter_missing_pct.csv')
wp_no_filter$dataset <- 'Weeplaces'
bzr_no_flter$dataset <- "Mobile Phone"

df_no_fileter <- do.call("rbind", list(wp_no_filter, 
                      bzr_no_flter))
df_no_fileter$category[df_no_fileter$category=="non-social co-location network"]<-"Non-social colocator(s)"
df_no_fileter$category[df_no_fileter$category=="social network"]<-"Social tie(s)"

df_no_fileter$category <- factor(df_no_fileter$category, 
                                   level=c('Social tie(s)',
                                           'Non-social colocator(s)'))
df_no_fileter$dataset <- factor(df_no_fileter$dataset, 
                                  level=c('Weeplaces',
                                          'Mobile Phone')) 


df_no_fileter$Rank <- as.factor(df_no_fileter$Rank)

ggplot(df_no_fileter, 
       aes(x=Rank, 
           y=pct,
           fill=category)) + 
  geom_bar(stat="identity", color="black", position=position_dodge()) + 
  # geom_hline(yintercept = mean(wp_10alters$Pi.x),
  #                                 size =1) + 
  theme(
    # legend.box="vertical", legend.margin=margin(),
    legend.position = 'top',
    strip.text = element_text(
      face = "bold",
      size = 14),
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
  ) +  catscale10 + catscale10_2 + 
  facet_wrap(~dataset) + 
  scale_y_continuous(labels = scales::percent) + 
  labs(x = "Included number of alters", 
       y = ''
  )

ggsave(
  filename = "no_filter_pct.pdf",
  device = "pdf",
  width = 7, height = 4,
  path = "fig"
)





## -------Real vs Random NSCN sampling-----------
bk_stats_vs_NSCN <- read.csv('final/extra/bk_stats_NSCN_real_vs_random.csv')
bk_stats_vs_NSCN$dataset <- 'BrightKite'
wp_stats_vs_NSCN <- read.csv('final/extra/wp_stats_NSCN_real_vs_random.csv')
wp_stats_vs_NSCN$dataset <- 'Weeplaces'
gw_stats_vs_NSCN <- read.csv('final/extra/gw_stats_NSCN_real_vs_random.csv')
gw_stats_vs_NSCN$dataset <- 'Gowalla'
bzr_stats_vs_NSCN <- read.csv('final/extra/BZR_stats_NSCN_real_vs_random.csv')
bzr_stats_vs_NSCN$dataset <- 'Mobile Phone'


stats_vs_NSCN <- do.call("rbind", list(bk_stats_vs_NSCN, 
                                       wp_stats_vs_NSCN,
                                       gw_stats_vs_NSCN,
                                       bzr_stats_vs_NSCN)
                         )

stats_vs_NSCN$Rank <- as.factor(stats_vs_NSCN$Rank)
stats_vs_NSCN$dataset<- factor(stats_vs_NSCN$dataset,
                                   level=c('BrightKite',
                                           'Gowalla',
                                           'Weeplaces',
                                           'Mobile Phone')
                                   )

stats_vs_NSCN_alters <- stats_vs_NSCN %>% select(Rank, 
                                       lower_CCE_alters, mean_CCE_alters, upper_CCE_alters,
                                       lower_Pi_alters, mean_Pi_alters, upper_Pi_alters,
                                       trajectory, dataset
) %>% 
  rename(lower_CCE = lower_CCE_alters, lower_Pi = lower_Pi_alters,
         mean_CCE = mean_CCE_alters, mean_Pi = mean_Pi_alters,
         upper_CCE = upper_CCE_alters, upper_Pi = upper_Pi_alters
  )


stats_vs_NSCN_ego_alters <- stats_vs_NSCN %>% select(Rank, 
                                                 lower_CCE_ego_alters, mean_CCE_ego_alters, upper_CCE_ego_alters,
                                                 lower_Pi_ego_alters, mean_Pi_ego_alters, upper_Pi_ego_alters,
                                                 trajectory, dataset
) %>% 
  rename(lower_CCE = lower_CCE_ego_alters, lower_Pi = lower_Pi_ego_alters,
         mean_CCE = mean_CCE_ego_alters, mean_Pi = mean_Pi_ego_alters,
         upper_CCE = upper_CCE_ego_alters, upper_Pi = upper_Pi_ego_alters
  )

p_CCE_vs_NSCN_alters <- ggplot(stats_vs_NSCN_alters, aes(x=Rank, y=mean_CCE, 
                                            color=trajectory)) + 
  geom_point(size=3)  + 
  theme(
    # legend.box="vertical", legend.margin=margin(),
    # legend.position = 'top',
    # legend.title = element_blank(),
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
  facet_wrap(~dataset, scales = 'free_y', nrow = 1)+ 
  geom_errorbar(aes(ymin=lower_CCE, ymax=upper_CCE), 
                width=.2,
                position=position_dodge(0.05)) + 
  labs(x = "Included number of alters in non-social colocation network", 
       y = "CCE"
  )

print(p_CCE_vs_NSCN_alters)


p_Pi_vs_NSCN_alters <- ggplot(stats_vs_NSCN_alters, aes(x=Rank, y=mean_Pi, 
                                                    color=trajectory)) + 
  geom_point(size=3) + 
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
  facet_wrap(~dataset, scales = 'free_y', nrow=1)+ 
  scale_y_continuous(labels = scales::percent) + 
  geom_errorbar(aes(ymin=lower_Pi, ymax=upper_Pi), 
                width=.2,
                position=position_dodge(0.05)) + 
  labs(x = "Included number of alters in non-social colocation network", 
       y = 'CCP'
  )
print(p_Pi_vs_NSCN_alters)


ggarrange(
  p_CCE_vs_NSCN_alters, p_Pi_vs_NSCN_alters, 
  labels = c("A", "B"), nrow = 2, ncol = 1,
  common.legend = TRUE, legend = "top"
)

ggsave(
  filename = "real_vs_random_NSCN.pdf", device = "pdf",
  width = 10, height =6,
  path = "fig/"
)

#---------MSCN---ego+alters-----------------
p_CCE_vs_NSCN_ego_alters <- ggplot(stats_vs_NSCN_ego_alters, 
                                   aes(x=Rank, 
                                       y=mean_CCE, 
                                       color=trajectory)) + 
  geom_point(size=3)  + 
  theme(
    # legend.box="vertical", legend.margin=margin(),
    # legend.position = 'top',
    # legend.title = element_blank(),
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
  facet_wrap(~dataset, scales = 'free_y', nrow = 1)+ 
  geom_errorbar(aes(ymin=lower_CCE, ymax=upper_CCE), 
                width=.2,
                position=position_dodge(0.05)) + 
  labs(x = "Included number of alters in non-social colocation network", 
       y = "CCE (ego + alters)"
  )

print(p_CCE_vs_NSCN_ego_alters)


p_Pi_vs_NSCN_ego_alters <- ggplot(stats_vs_NSCN_ego_alters, aes(x=Rank, y=mean_Pi, 
                                                        color=trajectory)) + 
  geom_point(size=3) + 
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
  facet_wrap(~dataset, scales = 'free_y', nrow=1)+ 
  scale_y_continuous(labels = scales::percent) + 
  geom_errorbar(aes(ymin=lower_Pi, ymax=upper_Pi), 
                width=.2,
                position=position_dodge(0.05)) + 
  labs(x = "Included number of alters in non-social colocation network", 
       y = 'CCP(ego+alters)'
  )
print(p_Pi_vs_NSCN_ego_alters)


ggarrange(
  p_CCE_vs_NSCN_ego_alters, p_Pi_vs_NSCN_ego_alters, 
  labels = c("A", "B"), nrow = 2, ncol = 1,
  common.legend = TRUE, legend = "top"
)

ggsave(
  filename = "real_vs_random_NSCN_ego_alters.pdf", device = "pdf",
  width = 10, height =6,
  path = "fig/"
)


## -------Real vs Random SRN sampling-----------
bk_stats_vs_SRN <- read.csv('final/extra/bk_stats_SRN_real_vs_random.csv')
bk_stats_vs_SRN$dataset <- 'BrightKite'
wp_stats_vs_SRN <- read.csv('final/extra/wp_stats_SRN_real_vs_random.csv')
wp_stats_vs_SRN$dataset <- 'Weeplaces'
gw_stats_vs_SRN <- read.csv('final/extra/gw_stats_SRN_real_vs_random.csv')
gw_stats_vs_SRN$dataset <- 'Gowalla'
bzr_stats_vs_SRN <- read.csv('final/extra/BZR_stats_SRN_real_vs_random.csv')
bzr_stats_vs_SRN$dataset <- 'Mobile Phone'


stats_vs_SRN <- do.call("rbind", list(bk_stats_vs_SRN, 
                                       wp_stats_vs_SRN,
                                       gw_stats_vs_SRN,
                                       bzr_stats_vs_SRN)
)

stats_vs_SRN$Rank <- as.factor(stats_vs_SRN$Rank)
stats_vs_SRN$dataset<- factor(stats_vs_SRN$dataset,
                               level=c('BrightKite',
                                       'Gowalla',
                                       'Weeplaces',
                                       'Mobile Phone')
)

stats_vs_SRN_alters <- stats_vs_SRN %>% select(Rank, 
                                                 lower_CCE_alters, mean_CCE_alters, upper_CCE_alters,
                                                 lower_Pi_alters, mean_Pi_alters, upper_Pi_alters,
                                                 trajectory, dataset
) %>% 
  rename(lower_CCE = lower_CCE_alters, lower_Pi = lower_Pi_alters,
         mean_CCE = mean_CCE_alters, mean_Pi = mean_Pi_alters,
         upper_CCE = upper_CCE_alters, upper_Pi = upper_Pi_alters
  )

stats_vs_SRN_ego_alters <- stats_vs_SRN %>% select(Rank, 
                                               lower_CCE_ego_alters, mean_CCE_ego_alters, upper_CCE_ego_alters,
                                               lower_Pi_ego_alters, mean_Pi_ego_alters, upper_Pi_ego_alters,
                                               trajectory, dataset
) %>% 
  rename(lower_CCE = lower_CCE_ego_alters, lower_Pi = lower_Pi_ego_alters,
         mean_CCE = mean_CCE_ego_alters, mean_Pi = mean_Pi_ego_alters,
         upper_CCE = upper_CCE_ego_alters, upper_Pi = upper_Pi_ego_alters
  )

p_CCE_vs_SRN_alters <- ggplot(stats_vs_SRN_alters, aes(x=Rank, y=mean_CCE, 
                                                    color=trajectory)) + 
  geom_point(size=3)  + 
  theme(
    # legend.box="vertical", legend.margin=margin(),
    # legend.position = 'top',
    # legend.title = element_blank(),
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
  facet_wrap(~dataset, scales = 'free_y', nrow = 1)+ 
  geom_errorbar(aes(ymin=lower_CCE, ymax=upper_CCE), 
                width=.2,
                position=position_dodge(0.05)) + 
  labs(x = "Included number of alters in social network", 
       y = "CCE"
  )

print(p_CCE_vs_SRN_alters)


p_Pi_vs_SRN_alters <- ggplot(stats_vs_SRN_alters, aes(x=Rank, y=mean_Pi, 
                                                   color=trajectory)) + 
  geom_point(size=3) + 
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
  facet_wrap(~dataset, scales = 'free_y', nrow=1)+ 
  scale_y_continuous(labels = scales::percent) + 
  geom_errorbar(aes(ymin=lower_Pi, ymax=upper_Pi), 
                width=.2,
                position=position_dodge(0.05)) + 
  labs(x = "Included number of alters in social network", 
       y = 'CCP'
  )
print(p_Pi_vs_SRN_alters)


ggarrange(
  p_CCE_vs_SRN_alters, p_Pi_vs_SRN_alters, labels = c("A", "B"), nrow = 2, ncol = 1,
  common.legend = TRUE, legend = "top"
)

ggsave(
  filename = "real_vs_random_SRN.pdf", device = "pdf",
  width = 10, height =6,
  path = "fig/"
)

#-------SRN, ego+alter------------------------

p_CCE_vs_SRN_ego_alters <- ggplot(stats_vs_SRN_ego_alters, 
                                  aes(x=Rank, 
                                      y=mean_CCE, 
                                      color=trajectory)
                                  ) + 
  geom_point(size=3)  + 
  theme(
    # legend.box="vertical", legend.margin=margin(),
    # legend.position = 'top',
    # legend.title = element_blank(),
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
  facet_wrap(~dataset, scales = 'free_y', nrow = 1)+ 
  geom_errorbar(aes(ymin=lower_CCE, ymax=upper_CCE), 
                width=.2,
                position=position_dodge(0.05)) + 
  labs(x = "Included number of alters in social network", 
       y = "CCE (ego+alters)"
  )

print(p_CCE_vs_SRN_ego_alters)


p_Pi_vs_SRN_ego_alters <- ggplot(stats_vs_SRN_ego_alters, 
                                 aes(x=Rank, 
                                     y=mean_Pi, 
                                     color=trajectory)) + 
  geom_point(size=3) + 
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
  facet_wrap(~dataset, scales = 'free_y', nrow=1)+ 
  scale_y_continuous(labels = scales::percent) + 
  geom_errorbar(aes(ymin=lower_Pi, ymax=upper_Pi), 
                width=.2,
                position=position_dodge(0.05)) + 
  labs(x = "Included number of alters in social network", 
       y = 'CCP (ego+alters)'
  )
print(p_Pi_vs_SRN_ego_alters)


ggarrange(
  p_CCE_vs_SRN_ego_alters, p_Pi_vs_SRN_ego_alters, 
  labels = c("A", "B"), nrow = 2, ncol = 1,
  common.legend = TRUE, legend = "top"
)

ggsave(
  filename = "real_vs_random_SRN_ego_alters.pdf", device = "pdf",
  width = 10, height =6,
  path = "fig/"
)


##--------bar plot for statistical tests----------------

p_rate_wp <- read.csv('final/extra/wp_less_test.csv')
p_rate_bk <- read.csv('final/extra/bk_less_test.csv')
p_rate_gw <- read.csv('final/extra/gw_less_test.csv')
p_rate_bzr <- read.csv('final/extra/bzr_less_test.csv')



p_rate <- do.call("rbind", list(p_rate_wp, 
                               p_rate_bk,
                               p_rate_gw,
                               p_rate_bzr)
)

p_rate$Rank <- as.factor(p_rate$Rank)
p_rate$dataset<- factor(p_rate$dataset,
                               level=c('BrightKite',
                                       'Gowalla',
                                       'Weeplaces',
                                       'Mobile Phone')
)

p_rate$network[p_rate$network=="Non-social colocation network"]<-"Non-social colocator(s)"
p_rate$network[p_rate$network=="Social network"]<-"Social tie(s)"
p_rate$network <- factor(p_rate$network, 
                         level=c('Social tie(s)',
                                 'Non-social colocator(s)'))


fig_bar005 <- ggplot(p_rate %>% subset(alpha==0.05), 
       aes(x=Rank,
           fill=network,
           y=Percentage)) + 
  geom_bar(position=position_dodge(), stat="identity", colour='black') +
  theme(
    # legend.box="vertical", legend.margin=margin(),
    legend.position = 'top',
    legend.title = element_blank(),
    # legend.position = "none",
    strip.text = element_text(size = 12),
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
  ) +  catscale10  + catscale10_2 + 
  facet_wrap(~dataset, nrow=1) + 
  scale_y_continuous(labels = scales::percent) +
  # scale_x_continuous(breaks = c(1,6,12,18,24)) +
  labs(x = "Included number of alters", 
       y = 'Pct'
  )

print(fig_bar005)

fig_bar001 <- ggplot(p_rate %>% subset(alpha==0.01), 
                     aes(x=Rank,
                         fill=network,
                         y=Percentage)) + 
  geom_bar(position=position_dodge(), stat="identity", colour='black') +
  theme(
    # legend.box="vertical", legend.margin=margin(),
    legend.position = 'top',
    legend.title = element_blank(),
    # legend.position = "none",
    strip.text = element_text(size = 12),
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
  ) +  catscale10  + catscale10_2 + 
  facet_wrap(~dataset, nrow=1) + 
  scale_y_continuous(labels = scales::percent) +
  # scale_x_continuous(breaks = c(1,6,12,18,24)) +
  labs(x = "Included number of alters", 
       y = 'Pct'
  )

print(fig_bar001)


ggarrange(
  fig_bar005, fig_bar001, 
  labels = c("A", "B"), nrow = 2, ncol = 1,
  common.legend = TRUE, legend = "top"
)

ggsave(
  filename = "reject_rate_alpha.pdf", device = "pdf",
  width = 11, height =7,
  path = "fig/"
)

#-------------------------------------------

bzr_KO_vs_NSCN <- read.csv('final/extra/BZR_stats_NSCN_KO_real_vs_random.csv')
bzr_KO_vs_NSCN$network <- 'Non-social colocation network'

bzr_KO_vs_SRN <- read.csv('final/extra/BZR_stats_SRN_KO_real_vs_random.csv')
bzr_KO_vs_SRN$network <- 'Social network'

bzr_KO_stats <- do.call("rbind", list(bzr_KO_vs_NSCN,
                                      bzr_KO_vs_SRN)
  )

bzr_KO_stats$Rank <- as.factor(bzr_KO_stats$Rank)

bzr_KO_stats_alters <- bzr_KO_stats %>% select(Rank, 
                                     lower_CCE_alters, mean_CCE_alters, upper_CCE_alters,
                                     lower_Pi_alters, mean_Pi_alters, upper_Pi_alters,
                                     trajectory, network
) %>% 
  rename(lower_CCE = lower_CCE_alters, lower_Pi = lower_Pi_alters,
         mean_CCE = mean_CCE_alters, mean_Pi = mean_Pi_alters,
         upper_CCE = upper_CCE_alters, upper_Pi = upper_Pi_alters
  )

bzr_KO_stats_ego_alters <- bzr_KO_stats %>% select(Rank, 
                                     lower_CCE_ego_alters, mean_CCE_ego_alters, upper_CCE_ego_alters,
                                     lower_Pi_ego_alters, mean_Pi_ego_alters, upper_Pi_ego_alters,
                                     trajectory, network
) %>% 
  rename(lower_CCE = lower_CCE_ego_alters, lower_Pi = lower_Pi_ego_alters,
         mean_CCE = mean_CCE_ego_alters, mean_Pi = mean_Pi_ego_alters,
         upper_CCE = upper_CCE_ego_alters, upper_Pi = upper_Pi_ego_alters
  )

p_bzr_KO_stats_alters <- ggplot(bzr_KO_stats_alters, 
                                aes(x=Rank, y=mean_Pi, 
                                    color=trajectory)) + 
  geom_point(size=3, position = position_dodge2(w = 0.2))  + 
  theme(
    # legend.box="vertical", legend.margin=margin(),
    # legend.position = 'top',
    # legend.title = element_blank(),
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
  facet_wrap(~network, nrow = 1)+ 
  scale_y_continuous(labels = scales::percent) + 
  geom_errorbar(aes(ymin=lower_Pi, ymax=upper_Pi), 
                width=.2,
                position=position_dodge(0.2)) + 
  labs(x = "Included number of alters", 
       y = 'CCP'
  )

print(p_bzr_KO_stats_alters)

p_bzr_KO_stats_ego_alters <- ggplot(bzr_KO_stats_ego_alters, 
                                    aes(x=Rank, y=mean_Pi, 
                                        color=trajectory)) + 
  geom_point(size=3, position = position_dodge2(w = 0.2)) + 
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
  facet_wrap(~network, nrow=1)+ 
  scale_y_continuous(labels = scales::percent) + 
  geom_errorbar(aes(ymin=lower_Pi, ymax=upper_Pi), 
                width=.2,
                position=position_dodge(0.2)) + 
  labs(x = "Included number of alters", 
       y = 'CCP'
  )
print(p_bzr_KO_stats_ego_alters)

ggarrange(
  p_bzr_KO_stats_alters, p_bzr_KO_stats_ego_alters, 
  labels = c("A", "B"), nrow = 2, ncol = 1,
  common.legend = TRUE, legend = "top"
)

ggsave(
  filename = "real_vs_random_KO.pdf", device = "pdf",
  width = 8.5, height =8,
  path = "fig/"
)


## -----------an example in BZR -------------------
df_BZR_example <- read.csv('final/extra/BZR_NSCN_common_top10_sample100_details.csv')


example1 <- df_BZR_example %>% subset((sample > 0) & 
                          (userid_x == df_BZR_example[1,1])&
                          (Rank ==1) ) %>% select(Pi_alters)



ggplot(example1, aes(x=Pi_alters))+
  geom_histogram(binwidth=0.005) + 
  geom_vline(xintercept=df_BZR_example[1,10],
               color="red", linetype="dashed", size=1)


ggsave(
  filename = "BZR_example_1_1.pdf", device = "pdf",
  width = 4, height =3,
  path = "fig/"
)



summary(example1$Pi_alters)
mean_sd(example1$Pi_alters)
sd(example1$Pi_alters)
