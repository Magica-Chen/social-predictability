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

## ----Fig 2: social ties vs colocator, CE and CP (weeplace)-------------
NSCLN_bk<- read.csv('extra/brightkite/non_social_CLN_network_details_H.csv')
SRN_bk<- read.csv('extra/brightkite/SRN_network_details_H.csv')

common_bk <- intersect(NSCLN_bk$userid_x, SRN_bk$userid_x)

NSCLN_bk$Pi_alters_ratio <- NSCLN_bk$Pi_alters / NSCLN_bk$Pi
SRN_bk$Pi_alters_ratio <- SRN_bk$Pi_alters / SRN_bk$Pi


SRN_1 <- SRN_bk %>% filter(userid_x %in% common_bk, Rank == '1') 
SRN_1$category <- 'Top social tie'

NSCLN_1 <- NSCLN_bk %>% filter(userid_x %in% common_bk, Rank == '1') 
NSCLN_1$category <- 'Top non-social colocator'

NSCLN_2 <- NSCLN_bk %>% filter(userid_x %in% common_bk, Rank == '2') 
NSCLN_2$category <- 'Top 2 non-social colocators'

bk_compare <- do.call("rbind", list(SRN_1, NSCLN_1, NSCLN_2)) 
# df_compare$Pi_alters_ratio <- df_compare$Pi_alters / df_compare$Pi

### ---- CE, CP top 1 non-social, social, top 3 non-social, histogram ------
bk_compare$category <- factor(bk_compare$category, 
                              levels = c('Top social tie',
                                         'Top non-social colocator',
                                         'Top 2 non-social colocators'))

signif_CE <- bk_compare %>% group_by(category)%>% 
  summarise(Mean=mean(CCE_alters), Median=median(CCE_alters), Std=sd(CCE_alters))
signif_CP <- bk_compare %>% group_by(category)%>% 
  summarise(Mean=mean(Pi_alters), Median=median(Pi_alters), Std=sd(Pi_alters))

p_CE <- ggplot(bk_compare, 
               aes(x = CCE_alters)) +
  geom_density(size=2, aes(color=category),show.legend=FALSE)+
  stat_density(aes(colour=category), size=2,
               geom="line",position="identity") +
  geom_vline(data=signif_CE, aes(xintercept = Median, color = category),
             size=1, linetype = "dotdash") + 
  catscale10 + 
  theme(
    legend.position = c(0.3, 0.15),
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


p_CP <- ggplot(bk_compare, 
               aes(x = Pi_alters)) +
  geom_density(size=2, aes(color=category),,show.legend=FALSE)+
  stat_density(aes(colour=category), size=2,
               geom="line",position="identity") +
  geom_vline(data=signif_CP, aes(xintercept = Median, color = category),
             size=1, linetype = "dotdash") + 
  catscale10 + 
  theme(
    legend.position = 'None',
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

SRN_bk$Rank <- as.numeric(SRN_bk$Rank)
NSCLN_bk$Rank <- as.numeric(NSCLN_bk$Rank)

bk_10alters <- inner_join(x=SRN_bk %>% filter(userid_x %in% common_bk, Rank <=10) ,
                          y=NSCLN_bk %>% filter(userid_x %in% common_bk, Rank <=10),
                          by=c("userid_x", "Rank")
)

bk_top10_alters <- inner_join(x=SRN_bk %>% filter(userid_x %in% common_bk, Rank ==1) , 
                              y=NSCLN_bk %>% filter(userid_x %in% common_bk, Rank ==2), 
                              by=c("userid_x")
)

p_scatter_pred <- ggplot(bk_top10_alters, aes(x=Pi_alters.x, y=Pi_alters.y)) +
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


p_scatter_ent <- ggplot(bk_top10_alters, aes(x=CCE_alters.x, y=CCE_alters.y)) +
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

bk_stats <- read.csv("final/extra/bk_stats_non_social_vs_social.csv")

bk_stats$category[bk_stats$category=="non-social co-location network"]<-"Non-social colocator(s)"
bk_stats$category[bk_stats$category=="social network"]<-"Social tie(s)"

bk_stats$Rank <- as.factor(bk_stats$Rank)

bk_stats_alters <- bk_stats %>% select(Rank, category, 
                                       lower_CCE_alters, mean_CCE_alters, upper_CCE_alters,
                                       lower_Pi_alters, mean_Pi_alters, upper_Pi_alters
) %>% 
  rename(lower_CCE = lower_CCE_alters, lower_Pi = lower_Pi_alters,
         mean_CCE = mean_CCE_alters, mean_Pi = mean_Pi_alters,
         upper_CCE = upper_CCE_alters, upper_Pi = upper_Pi_alters
  )
bk_stats_alters$type <- 'Alter(s) only'

bk_stats_ego_alters <- bk_stats %>% select(Rank, category, 
                                           lower_CCE_ego_alters, mean_CCE_ego_alters, upper_CCE_ego_alters,
                                           lower_Pi_ego_alters, mean_Pi_ego_alters, upper_Pi_ego_alters,
) %>% 
  rename(lower_CCE = lower_CCE_ego_alters, lower_Pi = lower_Pi_ego_alters, 
         mean_CCE = mean_CCE_ego_alters, mean_Pi = mean_Pi_ego_alters,
         upper_CCE = upper_CCE_ego_alters, upper_Pi = upper_Pi_ego_alters
  ) 
bk_stats_ego_alters$type <- 'Alter(s) and ego'


bk_stats_details <- do.call("rbind", list(bk_stats_alters, bk_stats_ego_alters))
bk_stats_details$category <- factor(bk_stats_details$category, 
                                    level=c('Social tie(s)',
                                            'Non-social colocator(s)'))

p_CCE_alters <- ggplot(bk_stats_details, aes(x=Rank, y=mean_CCE, 
                                             shape=type,color=category)) + 
  geom_point(size=3) + geom_hline(aes(yintercept = mean(bk_10alters$LZ_entropy.x), 
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
mean(bk_10alters$LZ_entropy.x)



p_Pi_alters <- ggplot(bk_stats_details, aes(x=Rank, y=mean_Pi, 
                                            shape=type,color=category)) + 
  geom_point(size=3) + geom_hline(yintercept = mean(bk_10alters$Pi.x),
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
mean(bk_10alters$Pi.x)
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
  filename = "bk_combined_CCP_social_vs_non_social.pdf", 
  device = "pdf",
  width = 15, height = 8,
  path = "fig/"
)


## ------overlap in locations relates to information flow------

NSCLN_bk['category'] <- 'Non-social colocators'
SRN_bk['category'] <- 'Social ties'

compare_bk <- do.call("rbind", list(NSCLN_bk, SRN_bk))

compare_bk_top10 <- subset(compare_bk, subset= (Rank==10))


compare_bk_top10$category <- factor(compare_bk_top10$category, 
                                    level=c('Social ties',
                                            'Non-social colocators'))

ggscatter(compare_bk_top10, x = "CODLR", y = "Pi_alters", color='category',
          add = "reg.line",
          add.params = list(color = "black", 
                            size = 1,
                            fill = "lightgray"),
          conf.int = TRUE, cor.coef = TRUE,
          cor.coeff.args = list(method = "pearson",
                                label.x.npc = 0.35, 
                                label.y.npc = 0.03),
          xlab = unname(TeX(c("$\\eta_{ego}(alters)"))),
          ylab = unname(TeX(c("$\\Pi_{ego|alters}")))) +
  theme(
    legend.title = element_blank(),
    strip.text = element_blank()
    # legend.position = "right",
    # strip.text.x = element_text(size = 8),
    # strip.text.y = element_blank(),
  )+
  scale_x_continuous(labels = scales::percent) +
  scale_y_continuous(labels = scales::percent) +
  catscale10 + catscale10_2 + 
  facet_wrap(~category,
             nrow = 1,
             ncol = 2,
             scales = 'free_y',
             # strip.position="right"
  )

ggsave(
  filename = "bk_rank10_CODLR_CCP_social_vs_non_social.pdf", device = "pdf",
  width =7, height = 4,
  path = "fig/"
)

### ------Focus on 1--10: CODLR vs CCP-----
# social

SRN_top10 <- SRN_bk %>% filter(Rank <=10)
SRN_top10$name <- paste0('Top-', SRN_top10$Rank, ' social ties')
SRN_top10$name[SRN_top10$name == 'Top-10 social ties'] = 'Top-XX social ties'

ggscatter(SRN_top10, 
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
  facet_wrap(~name,
             nrow = 3,
             ncol = 4,
             scales = 'free_y',
             # strip.position="right"
  )

ggsave(
  filename = "bk_all_CODLR_CCP_social.pdf",
  device = "pdf",
  width = 9.39, height = 6.24,
  path = "fig/"
)

# non-social
NSCLN_top10 <- NSCLN_bk %>% filter(Rank <=10)
NSCLN_top10$name <- paste0('Top-', NSCLN_top10$Rank, ' colocators')
NSCLN_top10$name[NSCLN_top10$name == 'Top-10 colocators'] = 'Top-XX colocators'


ggscatter(NSCLN_top10, 
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
  facet_wrap(~name,
             nrow = 3,
             ncol = 4,
             scales = 'free_y',
             # strip.position="right"
  )

ggsave(
  filename = "bk_all_CODLR_CCP_non_social.pdf",
  device = "pdf",
  width = 9.39, height = 6.24,
  path = "fig/"
)