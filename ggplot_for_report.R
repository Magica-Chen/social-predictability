library("data.table")
library("plyr")
library("dplyr")
library("ggplot2")
library("reshape2")
library("latex2exp")
library("boot")

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

#-----------------------------------------------
# plot only for H-MFN
final <- read.csv('final/150_all_H_MFN.csv')
final$included <- as.factor(final$included)

ggplot(final, aes(x=included, y=mean, color=I("black"))) + 
  geom_point() +
  theme(
    legend.title = element_blank(),
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
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2,
                position=position_dodge(0.05)) + 
  facet_wrap(~dataset, scales = "free") + 
  labs(x = "Included number of alters", 
       y = unname(TeX(c("$\\Pi_{alters}/ \\Pi_{ego}")))
      )

ggsave(
  filename = "H_MFN_relative_Pi.pdf", device = "pdf",
  width = 9.90, height = 2.66,
  path = "fig/"
)

#-----plot only for all categories-------
all_final <- read.csv('final/150_all_category.csv')
all_final$included <- as.factor(all_final$included)

ggplot(all_final, aes(x=included, y=mean, 
                      # shape=category,
                      color=category)) + 
  geom_point(size=3) +
  theme(
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
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2,
                position=position_dodge(0.05)) + 
  facet_wrap(~dataset, scales = "free") + 
  labs(x = "Included number of alters", 
       y = unname(TeX(c("$\\Pi_{alters}/ \\Pi_{ego}")))
  )

ggsave(
  filename = "ALL_relative_Pi.pdf", device = "pdf",
  width = 9.90, height = 2.66,
  path = "fig/"
)

#------plot user Jaccard similarity-----------------
wp_vip_sim <- read.csv('final/wp-150/wp_VIP_similarity_user.csv')
wp_vip_sim$dataset <- 'Weeplace'

bk_vip_sim <- read.csv('final/bk-150/bk_VIP_similarity_user.csv')
bk_vip_sim$dataset <- 'BrightKite'
  
gw_vip_sim <- read.csv('final/gws-150/gws_VIP_similarity_user.csv')
gw_vip_sim$dataset <- 'Gowalla'

df_vip_sim <- do.call("rbind", list(wp_vip_sim, bk_vip_sim, gw_vip_sim))


ggplot(df_vip_sim, aes(x=Compare, y=Jaccard, 
                      # shape=category,
                      fill=Compare
                      # color=Compare,
                      )) + 
  geom_boxplot() +
  theme(
    legend.title = element_blank(),
    axis.text.x = element_text(angle = 60, vjust = 1, hjust=1),
    legend.position = "none",
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
  ) +  catscale10  + catscale10_2 + 
  facet_wrap(~dataset, nrow = 3, 
             strip.position = 'right',
             scales = "free_y") + 
  labs(x='', y='Jaccard Similarity')

ggsave(
  filename = "VIP_similarity_user.pdf", device = "pdf",
  width = 10.5, height = 8.7,
  path = "fig/"
)


#----plot user USLR and SLR--------
vip_LR <- read.csv('final/150_all_LR.csv')
vip_LR$included <- as.factor(vip_LR$included)

ggplot(vip_LR, aes(x=included, y=mean, 
                      # shape=category,
                      color=category)) + 
  geom_point(size=3) +
  theme(
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
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2,
                position=position_dodge(0.05)) + 
  facet_grid(LR~dataset, 
             scales = "free_y") + 
  labs(x ='Rank', y='')

ggsave(
  filename = "VIP_LR.pdf", device = "pdf",
  width = 9.02, height = 4.15,
  path = "fig/"
)


#-------plot for CV----------------------

vip_CV <- read.csv('final/150_all_CV.csv')
vip_CV$included <- as.factor(vip_CV$included)

ggplot(vip_CV, aes(x=included, y=mean, 
                   # shape=category,
                   color=category)) + 
  geom_point(size=3) +
  theme(
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
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2,
                position=position_dodge(0.05)) + 
  facet_wrap(~dataset, 
             scales = "free_y") + 
  labs(x = "Included number of alters", 
       y = unname(TeX(c("$\\Pi_{alters}/ \\Pi_{ego}")))
       )

ggsave(
  filename = "VIP_CV.pdf", device = "pdf",
  width = 9.90, height = 2.66,
  path = "fig/"
)
