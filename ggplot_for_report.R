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

#---- plot only for H-MFN--------
# final <- read.csv('final/150_all_2_10_CP_H_MFN.csv')
# final <- read.csv('final/MeetupNp_Rank/150_all_2_10_MeetupNp_H_MFN.csv')

final <- read.csv('final/FreqNp_Rank/150_all_2_10_FreqNp_H_MFN.csv')
final$included <- as.factor(final$included)

ggplot(final, aes(x=included, y=mean, color="black")) + 
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
  facet_wrap(~dataset, scales = 'free') + 
  labs(x = "Included number of alters", 
       y = unname(TeX(c("$\\Pi_{alters}/ \\Pi_{ego}")))
      )

ggsave(
  # filename = "H_MFN_relative_Pi_CP.pdf", 
  # filename = "H_MFN_relative_Pi_MeetupNp.pdf", 
  filename = "H_MFN_relative_Pi_FreqNp.pdf", 
  device = "pdf",
  width = 9.90, height = 2.66,
  path = "fig/"
)

#-----plot only for all categories-------
# all_final <- read.csv('final/150_all_category_CP.csv')
all_final <- read.csv('final/MeetupNp_Rank/150_all_category_MeetupNp.csv', stringsAsFactors = FALSE)
# all_final <- read.csv('final/FreqNp_Rank/150_all_category_FreqNp.csv')
all_final$included <- as.factor(all_final$included)
all_final <- all_final %>% filter(category %in% c("CB-1H-MFN", 'TFN'))

all_final$category[all_final$category=="CB-1H-MFN"]<-"Co-locatorship"
all_final$category[all_final$category=="TFN"]<-"Friendship"
all_final$category<- as.factor(all_final$category)


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
  # filename = "ALL_relative_Pi_CP.pdf", 
  # filename = "ALL_relative_Pi_MeetupNp.pdf", 
  # filename = "ALL_relative_Pi_FreqNp.pdf", 
  filename = "ALL_relative_Pi_MeetupNp_part.pdf",
  device = "pdf",
  width = 9.90, height = 2.66,
  path = "fig/"
)

#------plot user Jaccard similarity-----------------
# wp_vip_sim <- read.csv('final/wp-150/wp_VIP_similarity_user_CP.csv')
# wp_vip_sim$dataset <- 'Weeplace'
# 
# bk_vip_sim <- read.csv('final/bk-150/bk_VIP_similarity_user_CP.csv')
# bk_vip_sim$dataset <- 'BrightKite'
#   
# gw_vip_sim <- read.csv('final/gws-150/gws_VIP_similarity_user_CP.csv')
# gw_vip_sim$dataset <- 'Gowalla'

wp_vip_sim <- read.csv('final/wp-150/wp_VIP_similarity_user_MeetupNp.csv')
wp_vip_sim$dataset <- 'Weeplace'

bk_vip_sim <- read.csv('final/bk-150/bk_VIP_similarity_user_MeetupNp.csv')
bk_vip_sim$dataset <- 'BrightKite'

gw_vip_sim <- read.csv('final/gws-150/gws_VIP_similarity_user_MeetupNp.csv')
gw_vip_sim$dataset <- 'Gowalla'

# wp_vip_sim <- read.csv('final/wp-150/wp_VIP_similarity_user_FreqNp.csv')
# wp_vip_sim$dataset <- 'Weeplace'
# 
# bk_vip_sim <- read.csv('final/bk-150/bk_VIP_similarity_user_FreqNp.csv')
# bk_vip_sim$dataset <- 'BrightKite'
# 
# gw_vip_sim <- read.csv('final/gws-150/gws_VIP_similarity_user_FreqNp.csv')
# gw_vip_sim$dataset <- 'Gowalla'

df_vip_sim <- do.call("rbind", list(wp_vip_sim, bk_vip_sim, gw_vip_sim))
df_vip_sim <- df_vip_sim %>% filter(Compare == "CB-1H-MFN vs TFN")

ggplot(df_vip_sim, aes(x=dataset, y=Jaccard,
                       # shape=category,
                       fill=dataset
                       # color=Compare,
)) +
  geom_boxplot(width=0.5) +
  theme(
    legend.title = element_blank(),
    axis.text.x = element_text(vjust = 0.5, hjust=0.5),
    legend.position = "none",
    # plot.title=element_text(face='bold', size=12,hjust = 0.5)
  ) +  catscale10  + catscale10_2  +
  labs(x='', y='Local Jaccard Similarity')

ggsave(
  filename = "VIP_similarity_user_MeetupNp_part.pdf", 
  device = "pdf",
  width = 2.45, height = 2.32,
  path = "fig/"
)


dataset = c('BrightKite', 'Gowalla', 'Weeplace')
sim = c(0.48, 0.53, 0.5)
df_vip_sim_network <- data.frame(dataset, sim )
ggplot(df_vip_sim_network, aes(x=dataset, y=sim, fill=dataset)) +
  geom_bar(stat="identity", width=0.5) + 
  theme(
    legend.title = element_blank(),
    axis.text.x = element_text(vjust = 0.5, hjust=0.5),
    legend.position = "none",
    # plot.title=element_text(face='bold', size=12,hjust = 0.5)
  ) +  catscale10  + catscale10_2  +
  labs(x='', y='Global Jaccard Similarity')

ggsave(
  filename = "VIP_similarity_MeetupNp_part.pdf", 
  device = "pdf",
  width = 2.45, height = 2.32,
  path = "fig/"
)

# ggplot(df_vip_sim, aes(x=Compare, y=Jaccard,
#                       # shape=category,
#                       fill=Compare
#                       # color=Compare,
#                       )) +
#   geom_boxplot() +
#   theme(
#     legend.title = element_blank(),
#     axis.text.x = element_text(angle = 60, vjust = 1, hjust=1),
#     legend.position = "none",
#     strip.text = element_text(size = 15),
#     legend.text = element_text(
#       face = "bold",
#       size = 12
#     ),
#     axis.text = element_text(
#       face = "bold",
#       size = 12
#     ),
#     axis.title = element_text(
#       face = "bold",
#       size = 12
#     ),
#     # plot.title=element_text(face='bold', size=12,hjust = 0.5)
#   ) +  catscale10  + catscale10_2 +
#   facet_wrap(~dataset, nrow = 3,
#              strip.position = 'right',
#              scales = "free_y") +
#   labs(x='', y='Jaccard Similarity')
# 
# ggsave(
#   # filename = "VIP_similarity_user_CP.pdf",
#   # filename = "VIP_similarity_user_MeetupNp.pdf", 
#   filename = "VIP_similarity_user_FreqNp.pdf", 
#   device = "pdf",
#   width = 10.5, height = 8.7,
#   path = "fig/"
# )


#----plot user USLR and SLR--------
# vip_LR <- read.csv('final/150_all_LR_CP.csv')
# vip_LR <- read.csv('final/MeetupNp_Rank/150_all_LR_MeetupNp.csv')

vip_LR <- read.csv('final/FreqNp_Rank/150_all_LR_FreqNp.csv')
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
  # filename = "VIP_LR_CP.pdf", 
  # filename = "VIP_LR_MeetupNp.pdf", 
  filename = "VIP_LR_FreqNp.pdf", 
  device = "pdf",
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
