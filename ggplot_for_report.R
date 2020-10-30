library("data.table")
library("plyr")
library("dplyr")
library("ggplot2")
library("reshape2")
library("latex2exp")
library("boot")
library("scales")
library("magrittr")

# ---------Global settings------------
theme_set(
  theme_bw() +
    theme(legend.position = "top")
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
all_final <- read.csv('final/MeetupNp_Rank/150_all_category_MeetupNp.csv', 
                      stringsAsFactors = FALSE)
# all_final <- read.csv('final/FreqNp_Rank/150_all_category_FreqNp.csv')

# # # only for part -----------------------------------
# all_final <- all_final %>% filter(category %in% c("CB-1H-MFN", 'TFN'))
# all_final$category[all_final$category=="CB-1H-MFN"]<-"Co-locationship"
# all_final$category[all_final$category=="TFN"]<-"Social relationship"
# only for all --------------------------------------
all_final$category[all_final$category=="CB-1H-MFN"]<-"CB-1H-CN"
all_final$category[all_final$category=="CB-1D-MFN"]<-"CB-1D-CN"
all_final$category[all_final$category=="SW-24H-MFN"]<-"SW-24H-CN"
all_final$category[all_final$category=="TFN"]<-"Social relationship"

all_final$category %<>% factor(levels= c("CB-1H-CN","Social relationship",
                                         "CB-1D-CN","SW-24H-CN"))
##--------------------------------------------------
all_final$category<- as.factor(all_final$category)
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
  scale_y_continuous(labels = scales::percent) + 
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2,
                position=position_dodge(0.05)) + 
  facet_wrap(~dataset, scales = "free") + 
  labs(x = "Included number of alters", 
       y = unname(TeX(c("$\\Pi_{alters}/ \\Pi_{ego}")))
  )

ggsave(
  # filename = "ALL_relative_Pi_CP.pdf", 
  filename = "ALL_relative_Pi_MeetupNp.pdf",
  # filename = "ALL_relative_Pi_FreqNp.pdf", 
  # filename = "ALL_relative_Pi_MeetupNp_part.pdf",
  device = "pdf",
  width = 9, height = 3.2,
  path = "fig/"
)

## CCE plot for all categories
ggplot(all_final, aes(x=included, y=mean_CCE, 
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
  scale_y_continuous(labels = scales::percent) + 
  geom_errorbar(aes(ymin=lower_CCE, ymax=upper_CCE), width=.2,
                position=position_dodge(0.05)) + 
  facet_wrap(~dataset, scales = "free") + 
  labs(x = "Included number of alters", 
       y = unname(TeX(c("$S_{alters}/ \\S_{ego}")))
  )

ggsave(
  # filename = "ALL_relative_CCE_CP.pdf", 
  filename = "ALL_relative_CCE_MeetupNp.pdf",
  # filename = "ALL_relative_CCE_FreqNp.pdf", 
  # filename = "ALL_relative_CCE_MeetupNp_part.pdf",
  device = "pdf",
  width = 9, height = 3.2,
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

wp_vip_sim <- read.csv('final/wp-150/wp_VIP_similarity_user_MeetupNp.csv', 
                       stringsAsFactors = FALSE)
wp_vip_sim$dataset <- 'Weeplace'

bk_vip_sim <- read.csv('final/bk-150/bk_VIP_similarity_user_MeetupNp.csv', 
                       stringsAsFactors = FALSE)
bk_vip_sim$dataset <- 'BrightKite'

gw_vip_sim <- read.csv('final/gws-150/gws_VIP_similarity_user_MeetupNp.csv', 
                       stringsAsFactors = FALSE)
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

# only for part ----------------------------------
# df_vip_sim <- df_vip_sim %>% filter(Compare == "CB-1H-MFN vs TFN")
# ggplot(df_vip_sim, aes(x=dataset, y=Jaccard,
#                        # shape=category,
#                        fill=dataset
#                        # color=Compare,
# )) +
#   geom_boxplot(width=0.5) +
#   theme(
#     legend.title = element_blank(),
#     axis.text.x = element_text(vjust = 0.5, hjust=0.5),
#     legend.position = "none",
#     # plot.title=element_text(face='bold', size=12,hjust = 0.5)
#   ) +  catscale10  + catscale10_2  +
#   labs(x='', y='Local Jaccard Similarity')
# 
# ggsave(
#   filename = "VIP_similarity_user_MeetupNp_part.pdf", 
#   device = "pdf",
#   width = 2.45, height = 2.32,
#   path = "fig/"
# )
# 
# 
# dataset = c('BrightKite', 'Gowalla', 'Weeplace')
# sim = c(0.48, 0.53, 0.5)
# df_vip_sim_network <- data.frame(dataset, sim )
# ggplot(df_vip_sim_network, aes(x=dataset, y=sim, fill=dataset)) +
#   geom_bar(stat="identity", width=0.5) + 
#   theme(
#     legend.title = element_blank(),
#     axis.text.x = element_text(vjust = 0.5, hjust=0.5),
#     legend.position = "none",
#     # plot.title=element_text(face='bold', size=12,hjust = 0.5)
#   ) +  catscale10  + catscale10_2  +
#   labs(x='', y='Global Jaccard Similarity')
# 
# ggsave(
#   filename = "VIP_similarity_MeetupNp_part.pdf", 
#   device = "pdf",
#   width = 2.45, height = 2.32,
#   path = "fig/"
# )
# 
# Only for all ---------------------------------

df_vip_sim$Compare[df_vip_sim$Compare=="SW-24H-MFN vs CB-1D-MFN"]<-"SW-24H-CN vs CB-1D-CN"
df_vip_sim$Compare[df_vip_sim$Compare=="SW-24H-MFN vs CB-1H-MFN"]<-"SW-24H-CN vs CB-1H-CN"
df_vip_sim$Compare[df_vip_sim$Compare=="SW-24H-MFN vs TFN"]<-"SW-24H-CN vs Social relationship"
df_vip_sim$Compare[df_vip_sim$Compare=="CB-1D-MFN vs CB-1H-MFN"]<-"CB-1D-CN vs CB-1H-CN"
df_vip_sim$Compare[df_vip_sim$Compare=="CB-1D-MFN vs TFN"]<-"CB-1D-CN vs Social relationship"
df_vip_sim$Compare[df_vip_sim$Compare=="CB-1H-MFN vs TFN"]<-"CB-1H-CN vs Social relationship"
df_vip_sim$Compare<- as.factor(df_vip_sim$Compare)

ggplot(df_vip_sim, aes(x=Compare, y=Jaccard,
                      # shape=category,
                      fill=Compare
                      # color=Compare,
                      )) +
  geom_boxplot() +
  theme(
    legend.title = element_blank(),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust=1),
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
  # filename = "VIP_similarity_user_CP.pdf",
  filename = "VIP_similarity_user_MeetupNp.pdf",
  # filename = "VIP_similarity_user_FreqNp.pdf",
  device = "pdf",
  width = 10.5, height = 8.7,
  path = "fig/"
)


#----plot user ODLR--------
# vip_LR <- read.csv('final/150_all_LR_CP.csv')
# vip_LR <- read.csv('final/FreqNp_Rank/150_all_LR_FreqNp.csv')
vip_LR <- read.csv('final/MeetupNp_Rank/150_all_LR_MeetupNp.csv',
                   stringsAsFactors = FALSE)
# # # only for part ---------------------
# vip_LR <- vip_LR %>% filter((LR == "USLR") & (category %in% c("CB-1H-MFN", 'TFN')))
# vip_LR$category[vip_LR$category=="CB-1H-MFN"]<-"Co-locationship"
# vip_LR$category[vip_LR$category=="TFN"]<-"Social relationship"


# only for all ----------------------
vip_LR$category[vip_LR$category=="CB-1H-MFN"]<-"CB-1H-CN"
vip_LR$category[vip_LR$category=="CB-1D-MFN"]<-"CB-1D-CN"
vip_LR$category[vip_LR$category=="SW-24H-MFN"]<-"SW-24H-CN"
vip_LR$category[vip_LR$category=="TFN"]<-"Social relationship"
vip_LR <- vip_LR %>% filter(LR == "USLR")
vip_LR$category %<>% factor(levels= c("CB-1H-CN","Social relationship",
                                       "CB-1D-CN","SW-24H-CN"))

# ----------------------------------------------
vip_LR$included <- as.factor(vip_LR$included)
vip_LR$category<- as.factor(vip_LR$category)


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
  facet_wrap(~dataset, 
             scales = "free_y") +  
  labs(x = "Alter's Rank", 
       y = unname(TeX(c("$\\eta_{ego}(alter)")))
  )

ggsave(
  # filename = "VIP_LR_CP.pdf", 
  # filename = "VIP_LR_MeetupNp.pdf",
  # filename = "VIP_LR_FreqNp.pdf", 
  filename = "VIP_LR_MeetupNp_All.pdf",
  device = "pdf",
  width = 9, height = 3.2,
  path = "fig/"
)

#----plot user cumulative ODLR-------
vip_CLR <- read.csv('final/MeetupNp_Rank/150_all_cumulative_LR_MeetupNp.csv', 
                    stringsAsFactors = FALSE)
# # only for part ---------------------
# vip_CLR <- vip_CLR %>% filter((LR == "USLR") & (category %in% c("CB-1H-MFN", 'TFN')))
# vip_CLR$category[vip_CLR$category=="CB-1H-MFN"]<-"Co-locationship"
# vip_CLR$category[vip_CLR$category=="TFN"]<-"Social relationship"

# only for all ----------------------
vip_CLR$category[vip_CLR$category=="CB-1H-MFN"]<-"CB-1H-CN"
vip_CLR$category[vip_CLR$category=="CB-1D-MFN"]<-"CB-1D-CN"
vip_CLR$category[vip_CLR$category=="SW-24H-MFN"]<-"SW-24H-CN"
vip_CLR$category[vip_CLR$category=="TFN"]<-"Social relationship"
vip_CLR <- vip_CLR %>% filter(LR == "USLR")
vip_CLR$category %<>% factor(levels= c("CB-1H-CN","Social relationship",
                                       "CB-1D-CN","SW-24H-CN"))
# -----------------------------------
vip_CLR$included <- as.factor(vip_CLR$included)
vip_CLR$category<- as.factor(vip_CLR$category)

ggplot(vip_CLR, aes(x=included, y=mean, 
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
  scale_y_continuous(labels = scales::percent) + 
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2,
                position=position_dodge(0.05)) + 
  facet_wrap(~dataset, 
             scales = "free_y") + 
  labs(x = "Included number of alters", 
       y = unname(TeX(c("$\\eta_{ego}(alters)")))
  )

ggsave(
  # filename = "VIP_cumulative_LR_MeetupNp.pdf",
  filename = "VIP_cumulative_LR_MeetupNp_all.pdf",
  device = "pdf",
  width = 9, height = 3.2,
  path = "fig/"
)


#-------plot for CV----------------------

vip_CV <- read.csv('final/MeetupNp_Rank/150_all_CV_MeetupNp.csv', 
                   stringsAsFactors = FALSE)
vip_CV$category[vip_CV$category=="Friendship"]<-"Social relationship"
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
  scale_y_continuous(labels = scales::percent) + 
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2,
                position=position_dodge(0.05)) + 
  facet_wrap(~dataset, 
             scales = "free_y") + 
  labs(x = "Included number of alters", 
       y = unname(TeX(c("$\\Pi_{alters}/ \\Pi_{ego}")))
       )

ggsave(
  filename = "VIP_CV_MeetupNp.pdf", device = "pdf",
  width = 9, height = 3.2,
  path = "fig/"
)
