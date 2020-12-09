library("data.table")
library("dplyr")
library("ggplot2")
library("reshape2")
library("latex2exp")
library("magrittr")
library("ggpubr")

colors_10 <- c(
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
)
catscale10 <- scale_colour_manual(values = colors_10)
catscale10_2 <- scale_fill_manual(values = colors_10)


df_all <- read.csv("final/MeetupNp_Rank/150_all_MeetupNp_CODLR_CCP.csv",
                   stringsAsFactors = FALSE)
# -----------ONLY WEEPLACE, RANK=10---------
df_all$category[df_all$category=="CB-1H-MFN"]<-"Co-locationship"
df_all$category[df_all$category=="TFN"]<-"Social Relationship"
Network<- subset(df_all, subset= (Included==10 & 
                                     category%in%c('Co-locationship', 
                                                   'Social Relationship') 
                                   & (dataset=='Weeplace')))

Network$Included <- as.factor(Network$Included)
Network$category <- as.factor(Network$category)


ggscatter(Network, x = "USLR", y = "Pi_alters_ratio", color='category',
          add = "reg.line", conf.int = TRUE,
          cor.coef = TRUE,
          cor.coeff.args = list(method = "pearson", label.x.npc = 0.60, label.y.npc = 0.04),
          xlab = "CODLR",
          ylab = unname(TeX(c("$\\Pi_{alters}/ \\Pi_{ego}")))) +
  theme(
    legend.title = element_blank(),
    strip.text = element_blank()
    # legend.position = "right",
    # strip.text.x = element_text(size = 8),
    # strip.text.y = element_blank(),
  )+
  scale_x_continuous(labels = scales::percent)+
  facet_wrap(~category,
             nrow = 1,
             ncol = 2,
             scales = 'free_y',
             # strip.position="right"
  )

ggsave(
  filename = "VIP_MeetupNp_CODLR_CCP_Weeplace.pdf", device = "pdf",
  width = 5, height = 3,
  path = "fig/"
)


# # ---------ONLY FOCUS on LAST ONE, Rank = 10
# H_MFN <-  subset(df_all, subset= (Included==10 & category%in%c('CB-1H-MFN')))
# TFN <- subset(df_all, subset= (Included==10 & category%in%c('TFN')))
# # focus on both datasets
# H_MFN$Included <- as.factor(H_MFN$Included)
# TFN$Included <- as.factor(TFN$Included)
# 
# g1 <- ggscatter(H_MFN, x = "USLR", y = "Pi_alters_ratio", color='dataset',
#           add = "reg.line", conf.int = TRUE,
#           cor.coef = TRUE,
#           cor.coeff.args = list(method = "pearson", label.x.npc = 0.60, label.y.npc = 0.04),
#           xlab = "CODLR",
#           ylab = unname(TeX(c("$\\Pi_{alters}/ \\Pi_{ego}")))) +
#   theme(
#     legend.title = element_blank(),
#     # legend.position = "none",
#     strip.text.x = element_text(size = 8),
#     strip.text.y = element_blank(),
#   )+
#   scale_x_continuous(labels = scales::percent)+
#   facet_wrap(~dataset,
#              nrow = 1,
#              ncol = 3,
#              scales = 'free_y',
#              # strip.position="right"
#   )
# print(g1)
# 
# g2 <- ggscatter(TFN, x = "USLR", y = "Pi_alters_ratio", color='dataset',
#                 add = "reg.line", conf.int = TRUE,
#                 cor.coef = TRUE, 
#                 cor.coeff.args = list(method = "pearson", label.x.npc = 0.60, label.y.npc = 0.04),
#                 xlab = "CODLR",
#                 ylab = unname(TeX(c("$\\Pi_{alters}/ \\Pi_{ego}")))) +
#   theme(
#     legend.title = element_blank(),
#     # legend.position = "none",
#     strip.text.x = element_text(size = 8),
#     strip.text.y = element_blank(),
#   )+
#   scale_x_continuous(labels = scales::percent)+
#   facet_wrap(~dataset,
#              nrow = 1,
#              ncol = 3,
#              scales = 'free_y',
#              # strip.position="right"
#   )
# 
# print(g2)
# 
# ggarrange(
#   g1, g2, labels = c("A", "B"), nrow = 2, ncol = 1,
#   common.legend = TRUE, legend = "top"
# )
# 
# 
# ggsave(
#   filename = "VIP_MeetupNp_CODLR_CCP_Combined.pdf", device = "pdf",
#   width = 7, height = 5,
#   path = "fig/"  
# )


# # H_MFN <-  subset(df_all, subset= (Included==10 & category=='CB-1H-MFN' & dataset=='Weeplace'))
# # focus on both datasets
# H_MFN <-  subset(df_all, subset= (category=='TFN'))
# H_MFN$Included <- as.factor(H_MFN$Included)
# 
# ggscatter(H_MFN, x = "USLR", y = "Pi_alters_ratio", color='dataset',
#           add = "reg.line", conf.int = TRUE,
#           cor.coef = TRUE, cor.method = "pearson",
#           xlab = "CODLR",
#           ylab = unname(TeX(c("$\\Pi_{alters}/ \\Pi_{ego}")))) +
#   theme(
#     legend.title = element_blank(),
#     # legend.position = "none",
#     strip.text.x = element_text(size = 8),
#     strip.text.y = element_blank(),
#   )+
#   scale_x_continuous(labels = scales::percent)+
#   facet_wrap(~dataset + Included,
#              nrow = 6,
#              ncol = 5,
#              scales = 'free_y',
#              # strip.position="right"
#              )
# 
# ggsave(
#   # filename = "VIP_MeetupNp_CODLR_CCP_Full_H_MFN.pdf",
#   filename = "VIP_MeetupNp_CODLR_CCP_Full_TFN.pdf",
#   device = "pdf",
#   width = 12, height = 15,
#   path = "fig/"
# )

# ## focus on weeplace dataset
# # H_MFN <-  subset(df_all, subset= (category=='CB-1H-MFN' & dataset=='Weeplace'))
# H_MFN <-  subset(df_all, subset= (category=='TFN' & dataset=='Weeplace'))
# H_MFN$Included <- as.factor(H_MFN$Included)
# 
# ggscatter(H_MFN, x = "USLR", y = "Pi_alters_ratio", color = colors_10[2],
#           add = "reg.line", conf.int = TRUE, 
#           cor.coef = TRUE, cor.method = "pearson",
#           xlab = "CODLR", 
#           ylab = unname(TeX(c("$\\Pi_{alters}/ \\Pi_{ego}")))) +
#   theme(
#     legend.title = element_blank(),
#     # legend.position = "none",
#     # strip.text.x = element_text(size = 8),
#     # strip.text = element_blank(),
#   )+   
#   scale_x_continuous(labels = scales::percent)+
#   facet_wrap(~Included, 
#              nrow = 3,
#              ncol = 4,
#              scales = 'free_y',
#              # strip.position="right"
#   )
# 
# ggsave(
#   # filename = "VIP_MeetupNp_CODLR_CCP_wp_H_MFN.pdf", 
#   filename = "VIP_MeetupNp_CODLR_CCP_wp_TFN.pdf", 
#   device = "pdf",
#   width = 9.39, height = 6.24,
#   path = "fig/"
# )