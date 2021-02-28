library("data.table")
library("plyr")
library("dplyr")
library("ggplot2")
library("reshape2")
library("latex2exp")
library("boot")
library("scales")
library("magrittr")
library("ggpubr")



colors_10 <- c(
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
)
catscale10 <- scale_colour_manual(values = colors_10)
catscale10_2 <- scale_fill_manual(values = colors_10)


CE_cut <- read.csv('Kelty/CrossEntropyEgoCutoff.csv')
names(CE_cut) <- c('X', 'X1', 'userid', 
                   'RatioCutOff', 'CE', 'WithCF')
set.seed(2020)
userlist = sample(unique(CE_cut$userid), 150)
CE_cut <- CE_cut%>% filter((CE >= 0) & (userid %in% userlist) & (WithCF=='False'))


p1 <- ggplot(CE_cut, aes(x=RatioCutOff, y=CE, color=userid)) +
  geom_line() +
  theme_bw() +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    strip.background = element_blank(),
    panel.border = element_rect(colour = "black", fill = NA),
    legend.spacing.x = unit(0.5, "char"),
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
  scale_x_continuous(labels = scales::percent) +
  labs(x = "Cutoff percentage of ego's text",
       y = "cross-entropy (bit) at cutoff"
  )
print(p1)


RatioStandard <- read.csv('Kelty/VarianceOfPartitionsConvergence2.csv')
RatioStandard <- na.omit(RatioStandard)
names(RatioStandard) <- c('Category', 'userid', 'CE')

q1 <- ggplot(RatioStandard, aes(x=CE)) + 
  geom_density(aes(fill = Category, color = Category),
               alpha = 0.8, position = "identity") +
  theme_bw() +
  theme(
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    strip.background = element_blank(),
    panel.border = element_rect(colour = "black", fill = NA),
    legend.spacing.x = unit(0.5, "char"),
    legend.title = element_blank(),
    legend.position =  c(0.2, 0.87),
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
    values = colors_10[1:2]
  ) + 
  scale_color_manual(values = colors_10[1:2]) + 
  labs(x = unname(TeX("$log_2 (\\sigma_2 / \\sigma_1)$")), 
       y = 'density')

print(q1)



ggarrange(p1, q1, nrow = 1, labels = c("A", "B"))


ggsave(
  filename = "CE_Convergence_plot.pdf",
  width = 8, height = 3.6,
  path = 'fig/'
)




# CE_cut <- read.csv('CrossEntropyEgoCutoff.csv')
# names(CE_cut) <- c('X', 'X1', 'userid', 
#                    'RatioCutOff', 'CE', 'WithCF')
# set.seed(2020)
# userlist = sample(unique(CE_cut$userid), 150)
# CE_cut <- CE_cut%>% filter((CE >= 0) & (userid %in% userlist))
# 
# levels(CE_cut$WithCF) <- c("Before", "After")
# 
# 
# p1 <- ggplot(CE_cut, aes(x=RatioCutOff, y=CE, color=userid)) + 
#   geom_line() + 
#   theme_bw() +
#   theme(
#     panel.grid.major = element_blank(), 
#     panel.grid.minor = element_blank(),
#     strip.background = element_blank(),
#     panel.border = element_rect(colour = "black", fill = NA),
#     legend.spacing.x = unit(0.5, "char"),
#     legend.title = element_blank(),
#     legend.position = "none",
#     strip.text = element_text(size = 15),
#     axis.text = element_text(
#       face = "bold",
#       size = 12
#     ),
#     axis.title = element_text(
#       face = "bold",
#       size = 12
#     ),
#     # plot.title=element_text(face='bold', size=12,hjust = 0.5)
#   ) + 
#   scale_x_continuous(labels = scales::percent) + 
#   facet_wrap(~WithCF) + 
#   labs(x = "Cutoff percentage of ego's text", 
#        y = "cross-entropy (bit) at cutoff"
#   )
# print(p1)


# -----------ROC---------

# ROC <- read.csv('Kelty/RateOfChangeofCEofFinal20Percent.csv')
# names(ROC) <- c('X', 'Before', 'After')
# ROC_melt <- melt(ROC,
#                  id.vars = c("X")
# )
# 
# p2 <- ggplot(ROC_melt, aes(x=value)) + 
#   geom_histogram(alpha = 0.7, position = "identity", binwidth = 0.01) + 
#   theme_bw() +
#   theme(
#     panel.grid.major = element_blank(), 
#     panel.grid.minor = element_blank(),
#     strip.background = element_blank(),
#     panel.border = element_rect(colour = "black", fill = NA),
#     legend.spacing.x = unit(0.5, "char"),
#     legend.position = "none",
#     strip.text = element_text(size = 15),
#     axis.text = element_text(
#       face = "bold",
#       size = 12
#     ),
#     axis.title = element_text(
#       face = "bold",
#       size = 12
#     ),
#     # plot.title=element_text(face='bold', size=12,hjust = 0.5)
#   ) + 
#   facet_wrap(~variable, scales = "free") + 
#   labs(x = "RoC of cross-entropy over final 20%", 
#        y = 'density')
# 
# print(p2)

# Ratio standard -----

# RatioStandard <- read.csv('Kelty/VarianceOfPartitionsConvergence.csv')
# names(RatioStandard) <- c('Category', 'userid', 'CE')
# 
# q1 <- ggplot(RatioStandard, aes(x=CE)) + 
#   geom_density(aes(fill = Category, color = Category),
#                  alpha = 0.8, position = "identity") +
#   theme_bw() +
#   theme(
#     panel.grid.major = element_blank(), 
#     panel.grid.minor = element_blank(),
#     strip.background = element_blank(),
#     panel.border = element_rect(colour = "black", fill = NA),
#     legend.spacing.x = unit(0.5, "char"),
#     legend.title = element_blank(),
#     legend.position =  c(0.9, 0.9),
#     axis.text = element_text(
#       face = "bold",
#       size = 12
#     ),
#     axis.title = element_text(
#       face = "bold",
#       size = 12
#     ),
#     # plot.title=element_text(face='bold', size=12,hjust = 0.5)
#   ) + 
#   scale_fill_manual(
#     values = colors_10[1:2]
#   ) + 
#   scale_color_manual(values = colors_10[1:2]) + 
#   labs(x = unname(TeX("$log_2 (\\sigma_2 / \\sigma_1)$")), 
#        y = 'density')
# 
# print(q1)
# 
# 
# ggarrange(
#   ggarrange(p1, p2, nrow = 2, labels = c("A", "B")), 
#   q1,
#   ncol = 2, 
#   labels = c(" ", "C")       # Label of the line plot
# ) 
# 
# ggsave(
#   filename = "CE_Convergence_plot.pdf", device = "pdf",
#   width = 12.40, height = 6.10,
#   path = "fig/"
# )


# --------------LZ entropy----

LZ_cut <- read.csv('Kelty/LZEntropyConvergence.csv')

names(LZ_cut) <- c('X', 'userid', 
                   'CutOff', 'entropy')

set.seed(2020)
userlist = sample(unique(LZ_cut$userid), 150)
LZ_cut <- LZ_cut%>% filter((entropy >= 0) & (userid %in% userlist) )


g1 <- ggplot(LZ_cut, aes(x=CutOff, y=entropy, color=userid)) + 
  geom_line() +   theme_bw() +
  theme(
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
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
  scale_x_continuous(labels = scales::percent) + 
  labs(x = "Cutoff percentage of ego's text", 
       y = "Entropy (bit) at cutoff"
  )
print(g1)

ggsave(
  filename = "LZ_Entropy_Convergence_plot.pdf", device = "pdf",
  width = 8, height = 4,
  path = "fig/"
)


## ---------------
TopJustification <- read.csv('Kelty/Top_Ten_Justification.csv')
TopJustification <- TopJustification %>% filter((Rank <= 30) )

ggplot(TopJustification, aes(x=Rank, y=Ratio)) + 
  geom_line(aes(color=NETWORK), size=1.5) + theme_bw() +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    legend.title = element_blank(),
    legend.position =  c(0.85, 0.8),
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
  labs(x = "Rank", 
       y = "Average Number of Co-locations"
  )

ggsave(
  filename = "average_meetups.pdf", device = "pdf",
  width = 5.74, height = 3.53,
  path = "fig/"
)