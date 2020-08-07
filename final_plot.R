library(data.table)
library("dplyr")
library(ggplot2)
library(reshape2)
library(latex2exp)

theme_set(
  theme_bw() +
    theme(legend.position = "right")
)

colors_10 = c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")
catscale10  <- scale_colour_manual(values=colors_10)
catscale10_2 <- scale_fill_manual(values=colors_10)


wp = read.csv('final/wp-150/wp-dataset-basic.csv')
bk = read.csv('final/bk-150/bk-dataset-basic.csv')
gw = read.csv('final/gws-150/gws-dataset-basic.csv')

wp = na.omit(wp)
bk = na.omit(bk)
gw = na.omit(gw)

wp$dataset = 'Weeplace'
bk$dataset = 'BrightKite'
gw$dataset = 'Gowalla'
bk[, 1] <- sapply(bk[, 1], as.factor)
gw[, 1] <- sapply(gw[, 1], as.factor)

df <- do.call("rbind", list(wp, bk, gw))


df_entropy <- df %>% select(userid, Shannon.Entropy, LZ.Entropy, dataset)
df_entropy_melt <- melt(df_entropy,
                        id.vars=c('userid', 'dataset'))

df_predictability <- df %>% select(userid, Shannon.Pi, LZ.Pi, dataset)
df_predictability_melt <- melt(df_predictability,
                               id.vars=c('userid', 'dataset'))

# # For entropy
# ggplot(df_entropy_melt, aes(x = value)) + 
#   geom_histogram(aes(color = variable, y = stat(density), fill=variable),
#                  alpha = 0.4, position = "identity", bins = 100) +
#   theme(legend.title=element_blank()) + 
#   geom_density(aes(color = variable), size = 1) + 
#   scale_fill_manual(values = colors_10[1:2]) +
#   scale_color_manual(values = colors_10[1:2]) + 
#   # catscale10 + catscale10_2 +
#   facet_wrap(~ dataset) + 
#   labs(x='Entropy (bit)') 
# 
#   # scale_fill_manual(values = c("#00AFBB", "#E7B800")) +
#   # scale_color_manual(values = c("#00AFBB", "#E7B800"))
# 
# # for predictability
# ggplot(df_predictability_melt, aes(x = value)) + 
#   geom_histogram(aes(color = variable, y = stat(density), fill=variable),
#                  alpha = 0.4, position = "identity", bins = 100) +
#   theme(legend.title=element_blank()) + 
#   geom_density(aes(color = variable), size = 1) + 
#   scale_fill_manual(values = colors_10[3:4]) +
#   scale_color_manual(values = colors_10[3:4]) + 
#   # catscale10 + catscale10_2 +
#   facet_wrap(~ dataset) + 
#   labs(x='Predictability') 


ggplot(df_entropy_melt, aes(x = value)) +
  geom_histogram(aes(fill=variable),
                 alpha = 0.8, position = "identity", bins = 100) +
  theme(legend.title=element_blank(), 
        legend.spacing.x = unit(0.5, "char"),
        strip.text = element_text(size=15),
        legend.text = element_text(face='bold',
                                   size=12),
        axis.text=element_text(face='bold',
                               size=12),
        axis.title=element_text(face='bold',
                                size=12),
        # plot.title=element_text(face='bold', size=12,hjust = 0.5)
        ) + 
  scale_fill_manual(labels=unname(TeX(c("$S_{SN}", "$S_{LZ}"))), 
                    values = colors_10[1:2]) +
  # scale_color_manual(labels=c('A', 'B'), values = colors_10[1:2]) + 
  facet_wrap(~ dataset) + 
  labs(x='Entropy (bit)') 

ggsave(filename = 'hist_entropy.pdf', device = 'pdf',
       width = 9.90, height = 2.66,
       path='fig/')



ggplot(df_predictability_melt, aes(x = value)) +
  geom_histogram(aes(fill=variable),
                 alpha = 0.8, position = "identity", bins = 100) +
  theme(legend.title=element_blank(),
        legend.spacing.x = unit(0.5, "char"),
        strip.text = element_text(size=15),
        legend.text = element_text(face='bold',
                                   size=12),
        axis.text=element_text(face='bold',
                               size=12),
        axis.title=element_text(face='bold',
                                size=12),
        # plot.title=element_text(face='bold', size=12,hjust = 0.5)
  )  + 
  scale_fill_manual(labels=unname(TeX(c("$\\Pi_{SN}", "$\\Pi_{LZ}"))), 
                    values = colors_10[3:4]) +
  facet_wrap(~ dataset) + 
  labs(x='Predictability')

ggsave(filename = 'hist_predictability.pdf', device = 'pdf',
       width = 9.90, height = 2.66,
       path='fig/')

