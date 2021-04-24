library("data.table")
library("dplyr")
library("ggplot2")
library("reshape2")
library("latex2exp")
library("magrittr")
library("ggpubr")
library("tidyr")

colors_10 <- c(
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
)
catscale10 <- scale_colour_manual(values = colors_10)
catscale10_2 <- scale_fill_manual(values = colors_10)


location_Pi <- read.csv('location_Pi.csv')

location_no_cat <- location_Pi %>% 
  distinct(placeid, target, .keep_all = TRUE) %>% 
  select(1:7) %>% melt(id.vars = c("placeid", 
                                   "count",
                                   "nunique",
                                   "target"))  

#-------No category-----------------------------



ggplot(location_no_cat, aes(x = value)) +
  geom_histogram( aes(fill=variable),
    alpha = 0.7, position = "identity", bins = 100
  ) +
  facet_wrap(~target + variable, scales = 'free_x') + 
  scale_fill_manual(
    values = colors_10[1:3]
  ) 

#--------Add category- (hist view)--------------------------

location_add_cat <- location_Pi %>% melt(id.vars = c("placeid", 
                                   "count",
                                   "nunique",
                                   "target",
                                   "category")) %>% 
  na_if("") %>%
  na.omit

unique(location_add_cat[c("category")])

ggplot(location_add_cat, aes(x = value)) +
  geom_density(size=1, aes(color=category)
  ) +
  facet_wrap(~target + variable, scales = 'free') + 
  catscale10 + catscale10_2


#--------Add category (violin view) --------------------

ggplot(location_add_cat, aes(x=category, y = value)) +
  geom_violin(aes(fill = category)
  ) +
  theme(axis.text.x = element_blank()) + 
  facet_wrap(~target + variable, scales = 'free') + 
  catscale10 + catscale10_2




