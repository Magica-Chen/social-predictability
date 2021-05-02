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



location_NYC <- fread('Location/dataset_TSMC2014_NYC.txt')
names(location_NYC) <- c('userid', 'placeid', 
                         'category_id', 'category_name',
                         'lat', 'lon',
                         'timezone', 'datetime')

# strptime(location_NYC[['datetime']], format='%Y.%m.%d  %H:%M:%S')


location_NYC$datetime <- format(as.POSIXct(location_NYC$datetime, 
                                           format = "%a %b %d %H:%M:%S %z %Y",
                                          tz = "UTC"),
                         "%Y-%m-%d %H:%M:%S"
)
  



write.csv(x = location_NYC, row.names = FALSE,
          file='Location//NYC_checkins.csv')
