library("data.table")
library("dplyr")
library("ggplot2")
library("reshape2")
library("latex2exp")
library("magrittr")
library("ggpubr")
library("tidyr")
library("ggmap")
library("viridis")

colors_10 <- c(
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
)
catscale10 <- scale_colour_manual(values = colors_10)
catscale10_2 <- scale_fill_manual(values = colors_10)


location_Pi <- read.csv('Location/location_Pi.csv')
checkin <- read.csv('data/weeplace_checkins.csv')

clear_placeid <- checkin %>% distinct(placeid, lat, lon, .keep_all = TRUE) %>%
  select(placeid, lat, lon) %>%
  group_by(placeid) %>%
  filter(n() == 1) %>% pull(placeid)

# location_no_cat <- location_Pi %>% 
#   distinct(placeid, target, .keep_all = TRUE) %>% 
#   select(1:7) %>% melt(id.vars = c("placeid", 
#                                    "count",
#                                    "nunique",
#                                    "target"))  

# --------------count and filter -------------
# The number of distinct placeid (ambiguous placeid, same placeid multiple lat and lon)
N_placeid <- location_Pi %>% distinct(placeid) %>% count()


clear_LP <- location_Pi %>% filter(placeid %in% clear_placeid)
N_clear_placeid <- clear_LP %>% distinct(placeid) %>% count()


location_no_cat <- clear_LP %>% 
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

location_add_cat <- clear_LP %>% melt(id.vars = c("placeid", 
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


# --------------focus on top-10 value ------------------

# top 20% -------------------

top10 <- location_add_cat %>% group_by(target, variable) %>%
  arrange(target, variable, desc(value)) %>% 
  filter(value > quantile(value, .9)) %>% 
  group_by(target, variable, category) %>% count()

bottom10 <- location_add_cat %>% group_by(target, variable) %>%
  arrange(target, variable, desc(value)) %>% 
  filter(value < quantile(value, .1)) %>% 
  group_by(target, variable, category) %>% count()

ggplot(top10, aes(x = 1, weight = n, fill = category)) +
  geom_bar(width = 1) +
  coord_polar(theta = "y") +
  facet_wrap(~target + variable) + 
  catscale10 + catscale10_2



ggplot(bottom10, aes(x = 1, weight = n, fill = category)) +
  geom_bar(width = 1) +
  coord_polar(theta = "y") +
  facet_wrap(~target + variable) + 
  catscale10 + catscale10_2

#----------location stats full -------

location_full <- read.csv('Location/location_stats_full.csv')

location_full_new <- location_full %>% group_by(lat, lon) %>% 
  filter(row_number() == 1)

ggplot(location_full_new, aes(lon, lat)) +
  geom_point(aes(colour = mean_Pi)) + 
  scale_colour_gradient(low = "white", high = "black")

#-------------Seoul ---------------

Seoul <- location_full_new %>% filter(city=='Seoul', lat < 37.75, lon < 127.25) 

ggplot(Seoul , aes(lon, lat)) +
  geom_point(aes(colour = mean_Pi)) + 
  scale_colour_gradient(low = "white", high = "black")

KR <- c(left = 126.75, bottom = 37.35, right = 127.25, top = 37.70)
# us <- c(left = -74.2, bottom = 40.6, right = -73.7, top = 40.9)
get_stamenmap(KR, zoom = 5, maptype = "toner-lite") %>% ggmap()

qmplot(lon, lat, data = Seoul, maptype = "toner-lite", color = mean_Pi) + 
  scale_colour_gradient(low = "yellow", high = "red")


#-------New York --------------------------
newyork <- location_full_new %>% filter(city=='New York', lat < 41, 
                                        lat > 10, lon < 0)

unique(newyork[c("placeid")])

us <- c(left = -74.2, bottom = 40.5, right = -73.6, top = 40.9)
# us <- c(left = -74.2, bottom = 40.6, right = -73.7, top = 40.9)
get_stamenmap(us, zoom = 5, maptype = "toner-lite") %>% ggmap()


my_breaks <- c(0.37, 0.41, 0.45, 0.49, 0.53, 0.57, 0.61, 0.65, 0.69)

p_food <- qmplot(lon, lat, data = newyork %>% filter(category == 'Food'), 
       maptype = "toner-lite", color = mean_Pi) + 
  scale_colour_gradientn(colours = rev(magma(10)),
                         breaks = my_breaks, labels = my_breaks,
                         limits = c(0.37, 0.7)
                         )

p_night<- qmplot(lon, lat, data = newyork %>% filter(category == 'Nightlife Spots'), 
       maptype = "toner-lite", color = mean_Pi) + 
  scale_colour_gradientn(colours = rev(magma(10)),
                         breaks = my_breaks, labels = my_breaks,
                         limits = c(0.37, 0.67)
  )


p_travel <- qmplot(lon, lat, data = newyork %>% filter(category == 'Travel Spots'), 
       maptype = "toner-lite", color = mean_Pi) + 
  scale_colour_gradientn(colours = rev(magma(10)),
                         breaks = my_breaks, labels = my_breaks,
                         limits = c(0.37, 0.70)
  )


p_park <- qmplot(lon, lat, data = newyork %>% filter(category == 'Parks & Outdoors'), 
                   maptype = "toner-lite", color = mean_Pi) + 
  scale_colour_gradientn(colours = rev(magma(10)),
                         breaks = my_breaks, labels = my_breaks,
                         limits = c(0.37, 0.70)
  )

p_shop <- qmplot(lon, lat, data = newyork %>% filter(category == 'Shops'), 
                 maptype = "toner-lite", color = mean_Pi) + 
  scale_colour_gradientn(colours = rev(magma(10)),
                         breaks = my_breaks, labels = my_breaks,
                         limits = c(0.37, 0.70)
  )

p_art <- qmplot(lon, lat, data = newyork %>% filter(category == 'Arts & Entertainment'), 
                 maptype = "toner-lite", color = mean_Pi) + 
  scale_colour_gradientn(colours = rev(magma(10)),
                         breaks = my_breaks, labels = my_breaks,
                         limits = c(0.37, 0.70)
  )

p_edu <- qmplot(lon, lat, data = newyork %>% filter(category == 'College & Education'), 
                maptype = "toner-lite", color = mean_Pi) + 
  scale_colour_gradientn(colours = rev(magma(10)),
                         breaks = my_breaks, labels = my_breaks,
                         limits = c(0.37, 0.70)
  )

p_work<- qmplot(lon, lat, data = newyork %>% filter(category == 'Homes, Work, Others'), 
                maptype = "toner-lite", color = mean_Pi) + 
  scale_colour_gradientn(colours = rev(magma(10)),
                         breaks = my_breaks, labels = my_breaks,
                         limits = c(0.37, 0.70)
  )

print(p_food)

ggarrange(
  p_food, p_night, p_travel, p_park,
  p_art, p_edu, p_shop, p_work,
  labels = c("Food", "Nightlife Spots", "Travel Spots", "Parks & Outdoors", 
             "Arts & Entertainment", "College & Education", "Shops", "Homes, Work, Others"), 
  nrow = 2, ncol = 4,
  legend = FALSE
)


# virlin plot --------------------------

newyork_full <- location_full %>% filter(city=='New York', lat < 41, 
                                        lat > 10, lon < 0)

length(unique(newyork$placeid))
length(unique(newyork_full$placeid))

ggplot(newyork_full, aes(x=category, y = mean_Pi)) +
  geom_violin(aes(fill = category)
  ) +
  theme(axis.text.x = element_blank()) + 
  catscale10 + catscale10_2

write.csv(newyork, 'Location/newyork.csv')
#-------------London ---------------


London <- location_full_new %>% filter(city=='London',
                                       lon > -5, lat>51.3,
                                       lat <51.75, lon>-0.5) 

ggplot(London, aes(lon, lat)) +
  geom_point(aes(colour = mean_Pi)) + 
  scale_colour_gradient(low = "white", high = "black")

UK <- c(left = -0.5, bottom = 51.30, right = 0.1, top = 51.7)
# us <- c(left = -74.2, bottom = 40.6, right = -73.7, top = 40.9)
get_stamenmap(UK, zoom = 5, maptype = "toner-lite") %>% ggmap()

qmplot(lon, lat, data = London, maptype = "toner-lite", color = mean_Pi) + 
  scale_colour_gradient(low = "yellow", high = "red")
