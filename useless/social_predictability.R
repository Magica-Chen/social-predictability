library(data.table)
library(readr)
library(tidyr)


df_friend <- read_csv("data/weeplace_friends.csv")
df_full <- read_csv("data/weeplace_checkins.csv")
# df_full <- df_full %>% drop_na(placeid)

df_hour <- df_full
df_hour$datetime <- trunc(df_full$datetime, 'hour')

b <- subset(df_hour, userid %in% c('00-a'), select = 'datetime')
e <- b[order(b$datetime, decreasing = TRUE),]

