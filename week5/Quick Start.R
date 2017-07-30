rm(list=ls())

#install.packages("prophet")
library(prophet)
library(dplyr)

df <- read.csv('/home/yeolpyeong/pragmaticML/week5/example_wp_peyton_manning.csv') %>% mutate(y = log(y))
m <- prophet(df)
future <- make_future_dataframe(m, period = 365)
tail(df)
tail(future)

forecast <- predict(m, future)
tail(forecast[c('ds', 'yhat', 'yhat_lower', 'yhat_upper')])

plot(m, forecast)
prophet_plot_components(m, forecast)
