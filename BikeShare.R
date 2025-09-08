library(tidyverse)
library(tidymodels)
library(vroom)
library(ggplot2)
library(patchwork)

dat_test <- vroom("test.csv")
dat_train <- vroom("train.csv")
glimpse(dat_test)
glimpse(dat_train)


weather <- ggplot(dat_train, aes(x = weather)) +
  geom_bar(fill= "steelblue") +
  labs(title = "Weather")


dat_train$season <- factor(as.numeric(dat_train$season),
                           levels = 1:4,
                           labels = c("Winter", "Spring", "Summer", "Fall"))

# Plot
seasonal <- ggplot(dat_train, aes(x = season, y = count, fill = season)) +
  geom_boxplot() +
  labs(title = "Bike Rentals by Season")


temperature <- ggplot(dat_train, aes(x = temp, y = count)) +
  geom_smooth(method = "loess", color = "steelblue") +
  labs(title = "Bike Rentals vs Temperature")


dat_train$holiday <- factor(as.numeric(dat_train$holiday),
                            levels = c(0, 1),
                            labels = c("No Holiday", "Holiday"))

holiday <- ggplot(dat_train, aes(x = holiday)) +
  geom_bar(fill = "steelblue") + 
  labs(title = "Bike Rentals on Holidays")

combined <- (weather + seasonal)   / (temperature + holiday)
combined  

ggsave(filename = "four_panel_plot.png",
       plot = combined,
       width = 10, height = 8,
       dpi = 300)                            

  