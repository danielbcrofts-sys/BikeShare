library(tidyverse)
library(tidymodels)
library(vroom)
library(dplyr)

dat_test <- vroom("test.csv")
dat_train <- vroom("train.csv")
glimpse(dat_test)
glimpse(dat_train)

dat_train <- dat_train %>%
  select(-casual, -registered) %>%
  mutate(count = log(count))

bike_recipe <- recipe(count ~ ., data = dat_train) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(weather = factor(weather)) %>%
  step_time(datetime, features = "hour") %>%
  step_mutate(season = factor(season)) %>%
  step_date(datetime, features = "dow")


lin_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

bike_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(lin_model) %>%
  fit(data = dat_train)

lin_preds <- predict(bike_workflow, new_data = dat_train)
lin_preds <- exp(lin_preds$.pred)

prepped_recipe <- prep(bike_recipe)
baked_train <- bake(prepped_recipe, new_data = dat_train)
head(baked_train, 5)

kaggle_submission <- tibble(datetime = dat_test$datetime, count = lin_preds)
write.csv(kaggle_submission, "submission2.csv", row.names = FALSE)
