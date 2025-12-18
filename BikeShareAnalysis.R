library(tidyverse)
library(tidymodels)
library(vroom)
library(dplyr)

dat_test <- vroom("test.csv")
dat_train <- vroom("train.csv")
glimpse(dat_test)
glimpse(dat_train)



my_linear_model <- linear_reg() %>% 
  set_engine("lm") %>% 
  set_mode("regression") %>% 
  fit(
    formula = log(count) ~ . - datetime,
    data = dat_train %>% select(-casual, -registered)  # drop so train/test align
  )

## Generate Predictions Using Linear Model
bike_predictions <- predict(
  my_linear_model,
  new_data = dat_test
)
bike_predictions


kaggle_submission <- bike_predictions %>%
  bind_cols(., dat_test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file9
vroom_write(x=kaggle_submission, file="./LinearPreds.csv", delim=",")
