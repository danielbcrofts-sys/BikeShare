library(tidyverse)
library(tidymodels)
library(vroom)
library(ggplot2)
library(patchwork)

dat_test <- vroom("test.csv")
dat_train <- vroom("train.csv")
glimpse(dat_test)
glimpse(dat_train)


# weather <- ggplot(dat_train, aes(x = weather)) +
#   geom_bar(fill= "steelblue") +
#   labs(title = "Weather")
# 
# 
# dat_train$season <- factor(as.numeric(dat_train$season),
#                            levels = 1:4,
#                            labels = c("Winter", "Spring", "Summer", "Fall"))
# 
# # Plot
# seasonal <- ggplot(dat_train, aes(x = season, y = count, fill = season)) +
#   geom_boxplot() +
#   labs(title = "Bike Rentals by Season")
# 
# 
# temperature <- ggplot(dat_train, aes(x = temp, y = count)) +
#   geom_smooth(method = "loess", color = "steelblue") +
#   labs(title = "Bike Rentals vs Temperature")
# 
# 
# dat_train$holiday <- factor(as.numeric(dat_train$holiday),
#                             levels = c(0, 1),
#                             labels = c("No Holiday", "Holiday"))
# 
# holiday <- ggplot(dat_train, aes(x = holiday)) +
#   geom_bar(fill = "steelblue") + 
#   labs(title = "Bike Rentals on Holidays")
# 
# combined <- (weather + seasonal)   / (temperature + holiday)
# combined  
# 
# ggsave(filename = "four_panel_plot.png",
#        plot = combined,
#        width = 10, height = 8,
#        dpi = 300)                            


###WORKFLOW

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

lin_preds <- predict(bike_workflow, new_data = dat_test)
lin_preds <- exp(lin_preds$.pred)

prepped_recipe <- prep(bike_recipe)
baked_train <- bake(prepped_recipe, new_data = dat_train)
head(baked_train, 5)

kaggle_submission <- tibble(
  datetime = as.character(format(dat_test$datetime, "%Y-%m-%d %H:%M:%S")),
  count = lin_preds
)
write.csv(kaggle_submission, "submission2.csv", row.names = FALSE)


###Penalized Regression


bike_recipe <- recipe(count ~ ., data = dat_train) %>%
  step_mutate(season = factor(season),
              holiday = factor(holiday),
              workingday = factor(workingday),
              weather = factor(weather)) %>%  
  step_time(datetime, features = "hour") %>%
  step_date(datetime, features = "dow") %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())


params <- tibble(
  penalty  = c(0.01, 5, 0.5, 3, 3),
  mixture  = c(0.1, 0.1, 0.9, 0.9, 0.5)
)

results <- list()

for(i in 1:nrow(params)) {
  enet_model <- linear_reg(
    penalty = params$penalty[i],
    mixture = params$mixture[i]
  ) %>%
    set_engine("glmnet") %>%
    set_mode("regression")
  
  bike_workflow <- workflow() %>%
    add_recipe(bike_recipe) %>%
    add_model(enet_model) %>%
    fit(data = dat_train)
  
  # predict on test set
  lin_preds <- predict(bike_workflow, new_data = dat_test)
  lin_preds <- exp(lin_preds$.pred)
  lin_preds <- pmax(lin_preds, 0)   # no negative counts
  
  results[[i]] <- list(
    penalty = params$penalty[i],
    mixture = params$mixture[i],
    preds = lin_preds
  )
}

for(i in 1:length(results)) {
  kaggle_submission <- tibble(
    datetime = as.character(format(dat_test$datetime, "%Y-%m-%d %H:%M:%S")),
    count = results[[i]]$preds
  )
  
  filename <- paste0("submission_penalty_", results[[i]]$penalty,
                     "_mixture_", results[[i]]$mixture, ".csv")
  
  write.csv(kaggle_submission, filename, row.names = FALSE)
}


###TUNING MODELS


## Penalized regression model
preg_model <- linear_reg(
  penalty = tune(),
  mixture = tune()
) %>%
  set_engine("glmnet") # Function to fit in R

## Set Workflow
preg_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(preg_model)

## Grid of values to tune over
grid_of_tuning_params <- grid_regular(
  penalty(),
  mixture(),
  levels = 10
) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(dat_train, v = 5, repeats = 1)

## Run CV
CV_results <- preg_wf %>%
  tune_grid(
    resamples = folds,
    grid = grid_of_tuning_params,
    metrics = metric_set(rmse, mae) # Or leave metrics NULL
  )

## Plot Results (example)
collect_metrics(CV_results) %>%
  filter(.metric == "rmse") %>%
  ggplot(aes(x = penalty, y = mean, color = factor(mixture))) +
  geom_line()

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric = "rmse")

final_wf <- preg_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = dat_train)

## Predict
final_preds <- final_wf %>%
  predict(new_data = dat_test)

final_preds <- exp(final_preds$.pred)   # undo log
final_preds <- pmax(final_preds, 0)     # no negatives

kaggle_submission <- tibble(
  datetime = as.character(format(dat_test$datetime, "%Y-%m-%d %H:%M:%S")),
  count = final_preds
)

write.csv(kaggle_submission, "submission_tuned.csv", row.names = FALSE)

