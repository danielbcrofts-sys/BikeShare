library(tidyverse)
library(tidymodels)
library(vroom)
library(ggplot2)
library(patchwork)
library(rpart)
library(ranger)
library(bonsai)
library(lightgbm)


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


### REGRESSION TREES
tree_model <- decision_tree(
  tree_depth = tune(),          # maximum depth of tree
  cost_complexity = tune(),     # pruning parameter
  min_n = tune()                # minimum samples per node
) %>%
  set_engine("rpart") %>%       # What R function to use
  set_mode("regression")        # Numeric prediction

## Set Workflow
tree_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%   # use your recipe from earlier
  add_model(tree_model)

## Grid of values to tune over
grid_tree <- grid_regular(
  tree_depth(),
  cost_complexity(),
  min_n(),
  levels = 3        # adjust if you want finer search (3^3 = 27 combos)
)

## Split data for CV
folds <- vfold_cv(dat_train, v = 5)

## Tune  model
tree_results <- tree_wf %>%
  tune_grid(
    resamples = folds,
    grid = grid_tree,
    metrics = metric_set(rmse, mae)
  )

## Find best tuning parameters
best_tree <- tree_results %>%
  select_best(metric = "rmse")

## Final workflow
final_tree_wf <- tree_wf %>%
  finalize_workflow(best_tree) %>%
  fit(data = dat_train)

## Predict on test data
tree_preds <- predict(final_tree_wf, new_data = dat_test)
tree_preds <- exp(tree_preds$.pred)   # backtransform log(count) → count
tree_preds <- pmax(tree_preds, 0)     # no negative counts

## Kaggle submission
kaggle_tree <- tibble(
  datetime = as.character(format(dat_test$datetime, "%Y-%m-%d %H:%M:%S")),
  count = tree_preds
)

write.csv(kaggle_tree, "tree_submission.csv", row.names = FALSE)



### RANDOM FOREST


## Random forest model
my_mod <- rand_forest(
  mtry = tune(),       # number of predictors considered at each split
  min_n = tune(),      # minimum samples per terminal node
  trees = 500          # number of trees in the forest
) %>%
  set_engine("ranger") %>%   # R package used for fitting
  set_mode("regression")     # numeric outcome

## Create workflow
my_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%  # use the recipe defined earlier
  add_model(my_mod)

## Grid of values to tune over
mygrid <- grid_regular(
  mtry(range = c(2, 6)),      # adjust based on number of predictors
  min_n(range = c(5, 20)),
  levels = 3                  # 3^2 = 9 total tuning combinations
)

## Split data for cross-validation
folds <- vfold_cv(dat_train, v = 5)

## Tune the model
CV_results <- my_wf %>%
  tune_grid(
    resamples = folds,
    grid = mygrid,
    metrics = metric_set(rmse, mae)
  )

## Find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric = "rmse")

## Final workflow with best parameters
final_wf <- my_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = dat_train)

## Predict on test data
rf_preds <- predict(final_wf, new_data = dat_test)
rf_preds <- exp(rf_preds$.pred)       # back-transform log(count)
rf_preds <- pmax(rf_preds, 0)         # no negative counts

## Kaggle submission
kaggle_submission <- tibble(
  datetime = as.character(format(dat_test$datetime, "%Y-%m-%d %H:%M:%S")),
  count = rf_preds
)

write.csv(kaggle_submission, "rf_submission.csv", row.names = FALSE)


###Boosted Trees

## Boosted Tree Model
my_mod <- boost_tree(
  tree_depth = tune(),
  trees = tune(),
  learn_rate = tune()
) %>%
  set_engine("lightgbm") %>%  # or "xgboost"
  set_mode("regression")

## Workflow with recipe
my_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(my_mod)

## Grid of values to tune over
mygrid <- grid_regular(
  tree_depth(),
  trees(),
  learn_rate(),
  levels = 3   # 3^3 = 27 combos
)

## Set up CV
folds <- vfold_cv(dat_train, v = 5)

## Tune the model
CV_results <- my_wf %>%
  tune_grid(
    resamples = folds,
    grid = mygrid,
    metrics = metric_set(rmse)
  )

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric = "rmse")

## Finalize Workflow and Fit
final_wf <- my_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = dat_train)

## Predict on Test Data
boost_preds <- final_wf %>%
  predict(new_data = dat_test)

boost_preds <- exp(boost_preds$.pred)      # back-transform
boost_preds <- pmax(boost_preds, 0)        # no negative counts

## Save Submission File
kaggle_submission <- tibble(
  datetime = as.character(format(dat_test$datetime, "%Y-%m-%d %H:%M:%S")),
  count = boost_preds
)

write.csv(kaggle_submission, "submission_boosted.csv", row.names = FALSE)


