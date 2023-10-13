#----------------------Amazon Kaggle Project----------------------------#

#PACKAGES
library(tidyverse)
library(tidymodels)
library(vroom)
library(DataExplorer)
library(GGally)
library(embed)
library(ggmosaic)


#LOAD DATA IN
Atrain <- vroom("C:/Users/isaac/Documents/Fall 2023 Real/New folder/archive/train.csv")
Atest  <- vroom("C:/Users/isaac/Documents/Fall 2023 Real/New folder/archive/test.csv")



#--------------------Logistic Regression--------------------------------#


#RECIPE
my_recipe <- recipe(ACTION ~., data=Atrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%    # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .01) %>%     # combines categorical values that occur <% into an "other" value
  step_dummy(all_nominal_predictors())                         # dummy variable encoding 
  
  #step_lencode_mixed(all_nominal_predictors(), outcome = vars(target_var)) #target encoding (must be 2-factor
  #also step_lencode_glm() and step_lencode_bayes()

prep <- prep(my_recipe)
baked <- bake(prep, new_data = Atrain)

Atrain$ACTION <- as.factor(Atrain$ACTION)

#MY MOD
my_mod <- logistic_reg() %>% #Type of model
  set_engine("glm")

#WORKFLOW
amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data = Atrain) # Fit the workflow

#PREDICTIONS
amazon_predictions <- predict(amazon_workflow,
                              new_data=Atest,
                              type="prob") # "class" or "prob" (see doc)

#FORMAT
amazon_predictions <- amazon_predictions %>%
  select(ACTION = .pred_1)
Apred <- data.frame(Id = Atest$id, amazon_predictions)

#MAKE FILE
vroom_write(Apred, file="LogsiAmazon.csv", delim=",")





#------------------------Penalized Logistic Regression--------------------#

#RECIPE
my_recipe <- recipe(ACTION ~., data=Atrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%    # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>%     # combines categorical values that occur <% into an "other" value
  #step_dummy(all_nominal_predictors())                         # dummy variable encoding 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) #target encoding (must be 2-factor
  #also step_lencode_glm() and step_lencode_bayes()

prep <- prep(my_recipe)
baked <- bake(prep, new_data = Atrain)
baked_test <- bake(prep, new_data = Atest)

Atrain$ACTION <- as.factor(Atrain$ACTION)

#MY MOD
my_mod <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model
  set_engine("glmnet")

#WORKFLOW
penlog_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)

#TUNING
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 3)

## Split data for CV
folds <- vfold_cv(Atrain, v = 3, repeats=2)

## Run the CV
CV_results <- penlog_workflow %>%
tune_grid(resamples=folds,
          grid=tuning_grid,
          metrics=metric_set(roc_auc)) #f_meas, sens, recall, spec,
                             #precision, accuracy)) #Or leave metrics NULL

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_wf <-
  penlog_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=Atrain)

## Predict
penlog_pred <- final_wf %>%
  predict(new_data = Atest, type="prob")


#FORMAT
penlog_pred <- penlog_pred %>%
  select(ACTION = .pred_1)
Apenlogpred <- data.frame(Id = Atest$id, penlog_pred)

#MAKE FILE
vroom_write(Apenlogpred, file="PenlogAmazon.csv", delim=",")


