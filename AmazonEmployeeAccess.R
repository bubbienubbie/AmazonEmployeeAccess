#----------------------Amazon Kaggle Project----------------------------#

#PACKAGES
install.packages("discrim")
install.packages("naivebayes")
install.packages("kknn")
install.packages("kernlab")
library(naivebayes)
library(discrim)
library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(kknn)
library(kernlab)


#------------------LOAD DATA IN----------------------------------#
Atrain <- vroom("C:/Users/isaac/Documents/Fall 2023 Real/AmazonEmployeeAccess/train.csv")
Atest  <- vroom("C:/Users/isaac/Documents/Fall 2023 Real/AmazonEmployeeAccess/test.csv")
#Atrain <- vroom("train.csv")
#Atest  <- vroom("test.csv")

#RECIPE
my_recipe <- recipe(ACTION ~., data=Atrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%    # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>%     # combines categorical values that occur <% into an "other" value
  step_dummy(all_nominal_predictors())  %>%                       # dummy variable encoding 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_predictors(), threshold=.8)
  #target encoding (must be 2-factor
  #also step_lencode_glm() and step_lencode_bayes()

prep <- prep(my_recipe)
baked <- bake(prep, new_data = Atrain)

Atrain$ACTION <- as.factor(Atrain$ACTION)






#--------------------Logistic Regression--------------------------------#
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
save(file="./LogsiAmazon.csv")





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
save(file="./PenlogAmazon.csv")






#----------------------RandomForests--------------#
#MyMod
for_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")


#WORKFLOW
for_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(for_mod)

#TUNING
tuning_grid <- grid_regular(mtry(range= c(1,10)),
                            min_n(),
                            levels = 5)


## Split data for CV
folds <- vfold_cv(Atrain, v = 5, repeats=2)

## Run the CV
CV_results <- for_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
finalfor_wf <-
  for_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=Atrain)

## Predict
for_pred <- finalfor_wf %>%
  predict(new_data = Atest, type="prob")


#FORMAT
for_pred <- for_pred %>%
  select(ACTION = .pred_1)
Aforpred <- data.frame(Id = Atest$id, for_pred)

#MAKE FILE
vroom_write(Aforpred, file="ForFinalAmazon.csv", delim=",")




#---------------Commands-------------
# NEED BYU VPN TO RUN THIS OUTSIDE OF EDUROAM
# ls
# $ cd shared/Kaggle/AmazonEmployeeAccess/
# git pull
# R CMD BATCH --no-save --no-restore Amazon_Heaton.R (file name) & (& - run in back)
# top (tells you what is running)
# less XXXX R.out will show you when error was hit

# R code
# save(file="filename.RData" , list=c(logReg_wf"))
# load("filename.RData")
# Do this so the cloud knows to save your results and not just run and delete




#--------------------------Naive Bayes----------------#
#MyMod
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes eng


#WORKFLOW
nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

#TUNING
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 3)


## Split data for CV
folds <- vfold_cv(Atrain, v = 5, repeats=1)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
finalnb_wf <-
  nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=Atrain)

## Predict
nb_pred <- finalnb_wf %>%
  predict(new_data = Atest, type="prob") %>%
  bind_cols(., Atest)


#FORMAT
nb_pred <- nb_pred %>%
  select(ACTION = .pred_1)
Anbpred <- data.frame(Id = Atest$id, nb_pred)

#MAKE FILE
vroom_write(Anbpred, file="NBAmazon.csv", delim=",")





#----------------------K Nearest-----------#
## knn model
knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)

#TUNING
tuning_grid <- grid_regular(neighbors(range = c(1,25)),
                            levels = 3)


## Split data for CV
folds <- vfold_cv(Atrain, v = 3, repeats=1)

## Run the CV
CV_results <- knn_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
finalknn_wf <-
  knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=Atrain)

## Predict
knn_pred <- finalknn_wf %>%
  predict(new_data = Atest, type="prob")


#FORMAT
knn_pred <- knn_pred %>%
  select(ACTION = .pred_1)
Aknnpred <- data.frame(Id = Atest$id, knn_pred)

#MAKE FILE
vroom_write(Aknnpred, file="KNNAmazon.csv", delim=",")




#-----------------------SVM--------------#
#MyMod
svmPoly <- svm_poly(degree=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

#WORKFLOW
svm_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmPoly)

#TUNING
tuning_grid <- grid_regular(cost(),
                            levels = 4)


## Split data for CV
folds <- vfold_cv(Atrain, v = 3, repeats=1)

## Run the CV
CV_results <- svm_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
finalsvm_wf <-
  svm_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=Atrain)

## Predict
svm_pred <- finalsvm_wf %>%
  predict(new_data = Atest, type="prob")


#FORMAT
svm_pred <- svm_pred %>%
  select(ACTION = .pred_1)
Asvmpred <- data.frame(Id = Atest$id, svm_pred)

#MAKE FILE
vroom_write(Asvmpred, file="SVMAmazon.csv", delim=",")
