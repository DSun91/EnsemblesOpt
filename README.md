# EnsemblesOpt 
<br/>



<p align="middle">
  
  <img src="https://user-images.githubusercontent.com/62545181/193424526-7e097194-c097-4cc5-9231-97cdf46d6147.gif" width="400" />
  <img src="https://user-images.githubusercontent.com/62545181/193418034-4cdb4aab-0d6d-410b-a648-841999caf560.gif" width="400" /> 
</p>

This repository contains the project for a package for speeding up the process of finding best base learners for building ensemble models trough Bayesian Optimization using Gaussian Processes as surrogate function and Expected Improvement (EI), Probability of Improvement (PI) or Upper Confidence Bound (UCB) as acquisition functions, along optimization routines developed using Optuna library.<br/>
The black-box function is defined as the n cross-validation score of the chosen evaluation metric for the ensemble considered during the iteration. Each base model is mapped to an integer value and their combination is passed to the objective function for evaluation.

Install by running:

```
!pip install EnsemblesOpt==0.0.8
```
# Code Snippets
First import the base models from where to search for the best ensemble of a given size
```

from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
```

## Bayesian optimization search:
```
#initialize the Bayesian_Voting_Ensemble
from EnsemblesOpt import Bayesian_Voting_Ensemble


BS=Bayesian_Voting_Ensemble(ensemble_size=2,
                            models_list=[ExtraTreeClassifier(),
                                             DecisionTreeClassifier(),
                                             MLPClassifier(),
                                             SGDClassifier(),
                                             KNeighborsClassifier()],
                           xi=0.05,
                           random_init_points=7,
                           maximize_obj=True,
                           scoring='roc_auc',
                           task='classification',
                           acquisition_func='EI')
                           
#fit the Bayesian_Voting_Ensemble                         
Best_model = BS.fit(X,y,
       Nfold=5,
       n_iters=9,
       stratify=True)
``` 
```
Output:
Collecting initial random points...
Searching best ensemble...
-trial  0 |Score value: 0.8626962395405989
-trial  1 |Score value: 0.8755565498352099
-trial  2 |Score value: 0.8742921444887171
-trial  3 |Score value: 0.8868338004352088
-trial  4 |Score value: 0.8562297244914867
-trial  5 |Score value: 0.8629782101656331
-trial  6 |Score value: 0.865559835850203
-trial  7 |Score value: 0.887221833391049
-trial  8 |Score value: 0.8534670721947504
-trial  9 |Score value: 0.8283346726135243
Best Ensemble:
 [LGBMClassifier(bagging_fraction=0.9861531786655775, bagging_freq=3,
               feature_fraction=0.14219334035549125,
               lambda_l1=7.009080384469092e-07, lambda_l2=5.029465681170278e-06,
               learning_rate=0.08695762873585877, max_bin=1255,
               min_child_samples=93, n_estimators =316, num_leaves=38,
               silent='warn'), GradientBoostingClassifier()] 
best score 0.887221833391049
```

Common parameters for the Bayesian_Voting_Ensemble class:<br/>

| Parameter  | Usage|
| ------------- | ------------- |
| **"ensemble_size"**  | Number of base estimators to build the ensemble, the bigger the ensemble the more time consuming and complex the final model will be.<br/>  |
| **"ensemble_size"**  | Number of base estimators to build the ensemble, the bigger the ensemble the more time consuming and complex the final model will be.<br/>  |
| **"models_list"**  | List of base models. If value provided is "None" preloaded list of models will be used.<br/> |
| **"xi"**  | Exploration parameter, higher values lead to more explorative behaviour and viceversa for lower value (default xi=0.01) .<br/>  |
| **"random_init_points"**  | Number of initial points to take from the objective function.<br/>  |
| **"maximize_obj"**  | Whether to maximize or minimize the objective function [True or False].<br/>  |
| **"scoring"**  | Metric to optimize.<br/>  |
| **"task"**  | Equals "classification" or "regression".<br/>  |
| **"type_p"**  | Only in case of classification problem select 'soft' or 'hard'.<br/>  |
| **"acquisition_func"**  | Acquisition function choose between "PI" (probability of improvement), "EI" (expected improvement) or "UCB" (upper confidence bound)<br/>  


Common parameters for the fit method:<br/>
| Parameter  | Usage|
| ------------- | ------------- |
| **"X"** | Training dataset without target variable.<br/>  |
| **"y"** | Target variable.<br/>  |
| **"n_iters"** | Number of trials to execute optimization.<br/>  |
| **"N_folds"** | Number of folds for cross validation.<br/> |
| **"stratify"** | Stratify cv splits based on target distribuition [True or False]<br/>  |

The **'scoring'** parameter takes the same values from sklearn API (link of available list: https://scikit-learn.org/stable/modules/model_evaluation.html)

## Optuna best stacking ensemble search:
``` 
from EnsemblesOpt import Optuna_StackEnsemble_Search

Opt=Optuna_StackEnsemble_Search(scoring_metric="roc_auc",
                                direction="maximize",
                                problem_type='classification',
                                size_stack=2,
                                models_list=[ExtraTreeClassifier(),
                                             DecisionTreeClassifier(),
                                             MLPClassifier(),
                                             SGDClassifier(),
                                             KNeighborsClassifier()],
                                             meta_learner=LogisticRegression())

Best_model, study = Opt.fit(X,y,n_trials=50,N_folds=3,stratify=True)
```
Common parameters for the Optuna_StackEnsemble_Search class:<br/>

| Parameter  | Usage|
| ------------- | ------------- |
| **"size_stack"**  | Number of base estimators to build the ensemble, the bigger the ensemble the more time consuming and complex the final model will be.<br/>  |
| **"models_list"**  | List of base models.<br/> |
| **"scoring_metric"**  | Metric to optimize.<br/>  |
| **"problem_type"**  | Equals "classification" or "regression".<br/>  |
| **"direction"**  | Equals "maximize" or "minimize".<br/>  |
| **"meta_learner"**  | Meta learner for the stack ensemble, if not provided Optuna will search for one from the base models.<br/>  |


Common parameters for the fit method:<br/>
| Parameter  | Usage|
| ------------- | ------------- |
| **"X"** | Training dataset without target variable.<br/>  |
| **"y"** | target variable.<br/>  |
| **"n_iters"** | Number of trials to execute optimization.<br/>  |
| **"N_folds"** | Number of folds for cross validation.<br/> |
| **"stratify"** | Stratify cv splits based on target distribuition [True or False]<br/>  |

## Optuna best voting ensemble search:

``` 
from EnsemblesOpt import Optuna_VotingEnsemble_Search

Opt=Optuna_VotingEnsemble_Search(scoring_metric="roc_auc",direction="maximize",
                                problem_type='classification',
                                ensemble_size=2,
                                models_list=[ExtraTreeClassifier(),
                                             DecisionTreeClassifier(),
                                             MLPClassifier(),
                                             SGDClassifier(),
                                             KNeighborsClassifier()],
                                voting_type='soft'
                               )
Best_model, study = Opt.fit(X,y,n_trials=10,
            N_folds=3,
            stratify=True)
```
Common parameters for the Optuna_StackEnsemble_Search class:<br/>
scoring_metric,direction,problem_type,size_stack=3,models_list=[]
| Parameter  | Usage|
| ------------- | ------------- |
| **"ensemble_size"**  | Number of base estimators to build the ensemble, the bigger the ensemble the more time consuming and complex the final model will be.<br/>  |
| **"models_list"**  | List of base models.<br/> |
| **"scoring_metric"**  | Metric to optimize.<br/>  |
| **"problem_type"**  | Equals "classification" or "regression".<br/>  |
| **"direction"**  | Equals "maximize" or "minimize".<br/>  |
| **"voting_type"**  | Voting type 'soft' or 'hard'.<br/>  |


Common parameters for the fit method:<br/>
| Parameter  | Usage|
| ------------- | ------------- |
| **"X"** | Training dataset without target variable.<br/>  |
| **"y"** | target variable.<br/>  |
| **"n_iters"** | Number of trials to execute optimization.<br/>  |
| **"N_folds"** | Number of folds for cross validation.<br/> |
| **"stratify"** | Stratify cv splits based on target distribuition [True or False]<br/>  |

## Optuna search best weights for voting ensemble:

``` 
from EnsemblesOpt import Optuna_Voting_weights_tuner

Opt=Optuna_Voting_weights_tuner(scoring_metric="roc_auc",
                                direction="maximize",
                                problem_type='classification',
                                models_list=[ExtraTreeClassifier(),
                                             DecisionTreeClassifier(),
                                             MLPClassifier(),
                                             SGDClassifier(),
                                             KNeighborsClassifier()],
                                voting_type='soft',
                                weights_list=[1,2,3]
                               )
Best_model, study = Opt.fit(X,y,n_trials=10,
        N_folds=3,
        stratify=True)
```
Common parameters for the Optuna_StackEnsemble_Search class:<br/>

| Parameter  | Usage|
| ------------- | ------------- |
| **"models_list"**  | List of base models.<br/> |
| **"scoring_metric"**  | Metric to optimize.<br/>  |
| **"problem_type"**  | Equals "classification" or "regression".<br/>  |
| **"direction"**  | Equals "maximize" or "minimize".<br/>  |
| **"voting_type"**  | Voting type 'soft' or 'hard'.<br/>  |
| **"weights_list"**  | Weights to test out type list of integers or floats ex. [0,1,2,3,...] .<br/>  |


Common parameters for the fit method:<br/>
| Parameter  | Usage|
| ------------- | ------------- |
| **"X"** | Training dataset without target variable.<br/>  |
| **"y"** | target variable.<br/>  |
| **"n_iters"** | Number of trials to execute optimization.<br/>  |
| **"N_folds"** | Number of folds for cross validation.<br/> |
| **"stratify"** | Stratify cv splits based on target distribuition [True or False]<br/>  |
