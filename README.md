# EnsemblesOpt 
<br/>



<p align="middle">
  
  <img src="https://user-images.githubusercontent.com/62545181/193418013-ffe15020-43cb-40a6-b230-a8c6136c13a8.gif" width="400" />
  <img src="https://user-images.githubusercontent.com/62545181/193418034-4cdb4aab-0d6d-410b-a648-841999caf560.gif" width="400" /> 
</p>

This repository contains the project for a package for speeding up the process of finding best base learners for building ensemble models trough Bayesian Optimization using Gaussian Processes as surrogate function and Expected Improvement as acquisition function, along optimization routines developed using Optuna library.<br/>
The black-box function is defined as the n cross-validation score of the chosen evaluation metric for the ensemble considered during the iteration. Each base model is mapped to an integer value and their combination is passed to the objective function for evaluation.

Install by running:

```
!pip install EnsemblesOpt
```

Code snippet:

```
#import the base models from where to search for the best ensemble of a given size

from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
```


```
#initialize the Bayesian_Voting_Ensemble
from EnsemblesOpt import Bayesian_Voting_Ensemble


BS=Bayesian_Voting_Ensemble(ensemble_size=2,
                            list_classifiers=[ExtraTreeClassifier(),
                                             DecisionTreeClassifier(),
                                             MLPClassifier(),
                                             SGDClassifier(),
                                             KNeighborsClassifier()],
                           xi=0.05,
                           random_init_points=7,
                           maximize_obj=True,
                           scoring='roc_auc',
                           task='classification')
                           
#fit the Bayesian_Voting_Ensemble                         
BS.fit(X,y,
       Nfold=5,
       n_iters=50,
       stratify=True)
       
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
-**"ensemble_size"**: number of base estimators to build the ensemble, the bigger the ensemble the more time consuming and complex the final model will be.<br/>
-**"list_classifiers"**: list of classifiers.<br/>
-**"xi"**: Exploration parameter, higher values lead to more explorative behaviour and viceversa for lower value (default xi=0.01) .<br/>
-**"random_init_points"**: number of initial points to take from the objective function.<br/>
-**"maximize_obj"**: whether to maximize or minimize the objective function [True or False].<br/>
-**"scoring"**: metric to optimize.<br/>
-**"task"**:"classification" or "regression".<br/>

Common parameters for the fit method:<br/>
-**"X"**: training dataset without target variable.<br/>
-**"y"**: target variable.<br/>
-**"n_iters"**: number of trials to execute optimization.<br/>
-**"N_folds"**: number of folds for cross validation.<br/>
-**"stratify"**: stratify cv splits based on target distribuition [True or False]<br/>

The 'scoring' parameter takes the same values from sklearn API (link of available list: https://scikit-learn.org/stable/modules/model_evaluation.html)




