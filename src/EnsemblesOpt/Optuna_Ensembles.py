from sklearn.ensemble import StackingClassifier,StackingRegressor
import optuna
from sklearn.ensemble import VotingClassifier,VotingRegressor
from itertools import product
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold,KFold
import numpy as np
from sklearn.tree import ExtraTreeClassifier,ExtraTreeRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.linear_model import SGDClassifier,SGDRegressor
from sklearn.linear_model import PassiveAggressiveClassifier,PassiveAggressiveRegressor  
from sklearn.ensemble import AdaBoostClassifier,AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.ensemble import BaggingClassifier,BaggingRegressor
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier,XGBRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
from catboost import CatBoostClassifier,CatBoostRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis




class Optuna_StackEnsemble_Search:
    def __init__(self,scoring_metric,direction,problem_type,size_stack=3,models_list=[],meta_classifier=LogisticRegression()):
        self.scoring_metric=scoring_metric
        self.problem_type=problem_type
        self.size_stack=size_stack
        self.models_list=models_list
        self.direction=direction
        self.meta_classifier=meta_classifier
        self.Regressors=[
             ExtraTreeRegressor(),
             DecisionTreeRegressor(),
             MLPRegressor(),
             KNeighborsRegressor(),
             XGBRegressor(),
             AdaBoostRegressor(),
             GradientBoostingRegressor(),
             BaggingRegressor(),
             ExtraTreesRegressor(),
             RandomForestRegressor(),
             LinearDiscriminantAnalysis(),
             LogisticRegression(),
             LGBMRegressor(),
             CatBoostRegressor(),
             ]
        self.Classifiers=[
             ExtraTreeClassifier(),
             DecisionTreeClassifier(),
             MLPClassifier(),
             #RadiusNeighborsClassifier(),
             KNeighborsClassifier(),
             CalibratedClassifierCV(base_estimator=SGDClassifier(class_weight='balanced'), method='sigmoid'),
             XGBClassifier(),
             CalibratedClassifierCV(base_estimator=PassiveAggressiveClassifier(), method='sigmoid'),
             AdaBoostClassifier(),
             GradientBoostingClassifier(),
             BaggingClassifier(),
             ExtraTreesClassifier(),
             RandomForestClassifier(),
             BernoulliNB(),
             CalibratedClassifierCV(),
             GaussianNB(),
             #LinearDiscriminantAnalysis(),
             LogisticRegression(),
             LGBMClassifier(),
             CatBoostClassifier()]
                
                
        

    def objective_Stacking(self,trial,X,y,N_folds,models_tps,stratify):

        param={}
        estims=[]
        for i in range(self.size_stack):
            name='estimator'+str(i+1)
            #print(name)
            param['estimator'+str(i+1)]=trial.suggest_categorical(name, models_tps)
            estims.append((name,param[name][1]))
        
        if not self.meta_classifier:
            param['meta_learner']=trial.suggest_categorical('meta_learner', models_tps)
        

        
        if stratify==True:
            cv=StratifiedKFold(n_splits=N_folds,shuffle=True)
        else:
            cv=KFold(n_splits=N_folds,shuffle=True)


        if self.problem_type=='classification':
            if self.meta_classifier:
                scores=cross_val_score(StackingClassifier(estimators=estims,final_estimator=self.meta_classifier),
                                X,y,scoring=self.scoring_metric, error_score="raise",cv=cv, n_jobs=-1)
            else:
                scores=cross_val_score(StackingClassifier(estimators=estims,final_estimator=param['meta_learner'][1]),
                                X,y,scoring=self.scoring_metric, error_score="raise",cv=cv, n_jobs=-1)
                
            
        else:
            if self.meta_classifier:
                scores=cross_val_score(StackingRegressor(estimators=estims,final_estimator=self.meta_classifier),X,y,error_score="raise",
                                scoring=self.scoring_metric, cv=cv, n_jobs=-1)
            else:
                scores=cross_val_score(StackingRegressor(estimators=estims,final_estimator=param['meta_learner'][1]),X,y,error_score="raise",
                                scoring=self.scoring_metric, cv=cv, n_jobs=-1)
            
        return np.mean(scores)
        
            
            
            
    def fit(self,X,y,
            n_trials,
            N_folds=3,
            stratify=False):

        if len(self.models_list)>0:
            models_tps=[(x,y) for x,y in zip(['C'+str(i) for i in range(len(self.models_list))],self.models_list)]
        elif len(self.models_list)==0 and self.problem_type=='classification':
            models_tps=[(x,y) for x,y in zip(['C'+str(i) for i in range(len(self.Classifiers))],self.Classifiers)]
        elif len(self.models_list)==0 and self.problem_type!='classification':
            models_tps=[(x,y) for x,y in zip(['C'+str(i) for i in range(len(self.Regressors))],self.Regressors)]
            
        func = lambda trial: self.objective_Stacking(trial,X,y,N_folds,models_tps,stratify)
        
        global study_stacking
        study_stacking = optuna.create_study(direction=self.direction)
        study_stacking.optimize(func, n_trials=n_trials)

        print("Number of finished trials: ", len(study_stacking.trials))
        print("Best trial:")
        trial = study_stacking.best_trial

        print("  Value: {}".format(trial.value))
        print("  Params: ")
        best_stack=[]
        for key, value in trial.params.items():
            best_stack.append(value)
            print("    {}: {}".format(key, value))
            
        if self.problem_type=='classification':
            return StackingClassifier(estimators=best_stack,final_estimator=self.meta_classifier),study_stacking
        else:
            return StackingRegressor(estimators=best_stack,final_estimator=self.meta_classifier),study_stacking


#///////////////////////////////////// VOTING ///////////////////////////////////////////////////////////////////

class Optuna_VotingEnsemble_Search:
    def __init__(self,scoring_metric,direction,problem_type,ensemble_size=3,voting_type=None,models_list=[]):
        self.scoring_metric=scoring_metric
        self.direction=direction
        self.problem_type=problem_type
        self.size_stack=ensemble_size
        self.voting_type=voting_type
        self.models_list=models_list
        self.Regressors=[
             ExtraTreeRegressor(),
             DecisionTreeRegressor(),
             MLPRegressor(),
             KNeighborsRegressor(),
             XGBRegressor(),
             AdaBoostRegressor(),
             GradientBoostingRegressor(),
             BaggingRegressor(),
             ExtraTreesRegressor(),
             RandomForestRegressor(),
             LinearDiscriminantAnalysis(),
             LogisticRegression(),
             LGBMRegressor(),
             CatBoostRegressor(),
             ]
        self.Classifiers=[
             ExtraTreeClassifier(),
             DecisionTreeClassifier(),
             MLPClassifier(),
             #RadiusNeighborsClassifier(),
             KNeighborsClassifier(),
             CalibratedClassifierCV(base_estimator=SGDClassifier(class_weight='balanced'), method='sigmoid'),
             XGBClassifier(),
             CalibratedClassifierCV(base_estimator=PassiveAggressiveClassifier(), method='sigmoid'),
             AdaBoostClassifier(),
             GradientBoostingClassifier(),
             BaggingClassifier(),
             ExtraTreesClassifier(),
             RandomForestClassifier(),
             BernoulliNB(),
             CalibratedClassifierCV(),
             GaussianNB(),
             #LinearDiscriminantAnalysis(),
             LogisticRegression(),
             LGBMClassifier(),
             CatBoostClassifier()]


    def objective_Voting(self,trial,X,y,
                        models_tps,
                        N_folds,
                        stratify):

        param={}
        estims=[]
        for i in range(self.size_stack):
            name='estimator'+str(i+1)
            #print(name)
            param['estimator'+str(i+1)]=trial.suggest_categorical(name, models_tps)
            estims.append((name,param[name][1]))
        

        
        if stratify==True:
            cv=StratifiedKFold(n_splits=N_folds,shuffle=True)
        else:
            cv=KFold(n_splits=N_folds,shuffle=True)


        if self.problem_type=='classification':
            scores=cross_val_score(VotingClassifier(estimators=estims,voting=self.voting_type),
                                X,y,scoring=self.scoring_metric, error_score="raise",cv=cv, n_jobs=-1)
            
        else:
            scores=cross_val_score(VotingRegressor(estimators=estims),X,y,error_score="raise",
                                scoring=self.scoring_metric, cv=cv, n_jobs=-1)
            
        return np.mean(scores)
        
            
            
            
    def fit(self,X,y,
                    n_trials,
                    N_folds=3,
                    stratify=False):

        if len(self.models_list)>0:
            models_tps=[(x,y) for x,y in zip(['C'+str(i) for i in range(len(self.models_list))],self.models_list)]
        elif len(self.models_list)==0 and self.problem_type=='classification':
            models_tps=[(x,y) for x,y in zip(['C'+str(i) for i in range(len(self.Classifiers))],self.Classifiers)]
        elif len(self.models_list)==0 and self.problem_type!='classification':
            models_tps=[(x,y) for x,y in zip(['C'+str(i) for i in range(len(self.Regressors))],self.Regressors)]
            
        func = lambda trial: self.objective_Voting(trial,X,y,models_tps,N_folds,stratify)
        
        global study_voting
        study_voting = optuna.create_study(direction=self.direction)
        study_voting.optimize(func, n_trials=n_trials)

        print("Number of finished trials: ", len(study_voting.trials))
        print("Best trial:")
        trial = study_voting.best_trial

        print("  Value: {}".format(trial.value))
        print("  Params: ")
        best_stack=[]
        for key, value in trial.params.items():
            best_stack.append(value)
            print("    {}: {}".format(key, value))
            
        if self.problem_type=='classification':
            return VotingClassifier(estimators=best_stack,voting=self.voting_type),study_voting
        else:
            return VotingRegressor(estimators=best_stack),study_voting



#///////////////////////////////////////////// WEIGHTS TUNING ////////////////////////////////////////



class Optuna_Voting_weights_tuner:

    def __init__(self,scoring_metric,direction,problem_type,models_list,voting_type=None,weights_list=[1,2,3]):
        self.scoring_metric=scoring_metric
        self.direction=direction
        self.problem_type=problem_type
        self.models_list=models_list
        self.voting_type=voting_type
        self.weights_list=weights_list

    def objective_Voting_weights(self,trial,X,y,models_tps,N_folds,stratify,weights_l):

        param={'weights':trial.suggest_categorical('weights', weights_l)}
        #param['weights']=trial.suggest_categorical('weights', weights_sys)
        
        #print(param['weights'])
        
        if stratify==True:
            cv=StratifiedKFold(n_splits=N_folds,shuffle=True)
        else:
            cv=KFold(n_splits=N_folds,shuffle=True)


        if self.problem_type=='classification':
            scores=cross_val_score(VotingClassifier(estimators=models_tps,voting=self.voting_type,weights=param['weights']),
                                X,y,scoring=self.scoring_metric, error_score="raise",cv=cv, n_jobs=-1)
            
        else:
            scores=cross_val_score(VotingRegressor(estimators=models_tps,weights=param['weights']),X,y,error_score="raise",
                                scoring=self.scoring_metric, cv=cv, n_jobs=-1)
            
        return np.mean(scores)
        
    def stack_best_weights(self,X,y,
                    n_trials,
                    N_folds=3,
                    stratify=False):

        if len(self.models_list)>0:
            models_tps=[(x,y) for x,y in zip(['C'+str(i) for i in range(len(self.models_list))],self.models_list)]
        
        p = product(self.weights_list,repeat=len(models_tps))
        weights_l=[list(x) for x in p]    
        func = lambda trial: self.objective_Voting_weights(trial,X,y,
                                                models_tps,
                                                N_folds,
                                                stratify,
                                                weights_l)
        
        global study_voting_weights
        study_voting_weights = optuna.create_study(direction=self.direction)
        study_voting_weights.optimize(func, n_trials=n_trials)

        print("Number of finished trials: ", len(study_voting_weights.trials))
        print("Best trial:")
        trial = study_voting_weights.best_trial

        print("  Value: {}".format(trial.value))
        print("  Params: ")
    
        for key, value in trial.params.items():
            
            print("    {}: {}".format(key, value))
            
        if self.problem_type=='classification':
            return VotingClassifier(estimators=self.models_list,voting=self.voting_type,weights=study_voting_weights.best_params),study_voting_weights
        else:
            return VotingRegressor(estimators=self.models_list,weights=study_voting_weights.best_params),study_voting_weights
            

