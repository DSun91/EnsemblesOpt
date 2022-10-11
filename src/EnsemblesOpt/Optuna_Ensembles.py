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
import warnings

warnings.filterwarnings("ignore", category=UserWarning)



class Optuna_StackEnsemble_Search:
    def __init__(self,scoring_metric,direction,problem_type,meta_learner,size_stack=3,models_list=[],n_avg=1):
        self.scoring_metric=scoring_metric
        self.problem_type=problem_type
        self.size_stack=size_stack
        self.models_list=models_list
        self.direction=direction
        self.meta_learner=meta_learner
        self.n_avg=n_avg
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
        
        if not self.meta_learner:
            param['meta_learner']=trial.suggest_categorical('meta_learner', models_tps)
        

        
        if stratify==True:
            cv=StratifiedKFold(n_splits=N_folds,shuffle=True)
        else:
            cv=KFold(n_splits=N_folds,shuffle=True)


        if self.problem_type=='classification':
            if self.meta_learner:
                temp=[]
                for i in range(self.n_avg):
                    scores=cross_val_score(StackingClassifier(estimators=estims,final_estimator=self.meta_learner),
                                X,y,scoring=self.scoring_metric, error_score="raise",cv=cv, n_jobs=-1)
                    temp.append(np.mean(scores))
            else:
                temp=[]
                for i in range(self.n_avg):
                    scores=cross_val_score(StackingClassifier(estimators=estims,final_estimator=param['meta_learner'][1]),
                                X,y,scoring=self.scoring_metric, error_score="raise",cv=cv, n_jobs=-1)
                    temp.append(np.mean(scores))
                
            
        else:
            if self.meta_learner:
                temp=[]
                for i in range(self.n_avg):
                    scores=cross_val_score(StackingRegressor(estimators=estims,final_estimator=self.meta_learner),X,y,error_score="raise",
                                scoring=self.scoring_metric, cv=cv, n_jobs=-1)
                    temp.append(np.mean(scores))
            else:
                temp=[]
                for i in range(self.n_avg):
                    scores=cross_val_score(StackingRegressor(estimators=estims,final_estimator=param['meta_learner'][1]),X,y,error_score="raise",
                                scoring=self.scoring_metric, cv=cv, n_jobs=-1)
                    temp.append(np.mean(scores))
            
        return np.mean(temp)
        
            
            
            
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
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study_stacking.optimize(func, n_trials=n_trials)

        print("Number of finished trials: ", len(study_stacking.trials))
        print("Best trial:")
        trial = study_stacking.best_trial

        print("  Value: {}".format(trial.value))
        print("  Params: ")
        best_stack=[]
        counter=0
        for _,value in trial.params.items():
            name='C'+str(counter)
            best_stack.append([name,value[1]])
            print("    {}: {}".format(counter, value))
            counter+=1
            
        if self.problem_type=='classification':
            return StackingClassifier(estimators=best_stack,final_estimator=self.meta_learner),study_stacking
        else:
            return StackingRegressor(estimators=best_stack,final_estimator=self.meta_learner),study_stacking


#///////////////////////////////////// VOTING ///////////////////////////////////////////////////////////////////

class Optuna_VotingEnsemble_Search:
    def __init__(self,scoring_metric,direction,problem_type,ensemble_size=3,voting_type=None,models_list=[],n_avg=1):
        self.scoring_metric=scoring_metric
        self.direction=direction
        self.problem_type=problem_type
        self.size_stack=ensemble_size
        self.voting_type=voting_type
        self.models_list=models_list
        self.n_avg=n_avg
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
            param[name]=trial.suggest_categorical(name, models_tps)
            estims.append((name,param[name][1]))
        

        
        if stratify==True:
            cv=StratifiedKFold(n_splits=N_folds,shuffle=True)
        else:
            cv=KFold(n_splits=N_folds,shuffle=True)


        if self.problem_type=='classification':
            temp=[]
            for i in range(self.n_avg):
                scores=cross_val_score(VotingClassifier(estimators=estims,voting=self.voting_type),
                                X,y,scoring=self.scoring_metric, error_score="raise",cv=cv, n_jobs=-1)
                temp.append(np.mean(scores))
            
        else:
            temp=[]
            for i in range(self.n_avg):
                scores=cross_val_score(VotingRegressor(estimators=estims),X,y,error_score="raise",
                                scoring=self.scoring_metric, cv=cv, n_jobs=-1)
                temp.append(np.mean(scores))
            
        return np.mean(temp)
        
            
            
            
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
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study_voting.optimize(func, n_trials=n_trials)

        print("Number of finished trials: ", len(study_voting.trials))
        print("Best trial:")
        trial = study_voting.best_trial

        print("  Value: {}".format(trial.value))
        print("  Params: ")
        best_stack=[]
        counter=0
        for _,value in trial.params.items():
            name='C'+str(counter)
            best_stack.append([name,value[1]])
            print("    {}: {}".format(counter, value))
            counter+=1
            
        if self.problem_type=='classification':
            return VotingClassifier(estimators=best_stack,voting=self.voting_type),study_voting
        else:
            return VotingRegressor(estimators=best_stack),study_voting



#///////////////////////////////////////////// WEIGHTS TUNING ////////////////////////////////////////



class Optuna_Voting_weights_tuner:

    def __init__(self,scoring_metric,direction,problem_type,models_list,voting_type=None,weights_list=[1,2,3],n_avg=1):
        self.scoring_metric=scoring_metric
        self.direction=direction
        self.problem_type=problem_type
        self.models_list=models_list
        self.voting_type=voting_type
        self.weights_list=weights_list
        self.n_avg=n_avg

    def objective_Voting_weights(self,trial,X,y,models_tps,N_folds,stratify,weights_l):

        param={'weights':trial.suggest_categorical('weights', weights_l)}
        #param['weights']=trial.suggest_categorical('weights', weights_sys)
        
        #print(param['weights'])
        
        if stratify==True:
            cv=StratifiedKFold(n_splits=N_folds,shuffle=True)
        else:
            cv=KFold(n_splits=N_folds,shuffle=True)


        if self.problem_type=='classification':
            temp=[]
            for i in range(self.n_avg):
                scores=cross_val_score(VotingClassifier(estimators=models_tps,voting=self.voting_type,weights=param['weights']),
                                X,y,scoring=self.scoring_metric, error_score="raise",cv=cv, n_jobs=-1)
                temp.append(np.mean(scores))
            
        else:
            temp=[]
            for i in range(self.n_avg):
                scores=cross_val_score(VotingRegressor(estimators=models_tps,weights=param['weights']),X,y,error_score="raise",
                                scoring=self.scoring_metric, cv=cv, n_jobs=-1)
                temp.append(np.mean(scores))
            
        return np.mean(temp)
        
    def fit(self,X,y,
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
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study_voting_weights.optimize(func, n_trials=n_trials)

        print("Number of finished trials: ", len(study_voting_weights.trials))
        print("Best trial:")
        trial = study_voting_weights.best_trial

        print("  Value: {}".format(trial.value))
        print("  Params: ")
    
        for key, value in trial.params.items():
            
            print("    {}: {}".format(key, value))
        
        estims=[]
        counter=0
        for i in self.models_list:
            estims.append(['C'+str(counter),i])
            counter+=1
            
        if self.problem_type=='classification':
            return VotingClassifier(estimators=estims,voting=self.voting_type,weights=study_voting_weights.best_params['weights']),study_voting_weights
        else:
            return VotingRegressor(estimators=estims,weights=study_voting_weights.best_params['weights']),study_voting_weights
            
            

