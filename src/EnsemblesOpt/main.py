from numpy import vstack
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier,VotingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

class Bayesian_Voting_Ensemble:
    
    def __init__(self,ensemble_size,list_classifiers,xi,random_init_points,scoring,maximize_obj,task):
        self.size_problem=ensemble_size
        self.list_classifiers=list_classifiers
        self.xi=xi
        self.random_init_points=random_init_points
        self.maximize_obj=maximize_obj
        self.db=dict()
        self.load_dict_classifiers(self.list_classifiers)
        self.space_dim=len(self.db)-1
        self.scoring=scoring
        self.task=task
        self.points_done=[]
        self.points_vs=dict()
        self.counter=0
        
    def fit(self,X_train,y_train,n_iters,Nfold,stratify=False):
        self.X_train=X_train
        self.y_train=y_train
        self.stratify=stratify
        self.Nfold=Nfold
        init_points=[]
        
       
      
        for i in range(self.size_problem):
            Xi=np.random.randint(0,self.space_dim,self.random_init_points)
            init_points.append(Xi)

        print('Collecting initial random points...')
        y=self.objective(init_points)
        print('Searching best ensemble...')
        X=pd.DataFrame(init_points).transpose()


        #sns.relplot(init_points[0],init_points[1])
        #plt.show()
        y = np.asarray(y).reshape(len(y), 1)

        #print(y)

        model = GaussianProcessRegressor()
        #model=RandomForestRegressor()
        model.fit(X, y)

        for i in range(10000):
            
            x = self.opt_acquisition(X, y, model,size_problem=self.size_problem,maximize=self.maximize_obj,xi=self.xi)

            next_p=[np.round(j) for j in x]
            if next_p not in self.points_done:
                actual = self.objective(next_p)
                self.points_done.append(next_p)
                self.points_vs[tuple(next_p)]=actual
                print('-trial ',self.counter,'|Score value:', actual[0])
                self.counter+=1
            else:
                actual=self.points_vs[tuple(next_p)]
               
            est, _ = self.surrogate(model, [next_p])
            
        
            X = vstack((X, [x]))          
            y = vstack((y, [[actual[0]]]))
           
            model.fit(X, y)
            if self.counter==n_iters:
                break

        if self.size_problem==2:
            self.plot(model)
            sns.relplot(X[:,0],X[:,1],hue=[x[0] for x in y]).set(title='Search Points')
            plt.show() 
        if self.maximize_obj==True:
            ix = np.argmax(y)
        else:
            ix = np.argmin(y)
        best_ensemble=[self.db[np.round(i)] for i in X[ix]]
        print('Best Ensemble:\n',best_ensemble,'\nbest score',y[ix][0])
        return best_ensemble

    
    def load_dict_classifiers(self,list_classifiers):
        for i in range(len(list_classifiers)):
            self.db[i]=self.list_classifiers[i]
        #print(self.db)
        return self.db
    
    def objective(self,X):
        rsults=[]
        counter=0
        for i in np.dstack(X)[0]:
            voting_stack=[]
            for j in i:
                #print(i, j)
                voting_stack.append(('C'+str(j)+'_'+str(counter),self.db[int(j)]))
                counter+=1
            if self.task=='classification':
                eclf1 = VotingClassifier(estimators=voting_stack,voting='soft')
            else:
                eclf1 = VotingRegressor(estimators=voting_stack)
            
            if self.stratify==True:
                cv =StratifiedKFold(n_splits=2,shuffle=True)
            else:
                cv=self.Nfold
            scores=cross_val_score(eclf1,self.X_train,self.y_train, scoring=self.scoring, cv=cv, n_jobs=-1,error_score="raise")  
            err=np.mean(scores)
            rsults.append(err)
        return rsults
    
    def surrogate(self,model, X):
        with catch_warnings():
            simplefilter("ignore")
            return model.predict(X, return_std=True)
    
    def acquisition(self,X, Xsamples, model,xi):
    
        yhat, _ = self.surrogate(model, X)
        best = min(yhat)
        mu, std = self.surrogate(model, Xsamples)
        #print(mu,best)
        probs = norm.cdf((mu - best - self.xi) / (std))
        return probs
 
    # optimize the acquisition function
    def opt_acquisition(self,X, y, model,size_problem=2,xi=0.01,maximize=True):
        # random search, generate random samples

        Xsamples=[]
        for i in range(size_problem):
            Xi=np.random.random(500)*self.space_dim
            Xsamples.append(Xi)
        Xsamples=pd.DataFrame(Xsamples).transpose()

        scores = self.acquisition(X, Xsamples, model,xi)
        if maximize==True:
            ix = np.argmax(scores)
        else:
            ix = np.argmin(scores)

        return Xsamples.iloc[ix]

    def plot(self,model):
        Xsamples=[]
        for i in range(self.size_problem):
            Xi=np.random.random(500)*self.space_dim
            Xsamples.append(Xi)
        Xsamples=pd.DataFrame(Xsamples).transpose()
        ysamples, _ = self.surrogate(model, Xsamples)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax = plt.axes(projection='3d')
        ax.plot_trisurf(Xsamples.iloc[:,0],  Xsamples.iloc[:,1], ysamples,cmap='summer', edgecolor='none',linewidth=0.0,antialiased=True,alpha=0.8)
        ax.set_title('Surrogate function')
        plt.show()
        
    