import numpy as np
import pandas as pd

from plotly import tools
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import xgboost as xgb
import os
import json
from tqdm import tqdm_notebook

X_train = pd.read_pickle("./train_features_7.pkl")
X_test = pd.read_pickle("./test_features_7.pkl")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_df = pd.DataFrame(data = X_train_scaled)
X_test_df = pd.DataFrame(data = X_test_scaled)

model = LogisticRegression(random_state=42, solver='liblinear')

    
n_fold = 5
folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)

params = []

PATH_TO_DATA = ''

df_train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, 
                                            'train_targets.csv'), 
                                   index_col='match_id_hash')

y = df_train_targets['radiant_win'].values


df_test_features = pd.read_csv(os.path.join(PATH_TO_DATA, 
                                             'test_features.csv'), 
                                    index_col='match_id_hash')


SEED = 17

def logit_cv(X_heroes_train, y_train, cv=5, random_state=SEED):
    logit = LogisticRegression(random_state=SEED, solver='liblinear')

    c_values = np.logspace(-2, 1, 20)

    logit_grid_searcher = GridSearchCV(estimator=logit, param_grid={'C': c_values},
                                       scoring='roc_auc',return_train_score=False, cv=cv,
                                       n_jobs=-1, verbose=0)

    logit_grid_searcher.fit(X_heroes_train, y_train)
    
    cv_scores = []
    for i in range(logit_grid_searcher.n_splits_):
        cv_scores.append(logit_grid_searcher.cv_results_[f'split{i}_test_score'][logit_grid_searcher.best_index_])
    print(f'CV scores: {cv_scores}')
    print(f'Mean: {np.mean(cv_scores)}, std: {np.std(cv_scores)}\n')
    
    return logit_grid_searcher.best_estimator_, np.array(cv_scores)

logit_1, cv_scores_1 = logit_cv(X_train_scaled, y, cv=folds, random_state=SEED)

logit_1.fit(X_train_scaled, y)
df_submission = pd.DataFrame(
    {'radiant_win_prob': logit_1.predict_proba(X_test_scaled)[:, 1]}, 
    index=df_test_features.index,)

df_submission.to_csv('submission_lr1_f7.csv')