import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from plotly import tools
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import warnings
warnings.filterwarnings("ignore")
#init_notebook_mode(connected=True)

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn import decomposition
import lightgbm as lgb
import xgboost as xgb

import os

import json
from tqdm import tqdm_notebook

# creating F4

def read_matches(matches_file):   # Read json file
    
    MATCHES_COUNT = {
        'test_matches.jsonl': 10000,
        'train_matches.jsonl': 39675, 
    }
    _, filename = os.path.split(matches_file)
    total_matches = MATCHES_COUNT.get(filename)
    
    with open(matches_file) as fin:
        for line in tqdm_notebook(fin, total=total_matches):
            yield json.loads(line)
            
            
import collections

MATCH_FEATURES = [
    ('game_time', lambda m: m['game_time']),
    ('game_mode', lambda m: m['game_mode']),
    ('lobby_type', lambda m: m['lobby_type']),
    ('objectives_len', lambda m: len(m['objectives'])),
    ('chat_len', lambda m: len(m['chat'])),
]

PLAYER_FIELDS = [
    'hero_id',
    
    'kills',
    'deaths',
    'assists',
    'denies',
    
    'gold',
    'lh',
    'xp',
    'health',
    'max_health',
    'max_mana',
    'level',

    'x',
    'y',
    
    'stuns',
    'creeps_stacked',
    'camps_stacked',
    'rune_pickups',
    'firstblood_claimed',
    'teamfight_participation',
    'towers_killed',
    'roshans_killed',
    'obs_placed',
    'sen_placed',
]

def extract_features_csv(match):
    row = [
        ('match_id_hash', match['match_id_hash']),
    ]
    
    for field, f in MATCH_FEATURES:
        row.append((field, f(match)))
        
    for slot, player in enumerate(match['players']):
        if slot < 5:
            player_name = 'r%d' % (slot + 1)
        else:
            player_name = 'd%d' % (slot - 4)

        for field in PLAYER_FIELDS:
            column_name = '%s_%s' % (player_name, field)
            row.append((column_name, player[field]))
        row.append((f'{player_name}_ability_level', len(player['ability_upgrades'])))
        row.append((f'{player_name}_max_hero_hit', player['max_hero_hit']['value']))
        row.append((f'{player_name}_purchase_count', len(player['purchase_log'])))
        row.append((f'{player_name}_count_ability_use', sum(player['ability_uses'].values())))
        row.append((f'{player_name}_damage_dealt', sum(player['damage'].values())))
        row.append((f'{player_name}_damage_received', sum(player['damage_taken'].values())))
            
    return collections.OrderedDict(row)

    
def extract_targets_csv(match, targets):
    return collections.OrderedDict([('match_id_hash', match['match_id_hash'])] + [
        (field, targets[field])
        for field in ['game_time', 'radiant_win', 'duration', 'time_remaining', 'next_roshan_team']
    ])


#PATH_TO_DATA = '../../data/dota_2/'
PATH_TO_DATA = r'C:\Users\Xianqin\Notebook_files\mlcourse.ai\data\dota_2'
df_new_features = []
df_new_targets = []


for match in read_matches(os.path.join(PATH_TO_DATA, 'train_matches.jsonl')):
    match_id_hash = match['match_id_hash']
    features = extract_features_csv(match)
    targets = extract_targets_csv(match, match['targets'])    
    df_new_features.append(features)
    df_new_targets.append(targets)
    
    
df_new_features = pd.DataFrame.from_records(df_new_features).set_index('match_id_hash')
df_new_targets = pd.DataFrame.from_records(df_new_targets).set_index('match_id_hash')

test_new_features = []
for match in read_matches(os.path.join(PATH_TO_DATA, 'test_matches.jsonl')):
    match_id_hash = match['match_id_hash']
    features = extract_features_csv(match)
    test_new_features.append(features)
test_new_features = pd.DataFrame.from_records(test_new_features).set_index('match_id_hash')


for c in ['kills', 'deaths', 'assists', 'denies', 'gold', 'lh', 'xp', 'health', 'max_health', 'max_mana', 'level', 'x', 'y', 'stuns', 'creeps_stacked', 'camps_stacked', 'rune_pickups',
          'firstblood_claimed', 'teamfight_participation', 'towers_killed', 'roshans_killed', 'obs_placed', 'sen_placed', 'ability_level', 'max_hero_hit', 'purchase_count',
          'count_ability_use', 'damage_dealt', 'damage_received']:
    r_columns = [f'r{i}_{c}' for i in range(1, 6)]
    d_columns = [f'd{i}_{c}' for i in range(1, 6)]
    
    df_new_features['r_total_' + c] = df_new_features[r_columns].sum(1)
    df_new_features['d_total_' + c] = df_new_features[d_columns].sum(1)
    df_new_features['total_' + c + '_ratio'] = df_new_features['r_total_' + c] / df_new_features['d_total_' + c]
    
    test_new_features['r_total_' + c] = test_new_features[r_columns].sum(1)
    test_new_features['d_total_' + c] = test_new_features[d_columns].sum(1)
    test_new_features['total_' + c + '_ratio'] = test_new_features['r_total_' + c] / test_new_features['d_total_' + c]
    
    df_new_features['r_std_' + c] = df_new_features[r_columns].std(1)
    df_new_features['d_std_' + c] = df_new_features[d_columns].std(1)
    df_new_features['std_' + c + '_ratio'] = df_new_features['r_std_' + c] / df_new_features['d_std_' + c]
    
    test_new_features['r_std_' + c] = test_new_features[r_columns].std(1)
    test_new_features['d_std_' + c] = test_new_features[d_columns].std(1)
    test_new_features['std_' + c + '_ratio'] = test_new_features['r_std_' + c] / test_new_features['d_std_' + c]
    
    df_new_features['r_mean_' + c] = df_new_features[r_columns].mean(1)
    df_new_features['d_mean_' + c] = df_new_features[d_columns].mean(1)
    df_new_features['mean_' + c + '_ratio'] = df_new_features['r_mean_' + c] / df_new_features['d_mean_' + c]
    
    test_new_features['r_mean_' + c] = test_new_features[r_columns].mean(1)
    test_new_features['d_mean_' + c] = test_new_features[d_columns].mean(1)
    test_new_features['mean_' + c + '_ratio'] = test_new_features['r_mean_' + c] / test_new_features['d_mean_' + c]
    
    
X = df_new_features.reset_index(drop=True)
X_test = test_new_features.copy().reset_index(drop=True)

#------------------------------------------------
# Creating F5

X_train = X.copy()

def add_new_features(df_features, matches_file):
    
    # Process raw data and add new features
    for match in read_matches(matches_file):
        match_id_hash = match['match_id_hash']

        # Counting ruined towers for both teams
        radiant_tower_kills = 0
        dire_tower_kills = 0
        for objective in match['objectives']:
            if objective['type'] == 'CHAT_MESSAGE_TOWER_KILL':
                if objective['team'] == 2:
                    radiant_tower_kills += 1
                if objective['team'] == 3:
                    dire_tower_kills += 1

        # Write new features
        df_features.loc[match_id_hash, 'radiant_tower_kills'] = radiant_tower_kills
        df_features.loc[match_id_hash, 'dire_tower_kills'] = dire_tower_kills
        df_features.loc[match_id_hash, 'diff_tower_kills'] = radiant_tower_kills - dire_tower_kills
        
        
# copy the dataframe with features
X_train_extended = X_train.copy()

# add new features
add_new_features(X_train_extended, 
                 os.path.join(PATH_TO_DATA, 
                              'train_matches.jsonl'))

X_train_tower_kills = X_train_extended.iloc[39675 : , -3 : ]
#X_train_tower_kills = X_train_extended.iloc[4 : , -3 : ]
X_train_extended_copy = X_train_extended.copy()
X_train_extended_copy.iloc[0:39676, -3:] = X_train_tower_kills
#X_train_extended_copy.iloc[0:5, -3:] = X_train_tower_kills
index_to_drop = list(X_train_extended_copy.iloc[39675:].index)
#index_to_drop = list(X_train_extended_copy.iloc[4:].index)
X_train_extended_copy.drop(index_to_drop, inplace = True)
X_train_extended_copy.iloc[:,-3:] = X_train_tower_kills.values
X_train_f5 = X_train_extended_copy.copy()


X_test_extended = X_test.copy()

add_new_features(X_test_extended, 
                 os.path.join(PATH_TO_DATA, 'test_matches.jsonl'))

X_test_extended.fillna(0)
X_test_extended.iloc[:10000, -3:] = X_test_extended.iloc[10000:, -3:].values
#X_test_extended.iloc[:4, -3:] = X_test_extended.iloc[4:, -3:].values
test_index_to_drop = list(X_test_extended.index)[10000:]
#test_index_to_drop = list(X_test_extended.index)[4:]
X_test_extended.drop(test_index_to_drop, inplace = True)
X_test_f5 = X_test_extended.copy()

#-------------------------------------------------
# Create hero_id_counter

from itertools import combinations
import os
from sklearn.feature_extraction.text import CountVectorizer

#PATH_TO_DATA = '../../data/dota_2/'
PATH_TO_DATA = ''
SEED = 17

# Train dataset
df_train_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_features.csv'), 
                                    index_col='match_id_hash')
df_train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_targets.csv'), 
                                   index_col='match_id_hash')

y_train = df_train_targets['radiant_win'].map({True: 1, False: 0})

# Test dataset
df_test_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_features.csv'), 
                                   index_col='match_id_hash')

df_full_features = pd.concat([df_train_features, df_test_features])

# Index to split the training and test data sets
idx_split = df_train_features.shape[0]

heroes_df = df_full_features[[f'{t}{i}_hero_id' for t in ['r', 'd'] for i in range(1, 6)]]

def bag_of_heroes(df, N=1, r_val=1, d_val=-1, r_d_val=0, return_as='csr'):
    '''
    Bag of Heroes. Returns a csr matrix (+ list of feature names) or dataframe where each column represents
    a hero (ID) and each row represents a match.
    
    The value of a cell (i, j) in the returned matrix is:
        cell[i, j] = 0, if the hero or combination of heroes of the j-th column is not present in the i-th match
        cell[i, j] = r_val, if the hero (N = 1) or combination of heroes (N > 1, synergy) of the j-th column is within the Radiant team,
        cell[i, j] = d_val, if the hero (N = 1) or combination of heroes (N > 1, synergy) of the j-th column is within the Dire team,
        cell[i, j] = r_d_val, if the combination of heroes of the j-th column is between the Radiant and Dire teams (N>1, anti-synergy).
    
    Parameters:
    -----------
        df: dataframe with hero IDs, with columns ['r1_hero_id', ..., 'r5_hero_id', 'd1_hero_id', ..., 'd5_hero_id']
        N: integer 1 <= N <= 10, for N heroes combinations
        return_as: 'csr' for scipy csr sparse matrix, 'df' for pandas dataframe
    '''
    if N < 1 or N > df.shape[1]:
        raise Exception(f'The number N of hero-combinations should be 1 <= N <= {df.shape[1]}')
        
    # Convert the integer IDs to strings of the form id{x}{x}{x}
    df = df.astype(str).applymap(lambda x: 'id' + '0'*(3 - len(x)) + x)
    
    # Create a list of all hero IDs present in df
    hero_ids = np.unique(df).tolist()

    # Break df into teams Radiant (r) and Dire (d)
    df_r = df[[col for col in df.columns if col[0] == 'r']]
    df_d = df[[col for col in df.columns if col[0] == 'd']]
    
    # Create a list of all the hero IDs in df, df_r and df_d respectively
    f = lambda x: ' '.join(['_'.join(c) for c in combinations(sorted(x), N)])
    
    df_list = df.apply(f, axis=1).tolist()
    df_list.append(' '.join(['_'.join(c) for c in combinations(hero_ids, N)]))

    df_r_list = df_r.apply(f, axis=1).tolist()
    df_r_list.append(' '.join(['_'.join(c) for c in combinations(hero_ids, N)]))
    
    df_d_list = df_d.apply(f, axis=1).tolist()
    df_d_list.append(' '.join(['_'.join(c) for c in combinations(hero_ids, N)]))
    
    # Create countvectorizers
    vectorizer = CountVectorizer()
    vectorizer_r = CountVectorizer()
    vectorizer_d = CountVectorizer()
    
    X = vectorizer.fit_transform(df_list)[:-1]
    X_r = vectorizer_r.fit_transform(df_r_list)[:-1]
    X_d = vectorizer_d.fit_transform(df_d_list)[:-1]
    X_r_d = (X - (X_r + X_d))  
    X = (r_val * X_r + d_val * X_d + r_d_val * X_r_d)
    
    feature_names = vectorizer.get_feature_names()
    
    if return_as == 'csr':
        return X, feature_names
    elif return_as == 'df':
        return pd.DataFrame(X.toarray(), columns=feature_names, index=df.index).to_sparse(0)
    
boh = bag_of_heroes(heroes_df, N=1, r_val=1, d_val=-1, return_as='df')

X_heroes_train = boh[:idx_split]
X_heroes_test  = boh[idx_split:]

#----------------------------------------------
# Create final features

df_train_pickle = X_train_f5.copy()
df_test_pickle = X_test_f5.copy()

df_train_pickle[df_train_pickle==np.inf]=np.nan
df_train_pickle = df_train_pickle.fillna(0)
df_test_pickle[df_test_pickle==np.inf]=np.nan
df_test_pickle = df_test_pickle.fillna(0)

df_train_heroes = X_heroes_train.copy()
df_test_heroes = X_heroes_test.copy()

X_train = df_train_pickle.copy()
X_test = df_test_pickle.copy()

heroes_ids = [f'{t}{i}_hero_id' for t in ['r', 'd'] for i in range(1, 6)]

X_train.drop(columns = heroes_ids, inplace = True)
X_test.drop(columns = heroes_ids, inplace = True)

columns = list(X_train.columns)
columns_to_drop = columns[5:295]
X_test.drop(columns = columns_to_drop, inplace = True)

X_train['r_R^2'] = X_train.r_mean_x**2 + X_train.r_mean_y**2
X_train['d_R^2'] = X_train.d_mean_x**2 + X_train.d_mean_y**2
X_test['r_R^2'] = X_test.r_mean_x**2 + X_test.r_mean_y**2
X_test['d_R^2'] = X_test.d_mean_x**2 + X_test.d_mean_y**2
X_train['x_diff'] = np.abs(X_train.r_mean_x - X_train.d_mean_x) 
X_train['y_diff'] = np.abs(X_train.r_mean_y - X_train.d_mean_y) 
X_test['x_diff'] = np.abs(X_test.r_mean_x - X_test.d_mean_x) 
X_test['y_diff'] = np.abs(X_test.r_mean_y - X_test.d_mean_y) 

df_train_heroes.index = X_train.index
df_test_heroes.index = X_test.index

hero_id_columns = list(df_train_heroes.columns)
train_columns = list(X_train.columns)

X_train_matrix = X_train.values
heroes_train_matrix = df_train_heroes.values
X_full_matrix = np.concatenate((X_train_matrix,heroes_train_matrix),axis=1)
all_columns = train_columns + hero_id_columns
X_train_full = pd.DataFrame(data = X_full_matrix, columns = all_columns)

X_test_matrix = X_test.values
heroes_test_matrix = df_test_heroes.values
X_full_test_matrix = np.concatenate((X_test_matrix, heroes_test_matrix),axis=1)
X_test_full = pd.DataFrame(data = X_full_test_matrix, columns = all_columns)

pd.to_pickle(X_train_full, "./train_features_77.pkl") #### DO NOT FORGET TO CHANGE !!!!
pd.to_pickle(X_test_full, "./test_features_77.pkl")   #### DO NOT FORGET TO CHANGE !!!!

X_train_full_cat = X_train_full.copy()
X_test_full_cat = X_test_full.copy()
X_train_full_cat[hero_id_columns].astype('category')
X_test_full_cat[hero_id_columns].astype('category');
pd.to_pickle(X_train_full_cat, "./train_features_7_cat.pkl") #### DO NOT FORGET TO CHANGE !!!!
pd.to_pickle(X_test_full_cat, "./test_features_7_cat.pkl")
