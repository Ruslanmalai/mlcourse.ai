import pandas as pd
import os

PATH_TO_DATA = ''
#df_train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, 
#                                            'train_targets.csv'), 
#                                   index_col='match_id_hash')

real_dataset = pd.read_pickle("./train_features_7.pkl")
real_dataset = real_dataset.iloc[:50]

testing_dataset = pd.read_pickle("./train_features_77.pkl")


print(real_dataset = testing_dataset)
