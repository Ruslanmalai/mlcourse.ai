import pandas as pd

submission_1 = pd.read_csv("./sumbmission_cat1_f7.csv") 
submission_2 = pd.read_csv("./submission_lr1_f7.csv") 
submission_3 = pd.read_csv("./sumbmission_lgb1_f7.csv") 

submission = pd.DataFrame()
submission['match_id_hash'] = submission_1['match_id_hash']
submission['radiant_win_prob'] =  0.5*submission_1['radiant_win_prob']+ \
                                  0.25*submission_2['radiant_win_prob']+ \
                                  0.25*submission_3['radiant_win_prob']

submission.to_csv('submission_blend.csv',index=False)
